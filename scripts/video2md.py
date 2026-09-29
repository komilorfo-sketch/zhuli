# -*- coding: utf-8 -*-
"""
video2md.py — 本地视频转写管线（一条命令出逐字稿）
流程: URL → yt-dlp 下载音频 → ffmpeg 抽音轨(16k mono wav) → faster-whisper 批量转写 → 逐字稿 .md

用法:
  python video2md.py <视频URL> [输出目录]           # 全量转写(默认)
  python video2md.py --spot <视频URL> [--start 秒] [--dur 秒] [--compare 候选稿.md]
                                                    # 抽头校验: 只转写片段, 与候选逐字稿比对相似度

环境变量:
  V2MD_MODEL   模型档位 tiny/base/small/medium (默认 small, 中文效果好、CPU 可跑)
  V2MD_OUT     默认输出目录 (默认 ./视频转写)
  V2MD_LANG    强制语言 zh/en/ja... (默认 auto 自动检测)
  V2MD_BATCH   批量推理 batch 大小 (默认 8; 0=禁用批量走顺序模式)
  V2MD_VAD     1=开启VAD过滤 0=关闭 (默认 1; 批量模式必须 VAD)
  V2MD_KEEP_VIDEO  1=保留下载的临时文件 (默认 0 清理)

依赖版本注意:
  onnxruntime 必须 == 1.19.2 (1.30.0 在部分 Windows 机器上 VAD 段错误!)
  ctranslate2 4.4.0 (4.5+ 在部分机器段错误)

输出:
  全量模式: <输出目录>/<视频标题>_逐字稿.md (带时间戳分段)
  校验模式: stdout JSON {"ok":true,"text":...,"similarity":0.xx,"verdict":"match|unsure|no_match"}
"""
import os
import re
import sys
import json
import time
import shutil
import tempfile
import subprocess

# ---------- 配置 ----------
FFMPEG = None  # 延迟加载
MODEL_SIZE = os.environ.get("V2MD_MODEL", "small")
OUTPUT_ROOT = os.environ.get("V2MD_OUT", os.path.join(os.getcwd(), "视频转写"))
FORCE_LANG = os.environ.get("V2MD_LANG", "auto")  # auto = 交给 whisper 自动检测
BATCH_SIZE = int(os.environ.get("V2MD_BATCH", "8"))
USE_VAD = os.environ.get("V2MD_VAD", "1") == "1"
KEEP_TMP = os.environ.get("V2MD_KEEP_VIDEO", "0") == "1"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


def get_ffmpeg():
    global FFMPEG
    if FFMPEG is None:
        try:
            import imageio_ffmpeg
            FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            FFMPEG = "ffmpeg"  # 依赖系统安装的 ffmpeg
    return FFMPEG


def safe_name(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\n\r\t]', '_', name).strip()
    return name[:80] if name else "untitled"


def fresh_ttwid_cookie_file(tmpdir: str):
    """通过字节跳动 ttwid 注册接口获取游客 ttwid (无需登录), 用于部分受限站点"""
    import urllib.request
    try:
        req = urllib.request.Request(
            "https://ttwid.bytedance.com/ttwid/union/register/",
            data=json.dumps({
                "region": "cn", "aid": 1768, "needFid": False,
                "service": "www.ixigua.com",
                "migrate_info": {"ticket": "", "source": "node"},
                "cbUrlProtocol": "https", "union": True,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": UA},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            m = re.search(r'ttwid=([^;]+)', resp.headers.get("Set-Cookie", ""))
        if not m:
            return None
        cookie_file = os.path.join(tmpdir, "guest_cookies.txt")
        now = int(time.time())
        lines = ["# Netscape HTTP Cookie File",
                 ".douyin.com\tTRUE\t/\tTRUE\t%d\tTRUE\tttwid\t%s" % (now + 86400 * 30, m.group(1))]
        with open(cookie_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return cookie_file
    except Exception:
        return None


def ydl_opts(tmpdir: str, url: str, download: bool, section=None):
    opts = {
        "quiet": True,
        "no_warnings": True,
        "http_headers": {"User-Agent": UA},
        "noplaylist": True,
    }
    if download:
        opts["outtmpl"] = os.path.join(tmpdir, "source.%(ext)s")
        opts["format"] = "bestaudio/best"
        if section:
            opts["download_ranges"] = __import__("yt_dlp").utils.download_range_func(None, [section])
            opts["force_keyframes_at_cuts"] = True
    if any(d in url for d in ("douyin", "ixigua", "iesdouyin")):
        cf = fresh_ttwid_cookie_file(tmpdir)
        if cf:
            opts["cookiefile"] = cf
    return opts


def probe_info(url: str, tmpdir: str):
    """只取元信息, 不下载"""
    import yt_dlp
    with yt_dlp.YoutubeDL(ydl_opts(tmpdir, url, download=False)) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "title": info.get("title") or "untitled",
        "uploader": info.get("uploader") or info.get("channel") or "",
        "duration": info.get("duration") or 0,
        "webpage_url": info.get("webpage_url") or url,
    }


def download_audio(url: str, tmpdir: str, section=None) -> str:
    """下载最佳音轨; section=(start,end) 时只下载片段"""
    import yt_dlp
    with yt_dlp.YoutubeDL(ydl_opts(tmpdir, url, download=True, section=section)) as ydl:
        ydl.extract_info(url, download=True)
    for fn in sorted(os.listdir(tmpdir)):
        if fn.startswith("source."):
            return os.path.join(tmpdir, fn)
    raise RuntimeError("下载后未找到源文件")


def is_local_file(path: str) -> bool:
    """判断输入是否为本地视频/音频文件 (非 URL 且真实存在)"""
    return "://" not in path and os.path.isfile(path)


def local_file_meta(path: str) -> dict:
    """本地文件的元信息: 文件名作标题, ffmpeg 读时长"""
    abspath = os.path.abspath(path)
    title = os.path.splitext(os.path.basename(abspath))[0]
    dur = 0.0
    try:
        r = subprocess.run([get_ffmpeg(), "-i", abspath], capture_output=True,
                           text=True, encoding="utf-8", errors="ignore")
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+)", r.stderr)
        if m:
            dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    except Exception:
        pass
    return {"title": title, "uploader": "", "duration": dur, "webpage_url": abspath}


def extract_wav(src: str, tmpdir: str, start: float = None, dur: float = None) -> str:
    """任意媒体 → 16kHz mono wav (whisper 最佳输入); 可只取片段"""
    wav = os.path.join(tmpdir, "audio_16k.wav")
    cmd = [get_ffmpeg(), "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if dur is not None:
        cmd += ["-t", str(dur)]
    cmd += ["-i", src, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", wav, "-loglevel", "error"]
    subprocess.run(cmd, check=True)
    if not os.path.exists(wav) or os.path.getsize(wav) < 1000:
        raise RuntimeError("音轨抽取失败: %s" % src)
    return wav


def fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def make_model():
    from faster_whisper import WhisperModel
    return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8",
                        cpu_threads=max(4, os.cpu_count() or 4))


_ZH_PROMPT = "以下是普通话的句子，请用简体中文转写。"


def _transcribe_kwargs():
    kwargs = {}
    if FORCE_LANG != "auto":
        kwargs["language"] = FORCE_LANG
        if FORCE_LANG == "zh":
            kwargs["initial_prompt"] = _ZH_PROMPT  # 引导简体输出, 避免繁体混杂
    return kwargs


def _collect(segments):
    segs = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            segs.append({"start": seg.start, "end": seg.end, "text": text})
    return segs


def transcribe(wav: str):
    """faster-whisper 批量转写(默认, 约快2倍); 批量失败自动降级顺序; 返回 (segments, info)
    中文结果强制转简体(opencc), 不依赖模型自觉"""
    kwargs = _transcribe_kwargs()
    if BATCH_SIZE > 0 and USE_VAD:
        try:
            from faster_whisper import BatchedInferencePipeline
            model = make_model()
            pipe = BatchedInferencePipeline(model=model)
            segments, info = pipe.transcribe(wav, batch_size=BATCH_SIZE,
                                             beam_size=1, vad_filter=True, **kwargs)
            segs = _collect(segments)
            return _maybe_simplify(segs, info), info
        except Exception as e:
            print("[warn] 批量转写失败(%s), 降级顺序模式" % e, flush=True)
    # 顺序模式
    model = make_model()
    seg_kwargs = dict(beam_size=1, vad_filter=USE_VAD)
    seg_kwargs.update(kwargs)
    segments, info = model.transcribe(wav, **seg_kwargs)
    segs = _collect(segments)
    return _maybe_simplify(segs, info), info


def _maybe_simplify(segs: list, info) -> list:
    """检测语言为中文时, 全部繁→简 (硬要求: 交付一律简体中文)"""
    lang = getattr(info, "language", None) or FORCE_LANG
    if lang == "zh" or FORCE_LANG == "zh":
        for s in segs:
            s["text"] = to_simplified(s["text"])
    return segs


# ---------- 校验匹配 ----------
_T2S = None
try:
    from opencc import OpenCC
    _T2S = OpenCC("t2s")
except Exception:
    _T2S = None  # 未装 opencc 时跳过繁简归一


def to_simplified(text: str) -> str:
    """繁→简（opencc 不可用时原样返回）"""
    if _T2S is None:
        return text
    try:
        return _T2S.convert(text)
    except Exception:
        return text


def normalize(text: str) -> str:
    """归一化: 繁→简, 去空白/标点, 英文转小写, 只留文字与数字"""
    text = re.sub(r'\*\*\[[^\]]*\]\*\*', '', text)  # 去时间戳标记
    if _T2S is not None:
        try:
            text = _T2S.convert(text)
        except Exception:
            pass
    text = re.sub(r'[^\w\u4e00-\u9fff]+', '', text.lower())
    return text


def similarity(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def best_window_similarity(spot: str, candidate: str, win: int = None) -> float:
    """spot 文本与候选稿的滑动窗口最大相似度 (窗口略大于 spot 长度, 保证对齐)"""
    win = win or max(200, int(len(spot) * 1.2))
    best = 0.0
    step = max(30, win // 20)
    if len(candidate) <= win:
        return similarity(spot, candidate)
    for i in range(0, len(candidate) - win + 1, step):
        r = similarity(spot, candidate[i:i + win])
        if r > best:
            best = r
            if best > 0.9:
                break
    return best


def write_md(meta: dict, segs: list, out_path: str, elapsed: float, lang: str):
    dur = meta.get("duration") or (segs[-1]["end"] if segs else 0)
    lines = [
        "# %s — 逐字稿" % meta["title"],
        "",
        "> **来源**: %s" % meta["webpage_url"],
    ]
    if meta.get("uploader"):
        lines.append("> **作者**: %s" % meta["uploader"])
    if dur:
        lines.append("> **时长**: %s" % fmt_ts(dur))
    lines += [
        "> **转写**: faster-whisper (%s 模型, 本地批量推理) | 用时 %.1f 分钟 | 语言: %s" % (MODEL_SIZE, elapsed / 60, lang),
        "> **性质**: AI 语音转写逐字稿(简体中文), 口语原样保留, 未做书面化润色; 个别同音字可能有误",
        "",
        "---",
        "",
    ]
    # 按约 60 秒分段落, 段首带时间戳
    para, para_start, para_end = [], None, None
    for s in segs:
        if para_start is None:
            para_start = s["start"]
        para.append(s["text"])
        para_end = s["end"]
        if para_end - para_start >= 60:
            lines.append("**[%s]** %s" % (fmt_ts(para_start), "".join(para)))
            lines.append("")
            para, para_start = [], None
    if para:
        lines.append("**[%s]** %s" % (fmt_ts(para_start or 0), "".join(para)))
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_full(url: str, out_root: str):
    """全量转写主流程"""
    os.makedirs(out_root, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="v2md_")
    try:
        t0 = time.time()
        if is_local_file(url):
            # 本地视频/音频文件: 免下载, 直接抽音轨
            meta = local_file_meta(url)
            print("[1/3] 本地文件: %s (时长 %ss)" % (meta["title"], round(meta["duration"])), flush=True)
            print("[2/3] 抽音轨 (16k mono)...", flush=True)
            wav = extract_wav(url, tmpdir)
        else:
            meta = probe_info(url, tmpdir)
            print("[1/3] 下载音频: %s" % meta["title"], flush=True)
            src = download_audio(url, tmpdir)
            print("[2/3] 抽音轨 (16k mono)...", flush=True)
            wav = extract_wav(src, tmpdir)
        title = safe_name(meta["title"])
        out_path = os.path.join(out_root, title + "_逐字稿.md")
        i = 1
        while os.path.exists(out_path):
            out_path = os.path.join(out_root, "%s_逐字稿(%d).md" % (title, i))
            i += 1

        print("[3/3] whisper 批量转写 (model=%s, batch=%s, vad=%s)..." % (MODEL_SIZE, BATCH_SIZE, USE_VAD), flush=True)
        segs, info = transcribe(wav)
        elapsed = time.time() - t0
        lang = getattr(info, "language", None) or FORCE_LANG
        write_md(meta, segs, out_path, elapsed, lang)
        print(json.dumps({
            "ok": True,
            "output": out_path,
            "summary_input": out_path,  # 逐字稿路径, 供下一步生成学习总结
            "segments": len(segs),
            "duration_sec": round(meta.get("duration") or (segs[-1]["end"] if segs else 0)),
            "language": lang,
            "title": meta["title"],
            "uploader": meta.get("uploader", ""),
            "webpage_url": meta["webpage_url"],
            "elapsed_sec": round(elapsed),
        }, ensure_ascii=False))
    finally:
        if not KEEP_TMP:
            shutil.rmtree(tmpdir, ignore_errors=True)


def run_spot(url: str, start: float, dur: float, compare_file: str = None):
    """抽头校验: 只下载/转写一小段, 与候选逐字稿比对相似度"""
    tmpdir = tempfile.mkdtemp(prefix="v2md_spot_")
    try:
        meta = probe_info(url, tmpdir)
        print("[1/3] 元信息: %s | 时长 %ss | 作者 %s" % (meta["title"], meta["duration"], meta.get("uploader", "")), flush=True)
        # 优先让 yt-dlp 只下载片段 (省流量); 失败则全量下载后 ffmpeg 剪切
        try:
            print("[2/3] 下载片段 [%s-%ss]..." % (start, start + dur), flush=True)
            src = download_audio(url, tmpdir, section=(start, start + dur))
            wav = extract_wav(src, tmpdir)
        except Exception as e:
            print("[2/3] 片段下载失败(%s), 全量下载后剪切..." % e, flush=True)
            src = download_audio(url, tmpdir)
            wav = extract_wav(src, tmpdir, start=start, dur=dur)
        print("[3/3] 转写片段...", flush=True)
        segs, info = transcribe(wav)
        spot_text = normalize("".join(s["text"] for s in segs))
        result = {
            "ok": True,
            "title": meta["title"],
            "uploader": meta.get("uploader", ""),
            "duration_sec": round(meta.get("duration") or 0),
            "spot_start": start,
            "spot_dur": dur,
            "spot_text": "".join(s["text"] for s in segs)[:500],
            "spot_chars": len(spot_text),
        }
        if compare_file:
            with open(compare_file, "r", encoding="utf-8") as f:
                cand = normalize(f.read())
            sim = best_window_similarity(spot_text, cand)
            result["similarity"] = round(sim, 3)
            result["verdict"] = "match" if sim >= 0.50 else ("unsure" if sim >= 0.25 else "no_match")
        print(json.dumps(result, ensure_ascii=False))
    finally:
        if not KEEP_TMP:
            shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    args = sys.argv[1:]
    # 校验模式: --spot <URL> [--start N] [--dur N] [--compare FILE]
    if args and args[0] == "--spot":
        url = args[1]
        start, dur, compare = 0.0, 60.0, None
        i = 2
        while i < len(args):
            if args[i] == "--start":
                start = float(args[i + 1]); i += 2
            elif args[i] == "--dur":
                dur = float(args[i + 1]); i += 2
            elif args[i] == "--compare":
                compare = args[i + 1]; i += 2
            else:
                i += 1
        run_spot(url, start, dur, compare)
        return
    # 全量模式
    if len(args) < 1:
        print(json.dumps({"ok": False, "error": "用法: python video2md.py <URL或本地视频/音频文件路径> [输出目录] | --spot <URL> [--compare 候选稿.md]"}, ensure_ascii=False))
        sys.exit(1)
    run_full(args[0], args[1] if len(args) > 1 else OUTPUT_ROOT)


if __name__ == "__main__":
    main()
