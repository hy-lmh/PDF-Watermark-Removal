import os
import io
import re
import json
import time
import shutil
import zipfile
import tempfile

import cv2
import numpy as np
import fitz
from fpdf import FPDF
from PIL import Image
from flask import (Flask, render_template, request, send_file, jsonify, abort,
                   after_this_request, make_response)

# --- 配置 ---
# 目录可被环境变量覆盖（Docker 部署时外置到 /data 卷）
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")   # 用户上传的原始 PDF（按会话隔离）
IMG_DIR = os.environ.get("IMG_DIR", "output_images")   # PDF 转图片的中间产物（按会话+文件隔离）
OUT_DIR = os.environ.get("OUT_DIR", "outputs")         # 去水印后生成的 PDF（按会话隔离）
ALLOWED_EXT = {".pdf"}
MAX_FILE_MB = 200
SESSION_MAX_AGE_HOURS = 24      # 启动时清理超过此时长的孤儿会话目录
CONVERT_DPI = 300              # 兜底管线的渲染分辨率（新管线直接清洗内嵌图，不重采样）

# --- 防滥用限流（公网无鉴权部署的兜底） ---
UPLOAD_RATE_N = 30             # 每个 IP 每窗口期允许上传的文件数
UPLOAD_RATE_WINDOW_S = 3600
_upload_times = {}             # ip -> [时间戳]，滑动窗口


def _client_ip():
    """取客户端 IP；反代场景优先 X-Forwarded-For 的第一跳。"""
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "?"


def _allow_upload():
    """滑动窗口限流：超频返回 False。列表操作在 GIL 下原子，够这个小工具用。"""
    now = time.time()
    ip = _client_ip()
    lst = [t for t in _upload_times.get(ip, []) if now - t < UPLOAD_RATE_WINDOW_S]
    _upload_times[ip] = lst
    if len(lst) >= UPLOAD_RATE_N:
        return False
    lst.append(now)
    return True

# --- 水印识别/清除参数 ---
WATERMARK_KEYWORDS = ["闹爸聊教育"]   # 命中即视为文字层水印（可扩展）
DIAGONAL_WM_MIN_SIZE = 30       # 斜排且字号不小于此值的 span 也视为水印
CHROMA_MAX = 12                 # 纯灰判定的色度容差（容忍 JPEG 色度噪声）
THICK_THR_NATIVE = 2.6          # 原生扫描图上水印笔画半宽阈值（px）
WM_DILATE_K = 3                 # 水印核心强洗环：扩张像素数（吃掉抗锯齿边缘）
WM_DILATE_K2 = 8                # 水印核心外圈弱洗环：更大半径，只洗浅灰残影
WM_RING2_THR = 200              # 外圈弱洗的灰度门槛（田字格 166 等深浅灰不受伤）
WM_FAINT_LO = 200               # 孤儿浅灰下限：淡水印笔画本身浅灰无深色邻居
WM_DARK_ADJ = 4                 # 判定浅灰「贴着深墨」的邻接半径（保护正文抗锯齿）
BG_CLEAN_THR = 238              # 背景/噪点漂白阈值：只清很亮的纯灰，不碰内容浅灰
PNG_TOLERANCE = 1.15            # PNG 体积不超过 JPEG 的此倍数即优先用 PNG（免灰带）
PHOTO_DENS_K = 31               # 照片检测窗口边长
PHOTO_DENS_THR = 0.45           # 窗口中灰覆盖率超过此值视为照片候选区域
PHOTO_MIN_AREA = 15000          # 照片保护的最小连通块面积（小于此的头像等仍清除）
PHOTO_MID_RATIO = 0.55          # 候选块内部中灰占比须达此值才算照片（排除二维码/密集文字）
PHOTO_MIN_SIDE = 200            # 照片保护要求连通块宽高均不小于此值（排除二维码等窄块）
IMG_COVER_SCAN = 0.9            # 判定「扫描页」：单图覆盖页面比例
IMG_COVER_CLEAN = 0.25          # 只清洗覆盖页面至少此比例的内嵌图（小图标不动）

A4_SIZE_PX_72DPI = (595, 842)   # A4 在 72dpi 下的像素尺寸

# session_id / file_id 只允许这些字符，防止路径穿越
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

app = Flask(__name__)


def safe_id(value):
    """校验会话/文件 id，合法返回原值，否则 None。"""
    if not value or not isinstance(value, str):
        return None
    return value if _ID_RE.match(value) else None


def _keep_large(mask_u8, min_area, midtone=None, mid_ratio=0.0, min_side=0):
    """只保留面积达标的连通块；可选再按中灰占比、块宽高过滤。

    二维码/密集文字区会被密度窗口误判成照片候选，但它们要么内部真正的
    中灰像素占比偏低，要么块形状细窄，两条都能把它们排除在保护之外。
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    out = np.zeros_like(mask_u8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        if min_side and (stats[i, cv2.CC_STAT_WIDTH] < min_side
                         or stats[i, cv2.CC_STAT_HEIGHT] < min_side):
            continue
        if midtone is not None:
            inside = lab == i
            if midtone[inside].mean() < mid_ratio:
                continue
        out[lab == i] = 1
    return out


def clean_raster(bgr, thick_thr):
    """清除栅格图里的灰色水印，返回新图。

    规则：
    1. 水印 = 粗笔画（距离变换 >= thick_thr）的纯灰结构，核心向外扩张
       WM_DILATE_K 像素吃掉抗锯齿边缘后漂白；正文/田字格等细笔画结构
       距离值远小于阈值，天然不受影响；
    2. 背景清理只针对很亮（>=BG_CLEAN_THR）的纯灰噪点；
    3. 照片/灰底保护：中灰高密度的大连通块不处理。

    注意不能搞「中亮纯灰(>=160)全漂白」：大量教辅的田字格/四线格就是
    灰度 165~190 的细线，会被整片洗掉。
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bi, gi, ri = cv2.split(bgr.astype(np.int16))
    chroma = np.maximum(np.maximum(abs(bi - gi), abs(gi - ri)), abs(bi - ri))
    pure = chroma <= CHROMA_MAX

    nonwhite = (gray < 250).astype(np.uint8)
    dist = cv2.distanceTransform(nonwhite, cv2.DIST_L2, 5)
    core = (dist >= thick_thr) & (gray >= 70)
    core_u8 = core.astype(np.uint8)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * WM_DILATE_K + 1,) * 2)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * WM_DILATE_K2 + 1,) * 2)
    # 内环强洗：任何纯灰；外圈弱洗：只洗浅灰残影，深浅灰内容（田字格）不受伤
    ring1 = (cv2.dilate(core_u8, k1) > 0) & pure & (gray >= 70)
    ring2 = (cv2.dilate(core_u8, k2) > 0) & pure & (gray >= WM_RING2_THR)

    # 孤儿浅灰：又细又淡的水印笔画成不了粗核心，但它是纯灰、够亮、且周围
    # 没有深色墨迹；正文抗锯齿永远贴着黑字（邻接保护），田字格约 166 不够亮。
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * WM_DARK_ADJ + 1,) * 2)
    dark_d = cv2.dilate((gray < 160).astype(np.uint8), kd) > 0
    orphan = pure & (gray >= WM_FAINT_LO) & (gray < BG_CLEAN_THR) & ~dark_d

    wm_mask = ring1 | ring2 | orphan

    midtone = (pure & (gray >= 60) & (gray <= 250)).astype(np.float32)
    frac = cv2.boxFilter(midtone, -1, (PHOTO_DENS_K, PHOTO_DENS_K),
                         borderType=cv2.BORDER_REPLICATE)
    photo = _keep_large((frac >= PHOTO_DENS_THR).astype(np.uint8), PHOTO_MIN_AREA,
                        midtone=(midtone > 0), mid_ratio=PHOTO_MID_RATIO,
                        min_side=PHOTO_MIN_SIDE)

    mask = (wm_mask | (pure & (gray >= BG_CLEAN_THR))) & (photo == 0)
    out = bgr.copy()
    out[mask] = 255
    return out


def detect_watermark_spans(page):
    """找出页面文字层里的水印 span 矩形：关键词命中，或斜排大字。"""
    rects = []
    try:
        d = page.get_text("dict")
    except Exception:  # noqa: BLE001 - 损坏的字体/编码不应中断处理
        return rects
    for blk in d["blocks"]:
        if blk["type"] != 0:
            continue
        for line in blk["lines"]:
            dir_y = line.get("dir", (1, 0))[1]
            diagonal = abs(dir_y) > 0.3
            for sp in line["spans"]:
                hit_kw = any(k and k in sp["text"] for k in WATERMARK_KEYWORDS)
                if hit_kw or (diagonal and sp["size"] >= DIAGONAL_WM_MIN_SIZE):
                    r = fitz.Rect(sp["bbox"])
                    r += (-2, -2, 2, 2)   # 留一点边距盖住字形边缘
                    rects.append(r)
    return rects


def remove_text_watermarks(doc):
    """无损删除文字层水印：只删文字，不动图片和矢量，不涂白。"""
    for page in doc:
        rects = detect_watermark_spans(page)
        if not rects:
            continue
        for r in rects:
            page.add_redact_annot(r, fill=False)
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                              graphics=fitz.PDF_REDACT_LINE_ART_NONE)


# /Artifact <</Subtype/Watermark ...>> BDC ... EMC：标准水印标记块（矢量水印）
_WM_BDC_RE = re.compile(
    rb"/Artifact\s*<<[^>]*?/Subtype\s*/Watermark[^>]*?>>\s*BDC(?:(?!EMC|BDC).)*EMC",
    re.DOTALL)


def remove_artifact_watermarks(doc):
    """删除内容流里 /Artifact /Subtype /Watermark 标记的绘制块（矢量水印）。

    这是 PDF 标准的水印写法，标记由生成方显式写入，误伤风险极低。
    块内不含嵌套 BDC/EMC 时才能整体匹配，嵌套时宁可不删也不误伤。
    """
    for page in doc:
        try:
            xrefs = page.get_contents()
        except Exception:  # noqa: BLE001
            continue
        for xref in xrefs:
            try:
                data = doc.xref_stream(xref)
            except Exception:  # noqa: BLE001
                continue
            new = _WM_BDC_RE.sub(b"", data)
            if new != data:
                doc.update_stream(xref, new)


def pixmap_to_bgr(pix):
    """fitz.Pixmap 转 BGR ndarray；无法安全转换的（如图像掩码）返回 None。"""
    if pix.colorspace is None:
        return None
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)
    if pix.colorspace.n >= 4:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 3:
        return np.ascontiguousarray(arr[:, :, ::-1])   # RGB -> BGR
    if pix.n == 1:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return None


def encode_image(bgr):
    """按体积择优编码：PNG 无损（文本页常更小且无 JPEG 灰带），JPEG 适合照片页。

    JPEG 在文字行附近会产生 250-254 的浅灰块（DCT 量化偏置），观感像
    水印残影；PNG 不存在此问题，故优先在体积不吃亏时用 PNG。
    """
    ok_png, png = cv2.imencode(".png", bgr)
    ok_jpg, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok_png and (not ok_jpg or png.size <= int(jpg.size * PNG_TOLERANCE)):
        return png.tobytes()
    if ok_jpg:
        return jpg.tobytes()
    return None


def rebuild_scan_pdf(doc, out_path):
    """扫描版 PDF 重建：每页只铺一张清洗后的主扫描图。

    相比在原文档里替换图像流（replace_image/update_stream 各有坑：前者
    留双份对象且引用不可靠，后者强改 /Filter 会让 iPhone 解码黑屏），
    重建一页一张标准 JPEG 对象，各阅读器通吃；文字层/矢量水印层天然丢弃。
    """
    out = fitz.open()
    cache = {}   # xref -> 清洗后的 JPEG 字节（跨页复用同一图像时免重算）
    for page in doc:
        page_area = abs(page.rect) or 1.0
        big = None
        for info in page.get_image_info(xrefs=True):
            r = fitz.Rect(info["bbox"])
            if not r.is_empty and abs(r) >= IMG_COVER_SCAN * page_area:
                big = info
                break
        rect = fitz.Rect(0, 0, page.rect.width, page.rect.height)
        npage = out.new_page(width=page.rect.width, height=page.rect.height)
        if page.rotation:
            npage.set_rotation(page.rotation)
        if big is None:
            continue   # is_scan_page 已保证有主图，这里只是兜底
        xref = big["xref"]
        if xref in cache:
            npage.insert_image(rect, stream=cache[xref])
            continue
        try:
            pix = fitz.Pixmap(doc, xref)
            bgr = pixmap_to_bgr(pix)
            if bgr is None:
                raise ValueError("无法解码的图像")
            cleaned = clean_raster(bgr, THICK_THR_NATIVE)
            jb = encode_image(cleaned)
            if jb is None:
                raise ValueError("图像编码失败")
            cache[xref] = jb
            npage.insert_image(rect, stream=jb)
        except Exception as exc:  # noqa: BLE001 - 清洗失败时保留原图
            print(f"[warn] 重建页 {page.number + 1} 失败: {exc}")
            try:
                base = doc.extract_image(xref)
                npage.insert_image(rect, stream=base["image"])
            except Exception:  # noqa: BLE001
                pass
    out.save(out_path, garbage=3, deflate=True)
    out.close()


def is_scan_page(page):
    """页面上是否有覆盖整页的大图（扫描页判定）。"""
    page_area = abs(page.rect) or 1.0
    for info in page.get_image_info(xrefs=True):
        r = fitz.Rect(info["bbox"])
        if not r.is_empty and abs(r) >= IMG_COVER_SCAN * page_area:
            return True
    return False


def pdf_to_images(doc, output_folder):
    """兜底管线：把已去文字层水印的 doc 每页渲染成 PNG 并清栅格水印。"""
    os.makedirs(output_folder, exist_ok=True)
    images = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(CONVERT_DPI / 72, CONVERT_DPI / 72))
        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        pix.save(image_path)
        try:
            img = cv2.imread(image_path)
            if img is not None:
                cv2.imwrite(image_path, clean_raster(img, CONVERT_DPI * 0.023))
        except Exception as exc:  # noqa: BLE001 - 单页失败不应中断整批
            print(f"[warn] 去水印失败 {image_path}: {exc}")
        images.append(image_path)
    return images


def images_to_pdf(image_paths, output_path):
    """把图片合并为 A4 PDF。转 JPEG(质量80)压缩嵌入，控制产物体积。

    PNG 无损约 1MB/页，公网隧道（如 cpolar 免费版限速 ~1Mbps）下载会超时；
    JPEG q80 可压到约 1/5，300dpi 下文字可读性几乎无损。
    """
    pdf = FPDF(unit="pt", format="A4")
    for image_path in image_paths:
        with Image.open(image_path) as img:
            pdf.add_page()
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            buf.seek(0)
            # fpdf2 同时给 w 和 h 会把图片铺满整页，无需预缩放
            # （预缩放比例会被二次拉伸抵消，反而引入 int 截断失真）
            pdf.image(buf, x=0, y=0, w=A4_SIZE_PX_72DPI[0], h=A4_SIZE_PX_72DPI[1])
    pdf.output(output_path)


def process_one(session_id, file_id):
    """处理单个已上传文件，返回结果字典。

    新管线（扫描版 PDF）：整册重建，每页只铺清洗后的主扫描图（原生分辨率），
    文字/矢量水印层天然丢弃；
    兜底管线（其余 PDF）：删两类水印后整页 300dpi 渲染 + 栅格清洗 + 重排 PDF。
    """
    pdf_path = os.path.join(UPLOAD_DIR, session_id, f"{file_id}.pdf")
    img_folder = os.path.join(IMG_DIR, session_id, file_id)
    out_path = os.path.join(OUT_DIR, session_id, f"{file_id}.pdf")

    if not os.path.exists(pdf_path):
        return {"id": file_id, "pages": 0, "status": "missing"}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        with fitz.open(pdf_path) as doc:
            scan_doc = doc.page_count > 0 and all(is_scan_page(p) for p in doc)
            if scan_doc:
                rebuild_scan_pdf(doc, out_path)
                pages = doc.page_count
            else:
                remove_artifact_watermarks(doc)
                remove_text_watermarks(doc)
                images = pdf_to_images(doc, img_folder)
                images_to_pdf(images, out_path)
                pages = len(images)
        return {"id": file_id, "pages": pages, "status": "done"}
    except Exception as exc:  # noqa: BLE001
        print(f"[error] 处理失败 {file_id}: {exc}")
        return {"id": file_id, "pages": 0, "status": "error", "error": str(exc)}
    finally:
        # 中间图片是大头（约 1MB/页），处理完立刻回收
        shutil.rmtree(img_folder, ignore_errors=True)


def zip_session(session_id):
    """把某会话下所有成品 PDF 流式打包到临时 zip 文件，返回路径。同名文件自动去重。"""
    out_folder = os.path.join(OUT_DIR, session_id)
    fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="pwr_")
    os.close(fd)
    used_names = {}
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in sorted(os.listdir(out_folder)):
                if not name.endswith(".pdf"):
                    continue
                arc = _read_meta(out_folder, name)  # 已含 basename 清洗与兜底
                base, ext = os.path.splitext(arc)
                candidate = arc
                i = 1
                while candidate.lower() in used_names:
                    candidate = f"{base} ({i}){ext}"
                    i += 1
                used_names[candidate.lower()] = True
                zf.write(os.path.join(out_folder, name), arcname=candidate)
    except Exception:
        try:
            os.remove(zip_path)
        except OSError:
            pass
        raise
    return zip_path


def cleanup_session(session_id):
    """删除某会话在三个根目录下的全部文件。"""
    if not safe_id(session_id):
        return
    for base in (UPLOAD_DIR, IMG_DIR, OUT_DIR):
        shutil.rmtree(os.path.join(base, session_id), ignore_errors=True)


def startup_cleanup():
    """启动时清理超过 SESSION_MAX_AGE_HOURS 的孤儿会话目录。"""
    cutoff = time.time() - SESSION_MAX_AGE_HOURS * 3600
    for base in (UPLOAD_DIR, IMG_DIR, OUT_DIR):
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            path = os.path.join(base, name)
            if not os.path.isdir(path):
                continue
            try:
                if os.path.getmtime(path) < cutoff:
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass


startup_cleanup()


@app.route("/")
def index():
    # 页面带内联 JS，更新频繁；禁掉缓存，避免手机端一直跑旧版逻辑
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-cache, max-age=0, must-revalidate"
    return resp


@app.route("/healthz")
def healthz():
    """部署探活用。"""
    return jsonify({"ok": True})


@app.route("/files")
def list_files():
    """按会话列出已处理完成的文件，供换浏览器后恢复列表（微信→浏览器跳转）。"""
    session_id = safe_id(request.args.get("session_id", ""))
    if not session_id:
        return jsonify([])
    out_folder = os.path.join(OUT_DIR, session_id)
    items = []
    if os.path.isdir(out_folder):
        for name in sorted(os.listdir(out_folder)):
            if not name.endswith(".pdf"):
                continue
            fid = name[:-len(".pdf")]
            pages = None
            try:
                with open(os.path.join(out_folder, name + ".meta.json"), "r", encoding="utf-8") as fh:
                    pages = json.load(fh).get("pages")
            except Exception:  # noqa: BLE001 - meta 缺失/损坏不影响列表
                pass
            items.append({
                "id": fid,
                "name": _read_meta(out_folder, name),
                "pages": pages,
                "status": "done",
            })
    return jsonify(items)


@app.route("/upload", methods=["POST"])
def upload():
    session_id = safe_id(request.form.get("session_id", ""))
    if not session_id:
        return jsonify({"error": "无效的 session_id"}), 400

    if not _allow_upload():
        return jsonify({"error": "上传太频繁啦，请一小时后再试"}), 429

    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "未收到文件"}), 400

    upload_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(upload_dir, exist_ok=True)

    manifest = []
    for f in files:
        original = f.filename or ""
        ext = os.path.splitext(original)[1].lower()
        if ext not in ALLOWED_EXT:
            manifest.append({"name": original, "size": 0, "status": "rejected"})
            continue

        # 读取并校验大小
        f.stream.seek(0, 2)
        size = f.stream.tell()
        f.stream.seek(0)
        if size > MAX_FILE_MB * 1024 * 1024:
            manifest.append({"name": original, "size": size, "status": "too_large"})
            continue

        file_id = os.urandom(8).hex()
        save_path = os.path.join(upload_dir, f"{file_id}.pdf")
        try:
            f.save(save_path)
            manifest.append({
                "id": file_id,
                "name": original,
                "size": size,
                "status": "uploaded",
            })
        except Exception as exc:  # noqa: BLE001
            manifest.append({"name": original, "size": size, "status": "error", "error": str(exc)})

    return jsonify(manifest)


@app.route("/process", methods=["POST"])
def process():
    data = request.get_json(silent=True) or {}
    session_id = safe_id(data.get("session_id", ""))
    file_id = safe_id(data.get("file_id", ""))
    if not session_id or not file_id:
        return jsonify({"error": "参数无效"}), 400

    result = process_one(session_id, file_id)

    # 记录原始文件名，供下载时还原命名
    if result.get("status") == "done":
        original = data.get("name") or f"{file_id}.pdf"
        meta_path = os.path.join(OUT_DIR, session_id, f"{file_id}.pdf.meta.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({"name": original, "pages": result.get("pages")}, fh, ensure_ascii=False)
        except OSError as exc:  # noqa: BLE001
            print(f"[warn] 写 meta.json 失败 {meta_path}: {exc}（下载名将回退为 {file_id}.pdf）")

    return jsonify(result)


def _read_meta(out_folder, pdf_name):
    """读取某个成品 PDF 的原始文件名，找不到则回退。结果剥离路径前缀，避免 Zip Slip。"""
    fallback = os.path.splitext(pdf_name)[0] + ".pdf"
    meta_path = os.path.join(out_folder, pdf_name + ".meta.json")
    name = fallback
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                name = json.load(fh).get("name", fallback)
        except Exception:  # noqa: BLE001
            name = fallback
    return os.path.basename(name) or fallback


@app.route("/download")
def download():
    session_id = safe_id(request.args.get("session", ""))
    if not session_id:
        abort(400, "无效的 session")

    out_folder = os.path.join(OUT_DIR, session_id)
    if not os.path.isdir(out_folder):
        abort(404, "没有可下载的结果")

    pdfs = sorted(n for n in os.listdir(out_folder) if n.endswith(".pdf"))
    if not pdfs:
        abort(404, "没有可下载的结果")

    # 指定 file_id：下载单个文件的纯净版（供队列里「下载纯净版」）
    file_id = safe_id(request.args.get("file", ""))
    if file_id:
        target = file_id + ".pdf"
        path = os.path.join(out_folder, target)
        if not os.path.isfile(path):
            abort(404, "文件不存在")
        return send_file(
            path,
            as_attachment=True,
            download_name=_read_meta(out_folder, target),
        )

    if len(pdfs) == 1:
        return send_file(
            os.path.join(out_folder, pdfs[0]),
            as_attachment=True,
            download_name=_read_meta(out_folder, pdfs[0]),
        )

    zip_path = zip_session(session_id)

    @after_this_request
    def _remove_zip(response):
        try:
            os.remove(zip_path)
        except OSError:
            pass
        return response

    return send_file(
        zip_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"去水印结果_{session_id[:8]}.zip",
    )


@app.route("/cleanup", methods=["POST"])
def cleanup():
    data = request.get_json(silent=True) or {}
    session_id = safe_id(data.get("session_id", ""))
    if session_id:
        cleanup_session(session_id)
    return jsonify({"ok": True})


if __name__ == "__main__":
    # 局域网可访问（手机测试），生产环境用 Docker 里的 gunicorn。
    # 默认 8000：macOS 的隔空播放占用 *:5000，手机访问 5000 会被它截走。
    port = int(os.environ.get("PORT", 8000))
    import socket
    try:
        lan_ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        lan_ip = "?.?.?.?"
    print(f"\n  PDF 去水印小工具已启动")
    print(f"  本机访问:   http://127.0.0.1:{port}")
    print(f"  手机访问:   http://{lan_ip}:{port}  (需同一 WiFi)")
    if port == 5000:
        print("  ⚠️ 注意: macOS 隔空播放占用 5000 端口，其他设备访问会白屏，建议用 8000\n")
    app.run(host="0.0.0.0", port=port, debug=True)
