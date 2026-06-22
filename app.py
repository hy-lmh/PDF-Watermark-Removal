import os
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
                   after_this_request, Response)

# --- 配置 ---
UPLOAD_DIR = "uploads"          # 用户上传的原始 PDF（按会话隔离）
IMG_DIR = "output_images"       # PDF 转图片的中间产物（按会话+文件隔离）
OUT_DIR = "outputs"             # 去水印后生成的 PDF（按会话隔离）
ALLOWED_EXT = {".pdf"}
MAX_FILE_MB = 200
SESSION_MAX_AGE_HOURS = 24      # 启动时清理超过此时长的孤儿会话目录
CONVERT_DPI = 300

A4_SIZE_PX_72DPI = (595, 842)   # A4 在 72dpi 下的像素尺寸

# session_id / file_id 只允许这些字符，防止路径穿越
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

app = Flask(__name__)


def safe_id(value):
    """校验会话/文件 id，合法返回原值，否则 None。"""
    if not value or not isinstance(value, str):
        return None
    return value if _ID_RE.match(value) else None


def remove_watermark(image_path):
    """强力去除灰色水印（含极淡/倾斜排布），仅保护彩色底色。

    灰水印与灰格子线颜色相同、无法用颜色区分；本方案优先保证去水印干净，
    故灰格子线会一并去除，而彩色底色（绿/蓝/粉/橙/紫等任意色，饱和度高）完整保留。
    单页异常不中断整批。
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        S = hsv[:, :, 1]
        # 1) 灰水印候选：S≤20（纯灰）+ V≥160（偏亮，含极淡水印）
        cand = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 20, 255]))
        # 2) 保护任意彩色底色：附近 7px 内有饱和度>35 的彩色 → 排除
        S_dil = cv2.dilate(S, np.ones((7, 7), np.uint8))
        cand[S_dil > 35] = 0
        img[cand > 0] = [255, 255, 255]
        cv2.imwrite(image_path, img)
    except Exception as exc:  # noqa: BLE001 - 单页失败不应中断整批
        print(f"[warn] 去水印失败 {image_path}: {exc}")


_ocr_engine = None


def get_ocr():
    """懒加载 OCR 引擎（模块级单例，避免每页重复加载模型）。"""
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _is_watermark_text(text, keyword):
    """判断 OCR 文字是否属于水印：与关键词字符重合度高、且文字不长（排除正文长句）。"""
    text = (text or "").strip()
    keyword = keyword.strip()
    if not text or not keyword:
        return False
    common = sum(1 for c in set(keyword) if c in text)
    overlap = common / len(set(keyword))
    # 水印就是这几个字（含部分识别），正文是长句 → 用长度卡掉正文
    return overlap >= 0.5 and len(text) <= len(keyword) + 3


def remove_text_watermark(image_path, keyword):
    """OCR 定位关键词水印框，仅在框内去除灰色水印像素。

    框内用较宽的灰色阈值（覆盖水印与底色叠加后淡化的部分，避免残留"绿字"），
    但保留框内饱和度高的彩色底（如题型标题绿底）与黑色正文。
    框外的格子、底纹、正文完全不碰。
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0
    try:
        result, _ = get_ocr()(img)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] OCR 失败 {image_path}: {exc}")
        return 0
    if not result:
        return 0

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    # 灰水印像素：低饱和 + 偏亮。框内用宽阈值 S≤55（题型标题绿底 S≈66 仍能保住）
    gray_wm = (S <= 55) & (V >= 150) & (V < 253)

    box_mask = np.zeros((h, w), dtype=np.uint8)
    count = 0
    for box, text, score in result:
        if not _is_watermark_text(text, keyword):
            continue
        try:
            conf = float(score)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < 0.5:
            continue
        cv2.fillPoly(box_mask, [np.array(box, dtype=np.int32)], 255)
        count += 1

    if count == 0:
        return 0
    # 仅轻微膨胀盖住文字笔画边缘，不向外扩张（避免碰到题型标题等彩色底）
    box_mask = cv2.dilate(box_mask, np.ones((3, 3), np.uint8), iterations=1)
    final = (box_mask == 255) & gray_wm
    img[final] = [255, 255, 255]
    cv2.imwrite(image_path, img)
    return count


def manual_remove(image_path, template):
    """用框选的水印样本做模板，多角度匹配定位后 inpaint 修复。

    能去除压在彩色底上的水印：inpaint 用周围彩色底色填充水印区域，底色不损。
    返回匹配到的水印框数量。
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0
    H, W = img.shape[:2]
    th, tw = template.shape[:2]
    if th >= H or tw >= W:
        return 0
    mask = np.zeros((H, W), dtype=np.uint8)
    count = 0
    # 多角度匹配（覆盖水平 + 倾斜水印）
    for ang in (0, -45, -60, 45, 30, -30, -15, 15):
        M = cv2.getRotationMatrix2D((tw / 2, th / 2), ang, 1.0)
        rot = cv2.warpAffine(template, M, (tw, th), borderValue=(255, 255, 255))
        if rot.shape[0] >= H or rot.shape[1] >= W:
            continue
        res = cv2.matchTemplate(img, rot, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= 0.55)
        for y, x in zip(ys, xs):
            cv2.rectangle(mask, (x, y), (x + tw, y + th), 255, -1)
            count += 1
    if mask.any():
        out = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
        cv2.imwrite(image_path, out)
    return count


def process_one_manual(session_id, file_id, template_path):
    """手动框选模式：用样本模板多角度匹配全文档水印并 inpaint 去除。"""
    pdf_path = os.path.join(UPLOAD_DIR, session_id, f"{file_id}.pdf")
    img_folder = os.path.join(IMG_DIR, session_id, file_id)
    out_path = os.path.join(OUT_DIR, session_id, f"{file_id}_manual.pdf")
    if not os.path.exists(pdf_path):
        return {"id": file_id, "mode": "manual", "pages": 0, "status": "missing"}
    template = cv2.imread(template_path)
    if template is None:
        return {"id": file_id, "mode": "manual", "status": "error", "error": "模板读取失败"}

    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    images = []
    try:
        doc = fitz.open(pdf_path)
        dpi = 150  # 手动框选模式用 150dpi，与预览一致、匹配更快
        for i in range(doc.page_count):
            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
            ip = os.path.join(img_folder, f"page_{i + 1}.png")
            pix.save(ip)
            images.append(ip)
            manual_remove(ip, template)
        doc.close()
        images_to_pdf(images, out_path)
        return {"id": file_id, "mode": "manual", "pages": len(images), "status": "done"}
    except Exception as exc:  # noqa: BLE001
        print(f"[error] 手动处理失败 {file_id}: {exc}")
        return {"id": file_id, "mode": "manual", "status": "error", "error": str(exc)}
    finally:
        shutil.rmtree(img_folder, ignore_errors=True)


def pdf_to_images(pdf_path, output_folder, remove_fn=None):
    """把 PDF 每页渲染成 PNG 并去水印，返回图片路径列表。"""
    os.makedirs(output_folder, exist_ok=True)
    images = []
    doc = fitz.open(pdf_path)
    try:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(CONVERT_DPI / 72, CONVERT_DPI / 72))
            image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
            pix.save(image_path)
            images.append(image_path)
            if remove_fn:
                remove_fn(image_path)
    finally:
        doc.close()
    return images


def images_to_pdf(image_paths, output_path):
    """把图片合并为 A4 PDF。直接喂 PIL Image 给 fpdf2，不再落临时文件。"""
    pdf = FPDF(unit="pt", format="A4")
    for image_path in image_paths:
        with Image.open(image_path) as img:
            pdf.add_page()
            # fpdf2 同时给 w 和 h 会把图片铺满整页，无需预缩放
            # （预缩放比例会被二次拉伸抵消，反而引入 int 截断失真）
            pdf.image(img, x=0, y=0, w=A4_SIZE_PX_72DPI[0], h=A4_SIZE_PX_72DPI[1])
    pdf.output(output_path)


def process_one(session_id, file_id, mode="all", keyword=""):
    """处理单个已上传文件：转图 → 去水印 → 合回 PDF，返回结果字典。

    mode: "text" = OCR 精准定位关键词水印（保留格子/底纹/正文）；
          "all"  = HSV 通用去灰水印（快，但会误伤灰格子/漏彩色）。
    """
    pdf_path = os.path.join(UPLOAD_DIR, session_id, f"{file_id}.pdf")
    img_folder = os.path.join(IMG_DIR, session_id, file_id)
    out_path = os.path.join(OUT_DIR, session_id, f"{file_id}_{mode}.pdf")

    if not os.path.exists(pdf_path):
        return {"id": file_id, "pages": 0, "status": "missing"}

    # 按模式选去除函数：text 用 OCR 精准定位，all 用 HSV 通用去灰
    if mode == "text" and keyword.strip():
        kw = keyword.strip()
        remove_fn = lambda p: remove_text_watermark(p, kw)
    else:
        remove_fn = remove_watermark

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        images = pdf_to_images(pdf_path, img_folder, remove_fn)
        images_to_pdf(images, out_path)
        return {"id": file_id, "mode": mode, "pages": len(images), "status": "done"}
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
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    session_id = safe_id(request.form.get("session_id", ""))
    if not session_id:
        return jsonify({"error": "无效的 session_id"}), 400

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

    mode = data.get("mode") or "all"
    keyword = (data.get("keyword") or "").strip()
    result = process_one(session_id, file_id, mode=mode, keyword=keyword)

    # 记录原始文件名（带方式后缀），供下载时还原命名
    if result.get("status") == "done":
        original = data.get("name") or f"{file_id}.pdf"
        base = os.path.splitext(original)[0]
        suffix = "文本水印" if mode == "text" else "所有水印"
        dl_name = f"{base}_{suffix}.pdf"
        meta_path = os.path.join(OUT_DIR, session_id, f"{file_id}_{mode}.pdf.meta.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({"name": dl_name}, fh, ensure_ascii=False)
        except OSError as exc:  # noqa: BLE001
            print(f"[warn] 写 meta.json 失败 {meta_path}: {exc}（下载名将回退为 {file_id}_{mode}.pdf）")

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

    # 指定 file_id + mode：下载某个去水印方式的纯净版
    file_id = safe_id(request.args.get("file", ""))
    mode = request.args.get("mode", "all")
    if mode not in ("all", "text", "manual"):
        mode = "all"
    if file_id:
        target = f"{file_id}_{mode}.pdf"
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


@app.route("/preview")
def preview():
    """渲染某页为 PNG 预览，供前端框选水印样本。"""
    session_id = safe_id(request.args.get("session", ""))
    file_id = safe_id(request.args.get("file", ""))
    page = request.args.get("page", "1")
    try:
        page = max(1, int(page))
    except ValueError:
        page = 1
    if not session_id or not file_id:
        abort(400, "参数无效")
    pdf_path = os.path.join(UPLOAD_DIR, session_id, f"{file_id}.pdf")
    if not os.path.isfile(pdf_path):
        abort(404, "文件不存在")
    doc = fitz.open(pdf_path)
    try:
        idx = min(page - 1, doc.page_count - 1)
        pix = doc[idx].get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72))
        data = pix.tobytes("png")
    finally:
        doc.close()
    return Response(data, mimetype="image/png")


@app.route("/process_manual", methods=["POST"])
def process_manual():
    """手动框选模式：接收模板图，多角度匹配全文档水印并 inpaint 去除。"""
    session_id = safe_id(request.form.get("session_id", ""))
    file_id = safe_id(request.form.get("file_id", ""))
    name = request.form.get("name") or f"{file_id}.pdf"
    if not session_id or not file_id:
        return jsonify({"error": "参数无效"}), 400
    tf = request.files.get("template")
    if not tf:
        return jsonify({"error": "请先框选水印样本"}), 400
    tpl_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(tpl_dir, exist_ok=True)
    tpl_path = os.path.join(tpl_dir, f"{file_id}_template.png")
    tf.save(tpl_path)
    result = process_one_manual(session_id, file_id, tpl_path)
    if result.get("status") == "done":
        base = os.path.splitext(name)[0]
        dl_name = f"{base}_手动框选.pdf"
        meta_path = os.path.join(OUT_DIR, session_id, f"{file_id}_manual.pdf.meta.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({"name": dl_name}, fh, ensure_ascii=False)
        except OSError as exc:  # noqa: BLE001
            print(f"[warn] 写 meta.json 失败 {meta_path}: {exc}")
    return jsonify(result)


@app.route("/cleanup", methods=["POST"])
def cleanup():
    data = request.get_json(silent=True) or {}
    session_id = safe_id(data.get("session_id", ""))
    if session_id:
        cleanup_session(session_id)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True)
