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
                   after_this_request)

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
    """就地去除单张图片的水印（针对灰/暗色水印）。单页异常不中断整批。"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return
        lower = np.array([160, 160, 160])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(img, lower, upper)
        mask = cv2.GaussianBlur(mask, (1, 1), 0)
        img[mask == 255] = [255, 255, 255]
        cv2.imwrite(image_path, img)
    except Exception as exc:  # noqa: BLE001 - 单页失败不应中断整批
        print(f"[warn] 去水印失败 {image_path}: {exc}")


def pdf_to_images(pdf_path, output_folder):
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
            remove_watermark(image_path)
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


def process_one(session_id, file_id):
    """处理单个已上传文件：转图 → 去水印 → 合回 PDF，返回结果字典。"""
    pdf_path = os.path.join(UPLOAD_DIR, session_id, f"{file_id}.pdf")
    img_folder = os.path.join(IMG_DIR, session_id, file_id)
    out_path = os.path.join(OUT_DIR, session_id, f"{file_id}.pdf")

    if not os.path.exists(pdf_path):
        return {"id": file_id, "pages": 0, "status": "missing"}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        images = pdf_to_images(pdf_path, img_folder)
        images_to_pdf(images, out_path)
        return {"id": file_id, "pages": len(images), "status": "done"}
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

    result = process_one(session_id, file_id)

    # 记录原始文件名，供下载时还原命名
    if result.get("status") == "done":
        original = data.get("name") or f"{file_id}.pdf"
        meta_path = os.path.join(OUT_DIR, session_id, f"{file_id}.pdf.meta.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({"name": original}, fh, ensure_ascii=False)
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
    app.run(debug=True)
