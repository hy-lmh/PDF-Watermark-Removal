FROM python:3.12-slim

# opencv-headless 仍需要 libglib
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates/ templates/
COPY static/ static/

# 数据目录外置到 /data，便于挂载卷持久化
ENV UPLOAD_DIR=/data/uploads \
    IMG_DIR=/data/output_images \
    OUT_DIR=/data/outputs \
    PORT=8000

EXPOSE 8000

# 长超时：大 PDF 的 300dpi 渲染比较吃时间
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "--timeout", "600", "--access-logfile", "-", "app:app"]
