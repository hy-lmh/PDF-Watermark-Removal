#!/bin/bash
# PDF 去水印小工具 · 一键启动（本地服务 + 公网隧道）
# 用法：电脑重启后，在项目目录跑一次 ./start.sh
cd "$(dirname "$0")"

# --- 1) Flask 服务（8000 端口；macOS 的 AirPlay 占用 *:5000，别用 5000）---
if curl -s -m 2 http://127.0.0.1:8000/healthz 2>/dev/null | grep -q '"ok"'; then
    echo "✓ 服务已在运行"
else
    nohup ./venv/bin/python app.py >> /tmp/pdfwm-app.log 2>&1 &
    for _ in $(seq 1 30); do
        curl -s -m 2 http://127.0.0.1:8000/healthz 2>/dev/null | grep -q '"ok"' && break
        sleep 0.5
    done
    if curl -s -m 2 http://127.0.0.1:8000/healthz 2>/dev/null | grep -q '"ok"'; then
        echo "✓ 服务已启动（日志: /tmp/pdfwm-app.log）"
    else
        echo "✗ 服务启动失败，请查看 /tmp/pdfwm-app.log"
        exit 1
    fi
fi

# --- 2) cpolar 公网隧道 ---
# 注意用 pgrep -x：cpolar 守护进程会改写进程标题，-f 按命令行匹配会漏判
if pgrep -x cpolar >/dev/null 2>&1; then
    echo "✓ 公网隧道已在运行"
else
    nohup cpolar http 8000 -log=stdout -log-level=info >> /tmp/pdfwm-cpolar.log 2>&1 &
    echo "✓ 公网隧道已启动（日志: /tmp/pdfwm-cpolar.log）"
fi

# --- 3) 打印访问地址 ---
URL=""
for _ in $(seq 1 15); do
    URL=$(grep -oE "https://[A-Za-z0-9.-]+\.cpolar\.cn" /tmp/pdfwm-cpolar.log 2>/dev/null | tail -1)
    [ -n "$URL" ] && break
    sleep 1
done
LAN=$(ipconfig getifaddr en5 2>/dev/null || ipconfig getifaddr en0 2>/dev/null || echo "?")

echo
echo "============================================"
echo "  本机使用:  http://127.0.0.1:8000"
echo "  手机使用:  http://${LAN}:8000   (需同一 WiFi)"
if [ -n "$URL" ]; then
    echo "  公网链接:  ${URL}"
    echo "  ↑ 发这个给朋友。重启后地址会变，重跑 ./start.sh 拿新地址"
else
    echo "  ⚠ 公网地址还没就绪，稍等后可执行:"
    echo "    grep -oE 'https://[A-Za-z0-9.-]+\.cpolar\.cn' /tmp/pdfwm-cpolar.log | tail -1"
fi
echo "============================================"
