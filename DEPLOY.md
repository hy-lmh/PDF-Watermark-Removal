# 手机使用 / 分享给朋友 · 部署指南

页面已经是手机优先的 H5（微信里打开链接即可用），处理 PDF 需要一台跑 Python 的机器，
下面三种方案按「省事程度」排序，选一种照着做就行。

> 电脑本地使用完全不受影响：`python app.py` 照旧。

---

## 方案一：家里电脑 / NAS + 内网穿透（免费，推荐先玩）

电脑上跑服务，用隧道拿到一个公网 HTTPS 链接，发给朋友就能用。
**缺点：电脑关机后链接就失效。**

```bash
# 1. 启动服务（Docker 方式，或直接 python app.py）
docker compose up -d --build     # 服务跑在 8000 端口

# 2. 装一个隧道工具，任选其一：
#    国内（需注册实名，免费版随机域名）：
natapp / cpolar / 花生壳 —— 按官网教程把隧道指向 localhost:8000

#    无需注册（Cloudflare，国内速度一般）：
brew install cloudflared         # macOS；Windows/Linux 见官网
cloudflared tunnel --url http://localhost:8000
# 输出的 https://xxx.trycloudflare.com 就是公网链接
```

朋友在微信里点链接 → 上传 PDF → 处理 → 点下载时页面会提示
「右上角 ··· → 在浏览器打开」再保存（微信内置浏览器不能直接下载文件，页面已做引导）。

---

## 方案二：轻量云服务器（稳定，约 ¥24~50/月）

腾讯云 / 阿里云「轻量应用服务器」，选 Docker 镜像模板，SSH 进去：

```bash
# 把项目传上去（git clone 或 scp），然后：
docker compose up -d --build
# 访问 http://服务器IP:8000 即可
```

- 想绑域名 + HTTPS：服务器在境内的话域名需要 ICP 备案；直接用 IP 也能在微信里打开。
- 建议在服务器防火墙 / 云控制台安全组只放行 8000 端口。

---

## 方案三：云函数（按量计费，几乎免费，进阶）

腾讯云 SCF「Web 函数」支持容器镜像部署：把本项目构建成镜像推送到
TCR，配置 Web 函数指向容器的 8000 端口，超时设到 600 秒。
冷启动需要 10~20 秒（拉镜像），适合使用频率很低的场景。

---

## 安全与滥用防护（已内置）

- 无鉴权、链接即用，但已内置：**每 IP 每小时最多上传 30 个文件**（`app.py` 里
  `UPLOAD_RATE_N` 可调）、单文件 ≤ 200MB、会话目录 24 小时自动清理。
- 数据都存在 `/data`（Docker）或项目目录（本地），不落任何第三方。
- 若要加简单口令：在 `app.py` 里给 `/upload` 加一个 form 字段校验即可。

## 常用运维

```bash
docker compose logs -f          # 看日志
docker compose restart          # 重启
docker compose down             # 停止（./data 里的临时文件保留，24h 后自动清）
```
