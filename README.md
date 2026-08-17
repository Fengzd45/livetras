# 🎙️ 四人会谈 – 同声传译系统（阿里百炼版）

基于 FastAPI + WebSocket 的实时多人多语言语音翻译助手。采用阿里百炼 DashScope 的语音识别（`fun-asr-realtime`）、机器翻译（`qwen-mt-turbo`）和语音合成（`cosyvoice-v2`）服务。

## ✨ 特性

- 支持 4 人同时在线，每个参会者可选自己的目标语言
- 实时语音识别（流式 WebSocket）
- 自动翻译并合成目标语言语音，推送给其他参会者
- 所有 AI 能力均使用阿里百炼 API，无需本地大模型，**免费版 Render 实例即可运行**

## 🛠️ 技术栈

- 后端：Python + FastAPI + WebSocket
- 前端：原生 HTML/CSS/JS
- 语音识别：DashScope `fun-asr-realtime`（流式）
- 翻译：DashScope `qwen-mt-turbo`
- TTS：DashScope `cosyvoice-v2`

## 📦 部署到 Render（免费）

1. **Fork 本仓库** 或直接上传代码。

2. **获取阿里百炼 API Key**：
   - 登录 [阿里百炼控制台](https://bailian.console.aliyun.com/)
   - 创建 API Key，并确保已开通语音识别、翻译、语音合成服务。

3. **在 Render 上创建 Web Service**：
   - 连接你的 GitHub 仓库。
   - 环境：Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - 计划选择 **Free**。

4. **设置环境变量**（在 Render Dashboard 的 Environment 中）：
   - `DASHSCOPE_API_KEY`：你的阿里百炼 API Key
   - `PYTHON_VERSION`：`3.11.11`（render.yaml 中已经配好；如果你是手动在 Dashboard 里创建服务而没有用 Blueprint，记得手动加上这一项。Render 从 2026 年起不再读取 `runtime.txt`，不设置的话会跑到默认的最新 Python 大版本上，可能和某些依赖不兼容）

5. **部署**，等待完成。访问生成的 `https://<your-app>.onrender.com` 即可使用。

## 💰 费用说明

- **语音识别**：`fun-asr-realtime` 单价 **0.00033元/秒**，国内新用户有 **10小时免费额度**（华北2（北京）地域）。
- **翻译**：`qwen-mt-turbo` 每月 **100万字符免费**。
- **语音合成**：`cosyvoice-v2` 每月有一定免费调用次数（以官方为准）。

个人测试基本可免费使用。

## 🖥️ 使用说明

1. 打开网页，输入你的名字，选择目标语言（即你希望收听的翻译语言）。
2. 点击 **“加入”** 并允许使用麦克风。
3. 当其他人说话时，你会听到翻译后的语音，并看到文字。
4. 你可以随时切换目标语言（改名字需要先离开再重新加入，避免和别人身份冲突）。
5. 点击 **“离开”** 退出会议。

### 🏠 多个会议室

默认所有人都会进入同一个房间 `room1`。如果想同时开多场互不干扰的会谈，在链接后面加 `?room=` 参数区分即可，例如：

- `https://<your-app>.onrender.com/?room=team-a`
- `https://<your-app>.onrender.com/?room=team-b`

只要 `room` 参数不一样，就是两个完全独立的房间，互相听不到对方说话。

## 📁 文件结构