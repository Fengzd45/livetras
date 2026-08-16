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
4. 你可以随时切换目标语言。
5. 点击 **“离开”** 退出会议。

## 📁 文件结构
