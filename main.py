import os
import json
import base64
import asyncio
import logging
import requests
import websockets
from typing import Dict, Optional
from fastapi import FastAPI, WebSocketDisconnect, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- 环境变量 ----------
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    logger.warning("⚠️ 环境变量 DASHSCOPE_API_KEY 未设置！")

# ---------- 阿里百炼 ASR 配置 ----------
ASR_URL = "wss://dashscope.aliyuncs.com/api/v1/services/audio/asr/realtime"
ASR_MODEL = "fun-asr-realtime"  # 或 qwen3-asr-flash-realtime

# ---------- 阿里百炼 翻译 配置 ----------
TRANSLATE_URL = "https://dashscope.aliyuncs.com/api/v1/services/machine-translation/translation"
TRANSLATE_MODEL = "qwen-mt-turbo"

# ---------- 阿里百炼 TTS 配置 ----------
TTS_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/speech"
TTS_MODEL = "cosyvoice-v2"
TTS_VOICE = "default"  # 可自定义音色

# 语言映射（前端代码 -> 百炼翻译目标语言代码）
LANG_MAP = {
    "zh": "zh",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "fr": "fr",
    "de": "de",
    "es": "es",
    "ru": "ru"
}

# ---------- 房间管理 ----------
rooms: Dict[str, Dict] = {}
asr_sessions: Dict[str, websockets.WebSocketClientProtocol] = {}

# ---------- ASR WebSocket 连接 ----------
async def connect_asr(client_id: str, on_result_callback):
    """建立与阿里百炼 ASR 的 WebSocket 连接"""
    if not DASHSCOPE_API_KEY:
        logger.error("DASHSCOPE_API_KEY 未设置，无法启动 ASR")
        return None
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        ws = await websockets.connect(ASR_URL, extra_headers=headers)
        # 发送启动消息
        start_msg = {
            "header": {
                "action": "start",
                "task": "asr",
                "model": ASR_MODEL
            },
            "payload": {
                "format": "pcm",
                "sample_rate": 16000,
                "channels": 1,
                "enable_punctuation": True,
                "enable_vad": True
            }
        }
        await ws.send(json.dumps(start_msg))
        logger.info(f"ASR 连接已建立: {client_id}")
        asr_sessions[client_id] = ws
        # 启动接收任务
        asyncio.create_task(receive_asr_results(ws, client_id, on_result_callback))
        return ws
    except Exception as e:
        logger.error(f"ASR 连接失败: {e}")
        return None

async def receive_asr_results(ws, client_id, callback):
    """持续接收 ASR 识别结果"""
    try:
        async for msg in ws:
            data = json.loads(msg)
            header = data.get("header")
            if header and header.get("action") == "result":
                text = data.get("payload", {}).get("text", "")
                if text.strip():
                    await callback(client_id, text.strip())
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"ASR 连接关闭: {client_id}")
    finally:
        asr_sessions.pop(client_id, None)

async def send_audio_to_asr(client_id, pcm_bytes):
    """向 ASR 发送音频二进制数据"""
    ws = asr_sessions.get(client_id)
    if not ws:
        logger.warning(f"ASR 会话不存在: {client_id}")
        return
    try:
        await ws.send(pcm_bytes)  # 直接发送二进制 PCM
    except Exception as e:
        logger.error(f"发送音频到 ASR 失败: {e}")

# ---------- 翻译 API ----------
def translate_dashscope(text: str, target_lang: str) -> str:
    """调用阿里百炼机器翻译"""
    if not DASHSCOPE_API_KEY:
        return text
    target = LANG_MAP.get(target_lang, "en")
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": TRANSLATE_MODEL,
        "input": {
            "text": text,
            "source_lang": "auto",
            "target_lang": target
        }
    }
    try:
        resp = requests.post(TRANSLATE_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            return result.get("output", {}).get("text", text)
        else:
            logger.error(f"翻译 API 失败: {resp.text}")
            return text
    except Exception as e:
        logger.error(f"翻译异常: {e}")
        return text

# ---------- TTS API ----------
def cosyvoice_tts(text: str) -> Optional[bytes]:
    """调用阿里百炼 CosyVoice TTS，返回 WAV 音频字节"""
    if not DASHSCOPE_API_KEY:
        return None
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": TTS_MODEL,
        "input": {"text": text},
        "voice": TTS_VOICE,
        "format": "wav"
    }
    try:
        resp = requests.post(TTS_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.content
        else:
            logger.error(f"TTS 失败: {resp.text}")
            return None
    except Exception as e:
        logger.error(f"TTS 异常: {e}")
        return None

# ---------- FastAPI 生命周期 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 同声传译服务器启动 (阿里百炼 ASR + 翻译 + CosyVoice)")
    yield
    # 关闭所有 ASR 连接
    for ws in asr_sessions.values():
        await ws.close()
    logger.info("🛑 服务器关闭")

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

# ---------- WebSocket 端点 ----------
@app.websocket("/ws/{room_id}/{client_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, client_id: str):
    await websocket.accept()
    logger.info(f"✅ 客户端 {client_id} 加入房间 {room_id}")

    if room_id not in rooms:
        rooms[room_id] = {"clients": {}, "languages": {}}
    rooms[room_id]["clients"][client_id] = websocket
    await broadcast_room_status(room_id)

    # 创建 ASR 会话
    asr_ws = None
    async def asr_callback(cid, text):
        """ASR 识别结果回调"""
        # 发送识别文本给说话者自己
        await websocket.send_text(json.dumps({
            "type": "asr_result",
            "text": text
        }))
        # 获取该客户端的语言设置（目标语言），向其他参会者分发翻译
        target_langs = {
            cid: lang for cid, lang in rooms[room_id]["languages"].items()
            if cid != client_id
        }
        if target_langs:
            tasks = []
            for target_cid, target_lang in target_langs.items():
                tasks.append(translate_and_synthesize(text, target_lang, target_cid, room_id, client_id))
            await asyncio.gather(*tasks)

    if DASHSCOPE_API_KEY:
        asr_ws = await connect_asr(client_id, asr_callback)
        if not asr_ws:
            logger.warning(f"ASR 连接失败，将无法识别语音")
    else:
        logger.warning("未设置 DASHSCOPE_API_KEY，ASR 禁用")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "set_language":
                target_lang = message.get("target_lang", "en")
                rooms[room_id]["languages"][client_id] = target_lang
                logger.info(f"   {client_id} 目标语言: {target_lang}")
                await broadcast_room_status(room_id)

            elif msg_type == "audio":
                audio_b64 = message.get("audio", "")
                if not audio_b64:
                    continue
                pcm_bytes = base64.b64decode(audio_b64)
                if asr_ws:
                    await send_audio_to_asr(client_id, pcm_bytes)
                else:
                    # 无 ASR 连接，可做降级处理（例如直接返回提示）
                    pass

    except WebSocketDisconnect:
        logger.info(f"❌ 客户端 {client_id} 断开连接")
    finally:
        # 清理
        if room_id in rooms:
            rooms[room_id]["clients"].pop(client_id, None)
            rooms[room_id]["languages"].pop(client_id, None)
            if not rooms[room_id]["clients"]:
                del rooms[room_id]
        if asr_ws:
            await asr_ws.close()
        asr_sessions.pop(client_id, None)
        await broadcast_room_status(room_id)

# ---------- 翻译 + TTS 分发 ----------
async def translate_and_synthesize(text: str, target_lang: str,
                                   target_client_id: str, room_id: str,
                                   speaker_id: str):
    """翻译并合成语音发送给目标客户端"""
    try:
        translated = translate_dashscope(text, target_lang)
        logger.info(f"翻译 ({target_lang}): {translated}")

        audio_bytes = cosyvoice_tts(translated)
        if not audio_bytes:
            return
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        if room_id in rooms and target_client_id in rooms[room_id]["clients"]:
            target_ws = rooms[room_id]["clients"][target_client_id]
            response = {
                "type": "translation",
                "from": speaker_id,
                "text": translated,
                "audio": audio_b64,
                "lang": target_lang
            }
            await target_ws.send_text(json.dumps(response))
    except Exception as e:
        logger.error(f"翻译合成失败: {e}")

# ---------- 广播状态 ----------
async def broadcast_room_status(room_id: str):
    if room_id not in rooms:
        return
    clients = rooms[room_id]["clients"]
    languages = rooms[room_id]["languages"]
    status = {
        "type": "room_status",
        "clients": [
            {"id": cid, "lang": languages.get(cid, "未设置")}
            for cid in clients.keys()
        ]
    }
    for ws in clients.values():
        try:
            await ws.send_text(json.dumps(status))
        except:
            pass