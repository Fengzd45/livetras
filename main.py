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

# --- 配置 ---
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    logger.warning("⚠️ 环境变量 DASHSCOPE_API_KEY 未设置！")

# 阿里百炼流式 ASR WebSocket 地址
ASR_WS_URL = "wss://dashscope.aliyuncs.com/api/v1/realtime/audio/asr"
ASR_MODEL = "fun-asr-realtime"  # 或 qwen3-asr-flash-realtime

# 翻译和 TTS（保持原有非流式）
TRANSLATE_URL = "https://dashscope.aliyuncs.com/api/v1/services/machine-translation/translation"
TTS_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/speech"
TRANSLATE_MODEL = "qwen-mt-turbo"
TTS_MODEL = "cosyvoice-v2"
TTS_VOICE = "default"

LANG_MAP = {
    "zh": "zh", "en": "en", "ja": "ja", "ko": "ko",
    "fr": "fr", "de": "de", "es": "es", "ru": "ru"
}

rooms: Dict[str, Dict] = {}

# ---------- 流式 ASR 会话管理 ----------
class ASRSession:
    """管理单个客户端的流式 ASR 连接，并对外提供同步识别接口"""
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.ws = None
        self._result_queue = asyncio.Queue()
        self._running = False
        self._task = None

    async def connect(self):
        """建立 WebSocket 连接并启动接收"""
        if not DASHSCOPE_API_KEY:
            raise Exception("DASHSCOPE_API_KEY 未设置")
        headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
        }
        self.ws = await websockets.connect(ASR_WS_URL, extra_headers=headers)
        # 发送开始参数
        start_msg = {
            "header": {"action": "start", "task": "asr", "model": ASR_MODEL},
            "payload": {
                "format": "pcm",
                "sample_rate": 16000,
                "channels": 1,
                "enable_punctuation": True,
                "enable_vad": True
            }
        }
        await self.ws.send(json.dumps(start_msg))
        self._running = True
        self._task = asyncio.create_task(self._receive_loop())
        logger.info(f"ASR 流式会话已连接: {self.client_id}")

    async def _receive_loop(self):
        """持续接收 ASR 结果，放入队列"""
        try:
            async for msg in self.ws:
                data = json.loads(msg)
                header = data.get("header")
                if header and header.get("action") == "result":
                    text = data.get("payload", {}).get("text", "").strip()
                    if text:
                        await self._result_queue.put(text)
                elif header and header.get("action") == "error":
                    error_msg = data.get("payload", {}).get("message", "ASR 错误")
                    logger.error(f"ASR 错误: {error_msg}")
                    await self._result_queue.put(None)  # 用 None 表示错误
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"ASR 连接关闭: {self.client_id}")
        finally:
            self._running = False
            await self._result_queue.put(None)  # 结束标记

    async def send_audio(self, pcm_bytes: bytes):
        """发送音频数据（二进制 PCM）"""
        if self.ws and self._running:
            try:
                await self.ws.send(pcm_bytes)
            except Exception as e:
                logger.error(f"发送音频失败: {e}")

    async def get_result(self, timeout=5.0) -> Optional[str]:
        """等待下一个识别结果，返回文本或 None（超时/错误）"""
        try:
            result = await asyncio.wait_for(self._result_queue.get(), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            return None

    async def close(self):
        """关闭连接"""
        self._running = False
        if self._task:
            self._task.cancel()
        if self.ws:
            await self.ws.close()

# 全局存储 ASR 会话
asr_sessions: Dict[str, ASRSession] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 同声传译服务器启动（流式 ASR + 阿里百炼）")
    yield
    # 关闭所有 ASR 会话
    for session in asr_sessions.values():
        await session.close()
    logger.info("🛑 服务器关闭")

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

@app.websocket("/ws/{room_id}/{client_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, client_id: str):
    await websocket.accept()
    logger.info(f"✅ 客户端 {client_id} 加入房间 {room_id}")

    if room_id not in rooms:
        rooms[room_id] = {"clients": {}, "languages": {}}
    rooms[room_id]["clients"][client_id] = websocket
    await broadcast_room_status(room_id)

    # 创建 ASR 会话
    asr_session = ASRSession(client_id)
    try:
        await asr_session.connect()
        asr_sessions[client_id] = asr_session
        # 发送就绪消息
        await websocket.send_text(json.dumps({
            "type": "asr_ready",
            "msg": "语音识别已就绪（流式）"
        }))
    except Exception as e:
        logger.error(f"ASR 初始化失败: {e}")
        await websocket.send_text(json.dumps({
            "type": "asr_error",
            "msg": f"语音识别启动失败: {str(e)}"
        }))
        # 即使 ASR 失败，仍继续运行（只是无法识别）

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
                logger.info(f"🎤 收到 {client_id} 的音频数据")
                audio_b64 = message.get("audio", "")
                if not audio_b64:
                    continue

                # 将音频数据发送给 ASR 流式会话
                pcm_bytes = base64.b64decode(audio_b64)
                if client_id in asr_sessions:
                    await asr_sessions[client_id].send_audio(pcm_bytes)

                # 检查是否有识别结果（非阻塞）
                # 注意：由于音频是连续流，不能每次请求都等待，我们采用异步回调方式
                # 下面启动一个后台任务来处理识别结果
                asyncio.create_task(
                    handle_asr_result(client_id, room_id, asr_session)
                )

    except WebSocketDisconnect:
        logger.info(f"❌ 客户端 {client_id} 断开连接")
    finally:
        if room_id in rooms:
            rooms[room_id]["clients"].pop(client_id, None)
            rooms[room_id]["languages"].pop(client_id, None)
            if not rooms[room_id]["clients"]:
                del rooms[room_id]
        if client_id in asr_sessions:
            await asr_sessions[client_id].close()
            del asr_sessions[client_id]
        await broadcast_room_status(room_id)

async def handle_asr_result(client_id: str, room_id: str, session: ASRSession):
    """从 ASR 会话获取一个结果并处理（翻译+分发）"""
    # 获取下一个识别结果（非阻塞，超时 0.5 秒）
    result = await session.get_result(timeout=0.5)
    if not result:
        return
    # 如果结果为空（None）表示错误或结束，忽略
    if result is None:
        return

    logger.info(f"ASR 识别结果: {client_id} -> {result}")

    # 发送识别文本给说话者自己
    if room_id in rooms and client_id in rooms[room_id]["clients"]:
        speaker_ws = rooms[room_id]["clients"][client_id]
        await speaker_ws.send_text(json.dumps({
            "type": "asr_result",
            "text": result
        }))

    # 获取目标语言列表（其他参会者）
    target_langs = {
        cid: lang for cid, lang in rooms[room_id]["languages"].items()
        if cid != client_id
    }
    if target_langs:
        tasks = []
        for target_cid, target_lang in target_langs.items():
            tasks.append(
                translate_and_synthesize(result, target_lang, target_cid, room_id, client_id)
            )
        await asyncio.gather(*tasks)

# ---------- 翻译和 TTS（完全复用之前的） ----------
async def translate_and_synthesize(text: str, target_lang: str,
                                   target_client_id: str, room_id: str,
                                   speaker_id: str):
    try:
        target = LANG_MAP.get(target_lang, "en")
        trans_headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json"
        }
        trans_payload = {
            "model": TRANSLATE_MODEL,
            "input": {"text": text, "source_lang": "auto", "target_lang": target}
        }
        trans_resp = requests.post(TRANSLATE_URL, headers=trans_headers, json=trans_payload, timeout=30)
        if trans_resp.status_code != 200:
            logger.error(f"翻译失败: {trans_resp.text}")
            return
        trans_result = trans_resp.json()
        translated_text = trans_result.get("output", {}).get("text", text).strip()
        logger.info(f"翻译 ({target_lang}): {translated_text}")

        tts_headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json"
        }
        tts_payload = {
            "model": TTS_MODEL,
            "input": {"text": translated_text},
            "voice": TTS_VOICE,
            "format": "wav"
        }
        tts_resp = requests.post(TTS_URL, headers=tts_headers, json=tts_payload, timeout=30)
        if tts_resp.status_code != 200:
            logger.error(f"TTS 失败: {tts_resp.text}")
            return
        audio_bytes = tts_resp.content
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        if room_id in rooms and target_client_id in rooms[room_id]["clients"]:
            target_ws = rooms[room_id]["clients"][target_client_id]
            response = {
                "type": "translation",
                "from": speaker_id,
                "text": translated_text,
                "audio": audio_b64,
                "lang": target_lang
            }
            await target_ws.send_text(json.dumps(response))
    except Exception as e:
        logger.error(f"翻译合成失败: {e}", exc_info=True)

# ---------- broadcast_room_status（保持不变） ----------
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
