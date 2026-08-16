import os
import json
import base64
import asyncio
import logging
import requests
from typing import Dict, Optional
from fastapi import FastAPI, WebSocketDisconnect, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

# 阿里百炼 SDK
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback
from dashscope.audio.tts_v2 import SpeechSynthesizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- 配置 ----------
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY

if not DASHSCOPE_API_KEY:
    logger.warning("⚠️ 环境变量 DASHSCOPE_API_KEY 未设置！")

ASR_MODEL = "fun-asr-realtime"

# 翻译 API
TRANSLATE_URL = "https://dashscope.aliyuncs.com/api/v1/services/machine-translation/translation"
TRANSLATE_MODEL = "qwen-mt-turbo"

# TTS 配置
TTS_MODEL = "cosyvoice-v2"
TTS_VOICE = "longxiaochun_v2"

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

rooms: Dict[str, Dict] = {}

# ---------- ASR 会话管理 ----------
class ASRSessionManager:
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
    
    def create_session(self, client_id: str, on_result, on_error):
        if not DASHSCOPE_API_KEY:
            on_error("DASHSCOPE_API_KEY 未设置")
            return None
        
        callback = _StreamingCallback(client_id, on_result, on_error)
        try:
            recognition = Recognition(
                model=ASR_MODEL,
                format="pcm",
                sample_rate=16000,
                callback=callback
            )
            recognition.start()
            self.sessions[client_id] = {
                "recognition": recognition,
                "callback": callback
            }
            logger.info(f"✅ ASR 会话已创建: {client_id}")
            return callback
        except Exception as e:
            logger.error(f"ASR 启动失败: {e}")
            on_error(f"ASR 启动失败: {str(e)}")
            return None
    
    def send_audio(self, client_id: str, pcm_bytes: bytes):
        session = self.sessions.get(client_id)
        if not session:
            return
        try:
            session["recognition"].send_audio_frame(pcm_bytes)
        except Exception as e:
            logger.error(f"发送音频失败: {e}")
            session["callback"].is_broken = True
    
    def close_session(self, client_id: str):
        session = self.sessions.pop(client_id, None)
        if session:
            try:
                session["recognition"].stop()
            except Exception as e:
                logger.error(f"关闭 ASR 会话失败: {e}")
            logger.info(f"ASR 会话已关闭: {client_id}")

class _StreamingCallback(RecognitionCallback):
    def __init__(self, client_id, on_result, on_error):
        self.client_id = client_id
        self.on_result = on_result
        self.on_error = on_error
        self.is_broken = False
    
    def on_open(self):
        logger.info(f"ASR 流式会话已建立: {self.client_id}")
    
    def on_close(self):
        self.is_broken = True
        logger.info(f"ASR 流式会话已关闭: {self.client_id}")
    
    def on_complete(self):
        logger.info(f"ASR 流式会话正常结束: {self.client_id}")
    
    def on_error(self, result):
        self.is_broken = True
        try:
            status_code = getattr(result, "status_code", None)
            message = getattr(result, "message", None)
            logger.error(f"ASR 错误: status_code={status_code}, message={message}")
            self.on_error(f"ASR 错误: {message}")
        except Exception:
            logger.error("ASR 错误（无法解析详细信息）")
            self.on_error("ASR 发生错误")
    
    def on_event(self, result):
        try:
            sentence = result.get_sentence()
        except Exception as e:
            logger.error(f"解析识别结果失败: {e}")
            return
        if not sentence:
            return
        if isinstance(sentence, list):
            for s in sentence:
                self._handle_sentence(s)
        else:
            self._handle_sentence(sentence)
    
    def _handle_sentence(self, sentence):
        if isinstance(sentence, dict):
            text = (sentence.get("text") or "").strip()
            is_end = bool(sentence.get("sentence_end", False))
        else:
            text = str(getattr(sentence, "text", "")).strip()
            is_end = bool(getattr(sentence, "sentence_end", False))
        if not text:
            return
        if is_end:
            logger.info(f"ASR 断句完成 [{self.client_id}]: '{text}'")
            self.on_result(self.client_id, text)
        else:
            logger.info(f"ASR 识别中 [{self.client_id}]: '{text}'")

asr_manager = ASRSessionManager()

# ---------- 翻译 ----------
def translate_text(text: str, target_lang: str) -> str:
    if not text or not DASHSCOPE_API_KEY:
        return text
    target = LANG_MAP.get(target_lang, "en")
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": TRANSLATE_MODEL,
        "input": {"text": text, "source_lang": "auto", "target_lang": target}
    }
    try:
        resp = requests.post(TRANSLATE_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            return result.get("output", {}).get("text", text)
        else:
            logger.error(f"翻译失败: {resp.text}")
            return text
    except Exception as e:
        logger.error(f"翻译异常: {e}")
        return text

# ---------- TTS ----------
def synthesize_speech(text: str) -> Optional[bytes]:
    if not text or not DASHSCOPE_API_KEY:
        return None
    try:
        synthesizer = SpeechSynthesizer(model=TTS_MODEL, voice=TTS_VOICE)
        return synthesizer.call(text)
    except Exception as e:
        logger.error(f"TTS 失败: {e}")
        return None

# ---------- FastAPI ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 同声传译服务器启动（阿里百炼 SDK 版）")
    yield
    for cid in list(asr_manager.sessions.keys()):
        asr_manager.close_session(cid)
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

    def on_result(cid, text):
        asyncio.create_task(handle_asr_result(cid, text, room_id))
    
    def on_error(msg):
        asyncio.create_task(websocket.send_text(json.dumps({
            "type": "asr_error", "msg": msg
        })))
    
    callback = asr_manager.create_session(client_id, on_result, on_error)
    if callback:
        await websocket.send_text(json.dumps({"type": "asr_ready", "msg": "语音识别已就绪"}))
    else:
        await websocket.send_text(json.dumps({"type": "asr_error", "msg": "语音识别启动失败"}))

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
                asr_manager.send_audio(client_id, pcm_bytes)

    except WebSocketDisconnect:
        logger.info(f"❌ 客户端 {client_id} 断开连接")
    finally:
        if room_id in rooms:
            rooms[room_id]["clients"].pop(client_id, None)
            rooms[room_id]["languages"].pop(client_id, None)
            if not rooms[room_id]["clients"]:
                del rooms[room_id]
        asr_manager.close_session(client_id)
        await broadcast_room_status(room_id)

async def handle_asr_result(client_id: str, text: str, room_id: str):
    # 发送给说话者自己
    if room_id in rooms and client_id in rooms[room_id]["clients"]:
        speaker_ws = rooms[room_id]["clients"][client_id]
        await speaker_ws.send_text(json.dumps({"type": "asr_result", "text": text}))
    
    # 翻译给其他人
    target_langs = {
        cid: lang for cid, lang in rooms[room_id]["languages"].items()
        if cid != client_id
    }
    if target_langs:
        tasks = []
        for target_cid, target_lang in target_langs.items():
            tasks.append(translate_and_synthesize(text, target_lang, target_cid, room_id, client_id))
        await asyncio.gather(*tasks)

async def translate_and_synthesize(text: str, target_lang: str,
                                   target_client_id: str, room_id: str,
                                   speaker_id: str):
    try:
        translated = translate_text(text, target_lang)
        logger.info(f"翻译 ({target_lang}): {translated}")
        
        audio_bytes = synthesize_speech(translated)
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8') if audio_bytes else ""
        
        if room_id in rooms and target_client_id in rooms[room_id]["clients"]:
            target_ws = rooms[room_id]["clients"][target_client_id]
            await target_ws.send_text(json.dumps({
                "type": "translation",
                "from": speaker_id,
                "text": translated,
                "audio": audio_b64,
                "lang": target_lang
            }))
    except Exception as e:
        logger.error(f"翻译合成失败: {e}")

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