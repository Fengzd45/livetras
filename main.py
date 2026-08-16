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

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback
from dashscope.audio.tts_v2 import SpeechSynthesizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY

if not DASHSCOPE_API_KEY:
    logger.warning("⚠️ 环境变量 DASHSCOPE_API_KEY 未设置！")

ASR_MODEL = "fun-asr-realtime"

TRANSLATE_URL = "https://dashscope.aliyuncs.com/api/v1/services/machine-translation/translation"
TRANSLATE_MODEL = "qwen-mt-turbo"

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
asr_queues: Dict[str, asyncio.Queue] = {}

# ---------- ASR 回调 ----------
class StreamingCallback(RecognitionCallback):
    def __init__(self, client_id: str, queue: asyncio.Queue):
        self.client_id = client_id
        self.queue = queue
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
        logger.error(f"ASR 错误: {result}")

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
            try:
                self.queue.put_nowait(text)
            except asyncio.QueueFull:
                logger.warning(f"队列已满，丢弃: {text}")

# ---------- 翻译 ----------
def translate_text(text: str, target_lang: str) -> str:
    if not text or not DASHSCOPE_API_KEY:
        return text
    target = LANG_MAP.get(target_lang, "en")
    headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": TRANSLATE_MODEL,
        "input": {"text": text, "source_lang": "auto", "target_lang": target}
    }
    try:
        resp = requests.post(TRANSLATE_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json().get("output", {}).get("text", text)
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

    # 创建队列
    queue = asyncio.Queue(maxsize=10)
    asr_queues[client_id] = queue

    # 创建 ASR 会话
    callback = StreamingCallback(client_id, queue)
    try:
        recognition = Recognition(
            model=ASR_MODEL,
            format="pcm",
            sample_rate=16000,
            callback=callback
        )
        recognition.start()
        asr_queues[client_id] = queue
        logger.info(f"✅ ASR 会话已创建: {client_id}")
        await websocket.send_text(json.dumps({"type": "asr_ready", "msg": "语音识别已就绪"}))
    except Exception as e:
        logger.error(f"ASR 启动失败: {e}")
        await websocket.send_text(json.dumps({"type": "asr_error", "msg": f"ASR 启动失败: {e}"}))
        recognition = None

    # 后台任务：从队列取识别结果
    async def process_queue():
        while True:
            try:
                text = await queue.get()
                if text is None:
                    break
                # 在 WebSocket 还存活时处理
                if room_id in rooms and client_id in rooms[room_id]["clients"]:
                    await handle_asr_result(client_id, text, room_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"处理队列失败: {e}")

    queue_task = asyncio.create_task(process_queue())

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
                if recognition:
                    try:
                        recognition.send_audio_frame(pcm_bytes)
                    except Exception as e:
                        logger.error(f"发送音频失败: {e}")

    except WebSocketDisconnect:
        logger.info(f"❌ 客户端 {client_id} 断开连接")
    finally:
        # 清理
        queue_task.cancel()
        try:
            await queue_task
        except asyncio.CancelledError:
            pass

        if recognition:
            try:
                recognition.stop()
            except Exception as e:
                logger.error(f"关闭 ASR 失败: {e}")

        if room_id in rooms:
            rooms[room_id]["clients"].pop(client_id, None)
            rooms[room_id]["languages"].pop(client_id, None)
            if not rooms[room_id]["clients"]:
                del rooms[room_id]

        asr_queues.pop(client_id, None)
        await broadcast_room_status(room_id)

# ---------- 处理识别结果 ----------
async def handle_asr_result(client_id: str, text: str, room_id: str):
    logger.info(f"📝 处理识别结果: {client_id} -> '{text}'")

    # 发送给说话者自己
    if room_id in rooms and client_id in rooms[room_id]["clients"]:
        speaker_ws = rooms[room_id]["clients"][client_id]
        try:
            await speaker_ws.send_text(json.dumps({
                "type": "asr_result",
                "text": text
            }))
            logger.info(f"✅ 已发送 asr_result 给 {client_id}")
        except Exception as e:
            logger.error(f"发送失败: {e}")

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

# ---------- 翻译 + TTS ----------
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
            logger.info(f"✅ 已发送翻译给 {target_client_id}")
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
