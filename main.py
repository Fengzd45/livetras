import os
import json
import base64
import asyncio
import logging
import queue
import tempfile
import time
from typing import Dict, Optional
from fastapi import FastAPI, WebSocketDisconnect, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

# 阿里百炼 SDK
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback

# 翻译和 TTS（保持原样，TTS 用 SDK 的 SpeechSynthesizer）
from dashscope.audio.tts_v2 import SpeechSynthesizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- 配置 ----------
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY

if not DASHSCOPE_API_KEY:
    logger.warning("⚠️ 环境变量 DASHSCOPE_API_KEY 未设置！")

# 阿里百炼 ASR 配置
ASR_MODEL = "fun-asr-realtime"  # 或 paraformer-realtime-v2

# 语言映射
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

# 房间管理
rooms: Dict[str, Dict] = {}

# ---------- ASR 会话管理器 ----------
class ASRSessionManager:
    """管理每个客户端的 ASR 识别会话"""
    def __init__(self):
        self.sessions: Dict[str, dict] = {}  # client_id -> {"recognition": Recognition, "callback": Callback}
    
    def create_session(self, client_id: str, language: str, on_result_callback, on_error_callback):
        """创建流式识别会话"""
        if not DASHSCOPE_API_KEY:
            on_error_callback("DASHSCOPE_API_KEY 未设置")
            return None
        
        # 创建回调
        callback = _StreamingCallback(
            client_id=client_id,
            language=language,
            on_result=on_result_callback,
            on_error=on_error_callback
        )
        
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
            on_error_callback(f"ASR 启动失败: {str(e)}")
            return None
    
    def send_audio(self, client_id: str, pcm_bytes: bytes):
        """发送音频数据到识别会话"""
        session = self.sessions.get(client_id)
        if not session:
            return
        try:
            session["recognition"].send_audio_frame(pcm_bytes)
        except Exception as e:
            logger.error(f"发送音频失败: {e}")
            # 标记会话已损坏，下次会重建
            session["callback"].is_broken = True
    
    def close_session(self, client_id: str):
        """关闭识别会话"""
        session = self.sessions.pop(client_id, None)
        if session:
            try:
                session["recognition"].stop()
            except Exception as e:
                logger.error(f"关闭 ASR 会话失败: {e}")
            logger.info(f"ASR 会话已关闭: {client_id}")

# ---------- ASR 回调类 ----------
class _StreamingCallback(RecognitionCallback):
    """流式识别回调"""
    def __init__(self, client_id: str, language: str, on_result, on_error):
        self.client_id = client_id
        self.language = language
        self.on_result = on_result
        self.on_error = on_error
        self.is_broken = False
    
    def on_open(self) -> None:
        logger.info(f"ASR 流式会话已建立: {self.client_id}")
    
    def on_close(self) -> None:
        self.is_broken = True
        logger.info(f"ASR 流式会话已关闭: {self.client_id}")
    
    def on_complete(self) -> None:
        logger.info(f"ASR 流式会话正常结束: {self.client_id}")
    
    def on_error(self, result) -> None:
        self.is_broken = True
        try:
            status_code = getattr(result, "status_code", None)
            message = getattr(result, "message", None)
            logger.error(f"ASR 错误: status_code={status_code}, message={message}")
            self.on_error(f"ASR 错误: {message}")
        except Exception:
            logger.error(f"ASR 错误（无法解析详细信息）")
            self.on_error("ASR 发生错误")
    
    def on_event(self, result) -> None:
        """处理识别结果"""
        try:
            sentence = result.get_sentence()
        except Exception as e:
            logger.error(f"解析识别结果失败: {e}")
            return
        if not sentence:
            return
        
        # 处理句子
        if isinstance(sentence, list):
            for s in sentence:
                self._handle_sentence(s)
        else:
            self._handle_sentence(sentence)
    
    def _handle_sentence(self, sentence) -> None:
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

# ---------- 全局 ASR 管理器 ----------
asr_manager = ASRSessionManager()

# ---------- FastAPI 生命周期 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 同声传译服务器启动（阿里百炼 SDK 版）")
    yield
    # 关闭所有 ASR 会话
    for client_id in list(asr_manager.sessions.keys()):
        asr_manager.close_session(client_id)
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

    # ASR 回调函数
    def on_asr_result(cid, text):
        """ASR 识别结果回调"""
        asyncio.create_task(handle_asr_result(cid, text, room_id))
    
    def on_asr_error(error_msg):
        """ASR 错误回调"""
        asyncio.create_task(websocket.send_text(json.dumps({
            "type": "asr_error",
            "msg": error_msg
        })))
    
    # 创建 ASR 会话
    my_lang = "zh"  # 默认中文，后续通过 set_language 更新
    callback = asr_manager.create_session(client_id, my_lang, on_asr_result, on_asr_error)
    
    if callback:
        await websocket.send_text(json.dumps({
            "type": "asr_ready",
            "msg": "语音识别已就绪"
        }))
    else:
        await websocket.send_text(json.dumps({
            "type": "asr_error",
            "msg": "语音识别启动失败"
        }))

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
                
                # 如果 ASR 会话需要重建（语言 hint 改变），可以重建
                # 但 fun-asr-realtime 不需要 language_hints

            elif msg_type == "audio":
                audio_b64 = message.get("audio", "")
                if not audio_b64:
                    continue
                pcm_bytes = base64.b64decode(audio_b64)
                # 发送到 ASR 会话
                asr_manager.send_audio(client_id, pcm_bytes)

    except WebSocketDisconnect:
        logger.info(f"❌ 客户端 {client_id} 断开连接")
    finally:
        # 清理
        if room_id in rooms:
            rooms[room_id]["clients"].pop(client_id, None)
            rooms[room_id]["languages"].pop(client_id, None)
            if not rooms[room_id]["clients"]:
                del rooms[room_id]
        asr_manager.close_session(client_id)
        await broadcast_room_status(room_id)

# ---------- 处理 ASR 结果 ----------
async def handle_asr_result(client_id: str, text: str, room_id: str):
    """处理 ASR 识别结果：发送给说话者 + 翻译给其他人"""
    # 发送给说话者自己（确认识别结果）
    if room_id in rooms and client_id in rooms[room_id]["clients"]:
        speaker_ws = rooms[room_id]["clients"][client_id]
        await speaker_ws.send_text(json.dumps({
            "type": "asr_result",
            "text": text
        }))
    
    # 翻译给其他参会者
    target_langs = {
        cid: lang for cid, lang in rooms[room_id]["languages"].items()
        if cid != client_id
    }
    if target_langs:
        tasks = []
        for target_cid, target_lang in target_langs.items():
            tasks.append(
                translate_and_synthesize(text, target_lang, target_cid, room_id, client_id)
            )
        await asyncio.gather(*tasks)

# ---------- 翻译和 TTS（使用 dashscope SDK） ----------
def translate_dashscope(text: str, target_lang: str) -> str:
    """调用阿里百炼翻译 API（使用 requests，因为 dashscope 暂时没有同步翻译接口）"""
    # 实际上要用 requests 调用翻译 API
    import requests
    target = LANG_MAP.get(target_lang, "en")
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen-mt-turbo",
        "input": {
            "text": text,
            "source_lang": "auto",
            "target_lang": target
        }
    }
    try:
        resp = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/machine-translation/translation",
            headers=headers, json=payload, timeout=30
        )
        if resp.status_code == 200:
            result = resp.json()
            return result.get("output", {}).get("text", text)
        else:
            logger.error(f"翻译失败: {resp.text}")
            return text
    except Exception as e:
        logger.error(f"翻译异常: {e}")
        return text

def synthesize_speech(text: str, lang_name: str) -> Optional[bytes]:
    """使用 CosyVoice 合成语音，返回音频字节"""
    if not text or not DASHSCOPE_API_KEY:
        return None
    try:
        # 使用成功案例中的方式
        synthesizer = SpeechSynthesizer(model="cosyvoice-v2", voice="longxiaochun_v2")
        audio_bytes = synthesizer.call(text)
        return audio_bytes
    except Exception as e:
        logger.error(f"TTS 失败: {e}")
        return None

async def translate_and_synthesize(text: str, target_lang: str,
                                   target_client_id: str, room_id: str,
                                   speaker_id: str):
    """翻译并合成语音发送给目标客户端"""
    try:
        # 翻译
        translated = translate_dashscope(text, target_lang)
        logger.info(f"翻译 ({target_lang}): {translated}")
        
        # TTS
        audio_bytes = synthesize_speech(translated, target_lang)
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8') if audio_bytes else ""
        
        # 发送
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
        logger.error(f"翻译合成失败: {e}", exc_info=True)

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
