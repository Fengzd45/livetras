import os
import json
import time
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

# 所有阻塞式网络调用（requests / dashscope SDK 同步接口）都丢进这个线程池执行，
# 避免卡住 asyncio 的单个事件循环，影响其他房间/其他用户。
EXECUTOR_MAX_WORKERS = 16

# 音频保活看门狗：前端理论上会持续发送音频帧（哪怕是静音），
# 但如果因为网络问题、浏览器切后台、麦克风异常等原因导致
# 前端真的断供，阿里云 ASR 流式连接会在约 23 秒无数据后自己超时断开。
# 与其被动等阿里云那边判超时，不如服务端自己主动监控每个连接
# "多久没收到音频帧了"，提前重建 ASR 会话，把断线时间压到最短。
WATCHDOG_CHECK_INTERVAL = 5    # 每隔几秒检查一次
WATCHDOG_IDLE_TIMEOUT = 15     # 超过这么久没收到音频帧，判定为需要重建（留出安全余量，早于阿里云的 23 秒）


# ---------- ASR 回调 ----------
class StreamingCallback(RecognitionCallback):
    def __init__(self, client_id: str, loop: asyncio.AbstractEventLoop, room_id: str):
        self.client_id = client_id
        self.loop = loop
        self.room_id = room_id
        self.is_broken = False  # 会话是否已经失效（断开/出错），供主循环主动检测并重建

    def on_open(self):
        logger.info(f"ASR 流式会话已建立: {self.client_id}")

    def on_close(self):
        # 正常/异常关闭都要标记为 broken，否则外层只在 on_error 时才会尝试重建，
        # 漏掉了"连接被服务端正常关闭但我们还想继续说话"的情况。
        self.is_broken = True
        logger.info(f"ASR 流式会话已关闭: {self.client_id}")

    def on_complete(self):
        logger.info(f"ASR 流式会话正常结束: {self.client_id}")

    def on_error(self, result):
        self.is_broken = True
        # 修复：原来 try 块里 str(result) 本身可能抛异常，
        # 一旦走进 except 分支 error_msg 还没被赋值，就会在下面
        # self._notify_error(error_msg) 触发 UnboundLocalError，
        # 导致这条日志之后的清理/通知代码全部执行不到，
        # 且异常发生在 dashscope 内部的后台线程里不会被任何人捕获。
        error_msg = "未知错误"
        try:
            error_msg = str(result) if result else "未知错误"
            if hasattr(result, 'status_code'):
                error_msg = f"status_code={result.status_code}, {error_msg}"
            if hasattr(result, 'message'):
                error_msg = f"message={result.message}, {error_msg}"
        except Exception as e:
            error_msg = f"无法解析错误详情: {e}"
        logger.error(f"ASR 错误: {error_msg}")

        try:
            asyncio.run_coroutine_threadsafe(
                self._notify_error(error_msg),
                self.loop
            )
        except Exception as e:
            logger.error(f"通知前端 ASR 错误失败: {e}")

    async def _notify_error(self, msg):
        room_id = self.room_id
        client_id = self.client_id
        if room_id in rooms and client_id in rooms[room_id]["clients"]:
            ws = rooms[room_id]["clients"][client_id]
            try:
                await ws.send_text(json.dumps({
                    "type": "asr_error",
                    "msg": f"ASR 错误: {msg}"
                }))
            except Exception:
                pass

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
        else:
            logger.info(f"ASR 中间结果 [{self.client_id}]: '{text}'")

        # 中间结果只用于实时字幕展示（发给说话者自己），不触发翻译/TTS；
        # 只有整句说完（sentence_end=True）才对其他人做翻译+合成。
        # 之前的版本对每一个中间结果都会调用一次翻译API+TTS API，
        # 一句话说到一半可能已经打了五六次请求，费用高、延迟大、
        # 播放的语音还会前后重叠。
        asyncio.run_coroutine_threadsafe(
            handle_asr_result(self.client_id, text, self.room_id, is_end),
            self.loop
        )


# ---------- 翻译（阻塞调用，需在线程池中执行） ----------
def _translate_text_blocking(text: str, target_lang: str) -> str:
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
        logger.error(f"翻译请求失败: status={resp.status_code}, body={resp.text[:300]}")
        return text
    except Exception as e:
        logger.error(f"翻译异常: {e}")
        return text


async def translate_text(text: str, target_lang: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _translate_text_blocking, text, target_lang)


# ---------- TTS（阻塞调用，需在线程池中执行） ----------
def _synthesize_speech_blocking(text: str) -> Optional[bytes]:
    if not text or not DASHSCOPE_API_KEY:
        return None
    try:
        synthesizer = SpeechSynthesizer(model=TTS_MODEL, voice=TTS_VOICE)
        return synthesizer.call(text)
    except Exception as e:
        logger.error(f"TTS 失败: {e}")
        return None


async def synthesize_speech(text: str) -> Optional[bytes]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _synthesize_speech_blocking, text)


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


def _create_recognition(callback: StreamingCallback) -> Recognition:
    """创建并启动一个新的 ASR 流式会话（阻塞调用）。"""
    recognition = Recognition(
        model=ASR_MODEL,
        format="pcm",
        sample_rate=16000,
        callback=callback,
        enable_intermediate_result=True,
    )
    recognition.start()
    return recognition


def _stop_recognition_blocking(recognition: Recognition):
    try:
        recognition.stop()
    except Exception as e:
        logger.error(f"关闭 ASR 会话失败: {e}")


class ConnectionState:
    """持有某个 WebSocket 连接当前的 ASR 会话状态。
    用一个可变对象包起来，是因为看门狗任务和主消息循环是两个
    并发运行的协程，都需要读取/替换同一个 recognition 对象——
    如果只用局部变量，看门狗那边根本没法把"重建后的新 recognition"
    同步给主循环使用。"""
    __slots__ = ("recognition", "callback", "last_audio_time")

    def __init__(self):
        self.recognition: Optional[Recognition] = None
        self.callback: Optional[StreamingCallback] = None
        self.last_audio_time: float = time.monotonic()


async def rebuild_recognition(state: ConnectionState, client_id: str, room_id: str,
                               loop: asyncio.AbstractEventLoop, reason: str):
    """停掉旧的 ASR 会话（如果有）并重新创建一个。
    start/stop 都是阻塞调用，丢进线程池执行，不卡事件循环。"""
    old_recognition = state.recognition
    if old_recognition is not None:
        await loop.run_in_executor(None, _stop_recognition_blocking, old_recognition)

    new_callback = StreamingCallback(client_id, loop, room_id)
    try:
        new_recognition = await loop.run_in_executor(None, _create_recognition, new_callback)
    except Exception as e:
        logger.error(f"ASR 重建失败（{reason}）: {e}")
        state.recognition = None
        state.callback = new_callback
        return

    state.recognition = new_recognition
    state.callback = new_callback
    state.last_audio_time = time.monotonic()
    logger.info(f"✅ ASR 会话已重建（{reason}）: {client_id}")


async def audio_watchdog(state: ConnectionState, client_id: str, room_id: str,
                          loop: asyncio.AbstractEventLoop):
    """后台常驻任务：定期检查这个连接多久没收到音频帧了。
    如果前端因为某种原因真的停止发送音频（网络问题、切后台、
    麦克风掉线等），与其等阿里云那边约 23 秒超时才发现，
    不如服务端自己提前判定并重建，缩短用户感知到的中断时间。"""
    try:
        while True:
            await asyncio.sleep(WATCHDOG_CHECK_INTERVAL)

            if state.recognition is None:
                # 还没建立过，或者上一次重建失败了——留给主循环下次
                # 收到音频帧时自然触发重建，看门狗这里不重复处理。
                continue

            idle = time.monotonic() - state.last_audio_time
            if idle > WATCHDOG_IDLE_TIMEOUT and not (state.callback and state.callback.is_broken):
                logger.warning(
                    f"⏰ {client_id} 已 {idle:.1f}s 未收到音频帧，"
                    f"主动重建 ASR 会话（避免被动等待阿里云侧超时）"
                )
                await rebuild_recognition(state, client_id, room_id, loop, reason="看门狗检测到音频中断")
    except asyncio.CancelledError:
        pass


@app.websocket("/ws/{room_id}/{client_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, client_id: str):
    await websocket.accept()
    logger.info(f"✅ 客户端 {client_id} 加入房间 {room_id}")

    if room_id not in rooms:
        rooms[room_id] = {"clients": {}, "languages": {}}
    rooms[room_id]["clients"][client_id] = websocket
    await broadcast_room_status(room_id)

    loop = asyncio.get_running_loop()

    state = ConnectionState()
    try:
        state.callback = StreamingCallback(client_id, loop, room_id)
        state.recognition = await loop.run_in_executor(None, _create_recognition, state.callback)
        state.last_audio_time = time.monotonic()
        logger.info(f"✅ ASR 会话已创建: {client_id}")
        await websocket.send_text(json.dumps({"type": "asr_ready", "msg": "语音识别已就绪"}))
    except Exception as e:
        logger.error(f"ASR 启动失败: {e}")
        await websocket.send_text(json.dumps({"type": "asr_error", "msg": f"ASR 启动失败: {e}"}))

    watchdog_task = asyncio.create_task(audio_watchdog(state, client_id, room_id, loop))

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
                state.last_audio_time = time.monotonic()

                # 主动检测：一旦上次回调标记会话已失效（断线/出错/正常关闭），
                # 在发送下一帧音频之前立刻重建，而不是等 send_audio_frame
                # 抛出异常才被动发现。这一路是"还有音频在发、但会话坏了"的情况；
                # 下面的 audio_watchdog 负责另一种情况："音频根本没发过来"。
                if state.recognition is None or state.callback.is_broken:
                    await rebuild_recognition(state, client_id, room_id, loop, reason="主动检测")
                    if state.recognition is None:
                        continue

                try:
                    await loop.run_in_executor(None, state.recognition.send_audio_frame, pcm_bytes)
                except Exception as e:
                    logger.error(f"发送音频失败: {e}")
                    state.callback.is_broken = True
                    # 不在这里同步重建，下一帧进来时会走上面的主动检测分支，
                    # 避免在异常处理路径里再做一次可能失败的重建。

    except WebSocketDisconnect:
        logger.info(f"❌ 客户端 {client_id} 断开连接")
    finally:
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass

        if state.recognition:
            await loop.run_in_executor(None, _stop_recognition_blocking, state.recognition)

        if room_id in rooms:
            rooms[room_id]["clients"].pop(client_id, None)
            rooms[room_id]["languages"].pop(client_id, None)
            if not rooms[room_id]["clients"]:
                await asyncio.sleep(5)
                if room_id in rooms and not rooms[room_id]["clients"]:
                    del rooms[room_id]
            else:
                await broadcast_room_status(room_id)


# ---------- 处理识别结果 ----------
async def handle_asr_result(client_id: str, text: str, room_id: str, is_end: bool):
    if room_id not in rooms:
        logger.warning(f"房间 {room_id} 不存在")
        return

    if client_id not in rooms[room_id]["clients"]:
        logger.warning(f"客户端 {client_id} 已离开")
        return

    # 无论是不是整句结束，都把文字实时发给说话者自己看（字幕滚动效果）
    speaker_ws = rooms[room_id]["clients"][client_id]
    try:
        await speaker_ws.send_text(json.dumps({
            "type": "asr_result",
            "text": text,
            "is_end": is_end
        }))
    except Exception as e:
        logger.error(f"发送识别结果失败: {e}")

    # 只有整句说完才翻译+合成语音发给其他人
    if not is_end:
        return

    target_langs = {
        cid: lang for cid, lang in rooms[room_id]["languages"].items()
        if cid != client_id
    }
    if target_langs:
        await asyncio.gather(*[
            translate_and_synthesize(text, target_lang, target_cid, room_id, client_id)
            for target_cid, target_lang in target_langs.items()
        ])


# ---------- 翻译 + TTS ----------
async def translate_and_synthesize(text: str, target_lang: str,
                                   target_client_id: str, room_id: str,
                                   speaker_id: str):
    try:
        translated = await translate_text(text, target_lang)
        logger.info(f"翻译 ({target_lang}): {translated}")

        audio_bytes = await synthesize_speech(translated)
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
        except Exception:
            pass
