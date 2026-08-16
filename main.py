import os
import json
import base64
import asyncio
import logging
import requests
from typing import Dict
from fastapi import FastAPI, WebSocketDisconnect, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 配置（阿里百炼） ---
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    logger.warning("⚠️ 环境变量 DASHSCOPE_API_KEY 未设置！")

# 阿里百炼 API 地址（非流式）
ASR_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
TRANSLATE_URL = "https://dashscope.aliyuncs.com/api/v1/services/machine-translation/translation"
TTS_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/speech"

ASR_MODEL = "fun-asr"                # 或 qwen-asr
TRANSLATE_MODEL = "qwen-mt-turbo"
TTS_MODEL = "cosyvoice-v2"
TTS_VOICE = "default"

# 语言映射（前端 -> 翻译目标代码）
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 同声传译服务器启动（阿里百炼版）")
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
    await broadcast_room_status(room_id)

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

                target_langs = {
                    cid: lang for cid, lang in rooms[room_id]["languages"].items()
                    if cid != client_id
                }
                
                if not target_langs:
                    asyncio.create_task(
                        process_audio_only(audio_b64, room_id, client_id)
                    )
                else:
                    asyncio.create_task(
                        process_audio_and_translate(
                            audio_b64, target_langs, room_id, client_id
                        )
                    )

    except WebSocketDisconnect:
        logger.info(f"❌ 客户端 {client_id} 断开连接")
    finally:
        if room_id in rooms:
            rooms[room_id]["clients"].pop(client_id, None)
            rooms[room_id]["languages"].pop(client_id, None)
            if not rooms[room_id]["clients"]:
                del rooms[room_id]
        await broadcast_room_status(room_id)

# ----- 使用阿里百炼 ASR（非流式 HTTP，base64）-----
def recognize_speech(wav_data: bytes) -> str:
    """使用阿里百炼 ASR 识别音频"""
    if not DASHSCOPE_API_KEY:
        logger.error("DASHSCOPE_API_KEY 未设置")
        return ""

    audio_b64 = base64.b64encode(wav_data).decode('utf-8')
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": ASR_MODEL,
        "input": {
            "audio": audio_b64,
            "format": "wav",
            "sample_rate": 16000
        },
        "parameters": {
            "enable_punctuation": True,
            "enable_vad": True
        }
    }

    try:
        response = requests.post(ASR_URL, headers=headers, json=payload, timeout=60)
        logger.info(f"ASR 响应状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            # 打印完整响应用于调试
            logger.info(f"ASR 响应内容: {json.dumps(result, ensure_ascii=False)}")
            text = result.get("output", {}).get("text", "").strip()
            if text:
                logger.info(f"✅ ASR 识别结果: {text}")
                return text
            else:
                logger.warning("ASR 返回空文本")
                return ""
        else:
            logger.error(f"ASR API 失败: {response.status_code} - {response.text}")
            return ""
    except Exception as e:
        logger.error(f"ASR 异常: {e}", exc_info=True)
        return ""

# ----- 以下三个函数保持原逻辑（仅调用 recognize_speech）-----
async def process_audio_only(audio_b64: str, room_id: str, speaker_id: str):
    try:
        pcm_bytes = base64.b64decode(audio_b64)
        wav_data = build_wav_header(len(pcm_bytes), sample_rate=16000) + pcm_bytes

        original_text = recognize_speech(wav_data)
        if not original_text:
            return
        logger.info(f"识别文字 (仅自己): {original_text}")

        if room_id in rooms and speaker_id in rooms[room_id]["clients"]:
            speaker_ws = rooms[room_id]["clients"][speaker_id]
            await speaker_ws.send_text(json.dumps({
                "type": "asr_result",
                "text": original_text
            }))
            logger.info(f"✅ 已向 {speaker_id} 发送识别结果: {original_text}")

    except Exception as e:
        logger.error(f"仅识别处理失败: {e}", exc_info=True)

async def process_audio_and_translate(audio_b64: str, target_langs: Dict[str, str],
                                      room_id: str, speaker_id: str):
    try:
        pcm_bytes = base64.b64decode(audio_b64)
        wav_data = build_wav_header(len(pcm_bytes), sample_rate=16000) + pcm_bytes

        original_text = recognize_speech(wav_data)
        if not original_text:
            return
        logger.info(f"识别文字: {original_text}")

        if room_id in rooms and speaker_id in rooms[room_id]["clients"]:
            speaker_ws = rooms[room_id]["clients"][speaker_id]
            await speaker_ws.send_text(json.dumps({
                "type": "asr_result",
                "text": original_text
            }))
            logger.info(f"✅ 已向 {speaker_id} 发送识别结果: {original_text}")

        tasks = []
        for target_client_id, target_lang in target_langs.items():
            tasks.append(
                translate_and_synthesize(
                    original_text, target_lang, target_client_id, room_id, speaker_id
                )
            )
        await asyncio.gather(*tasks)

    except Exception as e:
        logger.error(f"处理音频失败: {e}", exc_info=True)

# ----- 翻译和 TTS 改用阿里百炼 -----
async def translate_and_synthesize(text: str, target_lang: str,
                                   target_client_id: str, room_id: str,
                                   speaker_id: str):
    try:
        # 1. 翻译
        target = LANG_MAP.get(target_lang, "en")
        trans_headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json"
        }
        trans_payload = {
            "model": TRANSLATE_MODEL,
            "input": {
                "text": text,
                "source_lang": "auto",
                "target_lang": target
            }
        }
        trans_resp = requests.post(TRANSLATE_URL, headers=trans_headers, json=trans_payload, timeout=30)
        if trans_resp.status_code != 200:
            logger.error(f"翻译失败: {trans_resp.text}")
            return
        trans_result = trans_resp.json()
        translated_text = trans_result.get("output", {}).get("text", text).strip()
        logger.info(f"翻译 ({target_lang}): {translated_text}")

        # 2. TTS
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
        logger.error(f"翻译合成失败 (目标 {target_lang}): {e}", exc_info=True)

# ----- build_wav_header 保持不变 -----
def build_wav_header(data_len: int, sample_rate: int = 16000,
                     channels: int = 1, bits_per_sample: int = 16) -> bytes:
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    header = bytearray()
    header.extend(b'RIFF')
    header.extend((data_len + 36).to_bytes(4, 'little'))
    header.extend(b'WAVE')
    header.extend(b'fmt ')
    header.extend((16).to_bytes(4, 'little'))
    header.extend((1).to_bytes(2, 'little'))
    header.extend(channels.to_bytes(2, 'little'))
    header.extend(sample_rate.to_bytes(4, 'little'))
    header.extend(byte_rate.to_bytes(4, 'little'))
    header.extend(block_align.to_bytes(2, 'little'))
    header.extend(bits_per_sample.to_bytes(2, 'little'))
    header.extend(b'data')
    header.extend(data_len.to_bytes(4, 'little'))
    return bytes(header)

# ----- broadcast_room_status 保持不变 -----
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
