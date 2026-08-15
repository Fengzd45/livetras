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
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 配置 ---
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
if not SILICONFLOW_API_KEY:
    logger.warning("⚠️ 环境变量 SILICONFLOW_API_KEY 未设置！")

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    logger.warning("⚠️ 环境变量 HF_TOKEN 未设置！")

SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"
LLM_URL = f"{SILICONFLOW_BASE}/chat/completions"
TTS_URL = f"{SILICONFLOW_BASE}/audio/speech"

LLM_MODEL = "deepseek-ai/DeepSeek-V3"
TTS_MODEL = "fnlp/MOSS-TTSD-v0.5"
TTS_VOICE = "fnlp/MOSS-TTSD-v0.5:alex"

# HuggingFace Whisper API 配置
HF_WHISPER_API = "https://api-inference.huggingface.co/models/openai/whisper-small"

rooms: Dict[str, Dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 同声传译服务器启动（HuggingFace Whisper 版）")
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

# ----- 使用 HuggingFace Whisper 进行语音识别 -----
def recognize_speech(wav_data: bytes) -> str:
    """使用 HuggingFace 推理 API (Whisper) 识别音频"""
    if not HF_TOKEN:
        logger.error("HF_TOKEN 环境变量未设置，无法识别")
        return ""
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        # 注意：API 期望直接发送音频二进制数据，需指定 Content-Type
        # 但 requests 默认会使用 'application/octet-stream'，可以
        response = requests.post(
            HF_WHISPER_API,
            headers=headers,
            data=wav_data,
            timeout=60
        )
        if response.status_code == 200:
            result = response.json()
            text = result.get("text", "").strip()
            if text:
                logger.info(f"HF Whisper 识别结果: {text}")
                return text
            else:
                logger.warning("HF Whisper 返回空文本")
                return ""
        else:
            logger.error(f"HF API 失败: {response.status_code} {response.text}")
            # 如果是 503 或 504，可能是模型加载中，可稍后重试
            return ""
    except Exception as e:
        logger.error(f"HF ASR 异常: {e}")
        return ""

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
        logger.error(f"仅识别处理失败: {e}")

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
        logger.error(f"处理音频失败: {e}")

async def translate_and_synthesize(text: str, target_lang: str,
                                   target_client_id: str, room_id: str,
                                   speaker_id: str):
    try:
        lang_names = {
            "zh": "中文",
            "en": "英文",
            "ja": "日语",
            "ko": "韩语",
            "fr": "法语",
            "de": "德语",
            "es": "西班牙语",
            "ru": "俄语"
        }
        lang_name = lang_names.get(target_lang, target_lang)
        
        prompt = f"将以下内容翻译成{lang_name}，只输出翻译结果：\n{text}"
        llm_headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json"
        }
        llm_data = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        llm_resp = requests.post(LLM_URL, headers=llm_headers, json=llm_data)
        if llm_resp.status_code != 200:
            logger.error(f"LLM 翻译失败: {llm_resp.text}")
            return
        llm_result = llm_resp.json()
        translated_text = llm_result["choices"][0]["message"]["content"].strip()
        logger.info(f"翻译 ({target_lang}): {translated_text}")

        tts_headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json"
        }
        tts_data = {
            "model": TTS_MODEL,
            "input": translated_text,
            "voice": TTS_VOICE,
            "response_format": "wav",
            "stream": False
        }
        tts_resp = requests.post(TTS_URL, headers=tts_headers, json=tts_data)
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
        logger.error(f"翻译合成失败 (目标 {target_lang}): {e}")

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
