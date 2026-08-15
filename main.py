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
import speech_recognition as sr
from pydub import AudioSegment
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 配置 ---
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
if not SILICONFLOW_API_KEY:
    logger.warning("⚠️ 环境变量 SILICONFLOW_API_KEY 未设置！")

SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"
LLM_URL = f"{SILICONFLOW_BASE}/chat/completions"
TTS_URL = f"{SILICONFLOW_BASE}/audio/speech"

LLM_MODEL = "deepseek-ai/DeepSeek-V3"
TTS_MODEL = "fnlp/MOSS-TTSD-v0.5"
TTS_VOICE = "fnlp/MOSS-TTSD-v0.5:alex"

# Google 语音识别语言代码
RECOGNIZER_LANG = "zh-CN"  # 中文简体

rooms: Dict[str, Dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 同声传译服务器启动（Google ASR 版）")
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

def recognize_speech(wav_data: bytes) -> str:
    """使用 Google Speech Recognition 识别音频"""
    recognizer = sr.Recognizer()
    try:
        audio_segment = AudioSegment.from_wav(io.BytesIO(wav_data))
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        with sr.AudioFile(wav_io) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language=RECOGNIZER_LANG)
        return text
    except sr.UnknownValueError:
        logger.warning("Google ASR 无法识别语音")
        return ""
    except sr.RequestError as e:
        logger.error(f"Google ASR 请求失败: {e}")
        return ""
    except Exception as e:
        logger.error(f"ASR 处理异常: {e}")
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
