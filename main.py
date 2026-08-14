import os
import json
import base64
import asyncio
import logging
import requests
from typing import Dict
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 配置 ---
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
if not SILICONFLOW_API_KEY:
    logger.warning("⚠️ 环境变量 SILICONFLOW_API_KEY 未设置！")

# 硅基流动 API 端点
SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"
ASR_URL = f"{SILICONFLOW_BASE}/audio/transcriptions"
LLM_URL = f"{SILICONFLOW_BASE}/chat/completions"
TTS_URL = f"{SILICONFLOW_BASE}/audio/speech"

# 使用的模型（可自行更换）
ASR_MODEL = "FunAudioLLM/SenseVoiceSmall"          # 修正：原为 SenseVoic。eSmall
LLM_MODEL = "deepseek-ai/DeepSeek-V3"  # 或 "Qwen/Qwen-3-8B" 等
TTS_MODEL = "fnlp/MOSS-TTSD-v0.5"
TTS_VOICE = "fnlp/MOSS-TTSD-v0.5:alex"  # 音色

# --- 应用状态 ---
rooms: Dict[str, Dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 同声传译服务器启动（硅基流动版）")
    yield
    logger.info("🛑 服务器关闭")

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

# --- WebSocket 端点 ---
@app.websocket("/ws/{room_id}/{client_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, client_id: str):
    # 修复：此处必须是单独一行，不能有 pass 或其他语句
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
                audio_b64 = message.get("audio", "")
                if not audio_b64:
                    continue

                # 获取该房间所有客户端的翻译目标语言（排除自己）
                target_langs = {
                    cid: lang for cid, lang in rooms[room_id]["languages"].items()
                    if cid != client_id
                }
                if not target_langs:
                    continue

                # 异步处理：先识别文字，再分别翻译+合成
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

# --- 核心处理函数 ---
async def process_audio_and_translate(audio_b64: str, target_langs: Dict[str, str],
                                      room_id: str, speaker_id: str):
    """
    1. 调用 ASR 识别文字
    2. 对每个目标语言调用 LLM 翻译，再调用 TTS 合成
    3. 将翻译结果（文字+音频）发送给对应客户端
    """
    try:
        # 解码 Base64 音频（16kHz 16bit PCM）
        pcm_bytes = base64.b64decode(audio_b64)

        # 构建 WAV 头（便于 ASR 接口识别）
        wav_data = build_wav_header(len(pcm_bytes), sample_rate=16000) + pcm_bytes

        # ---- 1. 语音识别 ----
        # 修正：model 应作为表单数据（data），而非文件
        files = {"file": ("audio.wav", wav_data, "audio/wav")}
        data = {"model": ASR_MODEL}
        asr_headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}"}
        asr_resp = requests.post(ASR_URL, headers=asr_headers, files=files, data=data)
        if asr_resp.status_code != 200:
            logger.error(f"ASR 失败: {asr_resp.text}")
            return
        asr_result = asr_resp.json()
        original_text = asr_result.get("text", "").strip()
        if not original_text:
            logger.warning("ASR 返回空文本")
            return
        logger.info(f"识别文字: {original_text}")

        # ---- 2. 对每个目标语言并行处理 ----
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
    """
    翻译文字并合成语音，发送给指定客户端
    """
    try:
        # ---- 2a. 翻译 (LLM) ----
        prompt = f"将以下内容翻译成{target_lang}，只输出翻译结果：\n{text}"
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

        # ---- 2b. 语音合成 (TTS) ----
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

        # ---- 2c. 发送给客户端 ----
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
    """生成 WAV 文件头"""
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
