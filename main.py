import os
import json
import base64
import asyncio
import logging
from typing import Dict, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import websockets
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 配置 ---
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    logger.warning("⚠️ 环境变量 DASHSCOPE_API_KEY 未设置！")

# 百炼 WebSocket 地址 (北京地域)
BAILIAN_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"

# --- 应用状态 ---
# 存储所有房间: { room_id: { "clients": { client_id: websocket }, "languages": { client_id: target_lang } } }
rooms: Dict[str, Dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    logger.info("🚀 同声传译服务器启动")
    yield
    # 关闭时
    logger.info("🛑 服务器关闭")

app = FastAPI(lifespan=lifespan)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    """返回前端页面"""
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

# --- WebSocket 端点 ---
@app.websocket("/ws/{room_id}/{client_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, client_id: str):
    """处理客户端 WebSocket 连接"""
    await websocket.accept()
    logger.info(f"✅ 客户端 {client_id} 加入房间 {room_id}")

    # 初始化房间
    if room_id not in rooms:
        rooms[room_id] = {"clients": {}, "languages": {}}
    rooms[room_id]["clients"][client_id] = websocket

    # 通知其他客户端有新用户加入
    await broadcast_room_status(room_id)

    try:
        while True:
            # 接收客户端消息 (JSON 格式)
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "set_language":
                # 客户端设置目标语言
                target_lang = message.get("target_lang", "en")
                rooms[room_id]["languages"][client_id] = target_lang
                logger.info(f"   {client_id} 目标语言: {target_lang}")
                await broadcast_room_status(room_id)

            elif msg_type == "audio":
                # 收到音频数据 (Base64 编码的 PCM)
                audio_b64 = message.get("audio", "")
                if not audio_b64:
                    continue

                # 获取该房间所有客户端的翻译目标语言
                target_langs = rooms[room_id]["languages"]
                if not target_langs:
                    continue

                # 为每个目标语言调用百炼 API 进行翻译
                # 注意: 实际场景中应优化为同一音频流复用，此处为简化示例
                for target_client_id, target_lang in target_langs.items():
                    # 不翻译给自己
                    if target_client_id == client_id:
                        continue
                    # 异步调用百炼 API
                    asyncio.create_task(
                        translate_and_send(
                            audio_b64,
                            target_lang,
                            target_client_id,
                            room_id
                        )
                    )

    except WebSocketDisconnect:
        logger.info(f"❌ 客户端 {client_id} 断开连接")
    finally:
        # 清理断开连接的客户端
        if room_id in rooms:
            rooms[room_id]["clients"].pop(client_id, None)
            rooms[room_id]["languages"].pop(client_id, None)
            if not rooms[room_id]["clients"]:
                del rooms[room_id]
        await broadcast_room_status(room_id)

# --- 辅助函数 ---
async def translate_and_send(audio_b64: str, target_lang: str, target_client_id: str, room_id: str):
    """调用百炼 API 翻译音频并发送给指定客户端"""
    try:
        # 1. 连接百炼 WebSocket API[reference:2][reference:3]
        headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
        async with websockets.connect(BAILIAN_WS_URL, extra_headers=headers) as bailian_ws:
            # 发送会话配置 (自动识别源语言)
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "input_audio_format": "pcm",
                    "input_audio_transcription": {"model": "qwen3-asr-flash-realtime", "language": "auto"},
                    "output_audio_format": "pcm_24000hz_mono_16bit",
                    "translation": {"target_language": target_lang, "source_language": "auto"},
                    "voice": "Cherry"
                }
            }
            await bailian_ws.send(json.dumps(session_config))

            # 发送音频数据
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            }
            await bailian_ws.send(json.dumps(audio_event))

            # 发送结束信号
            await bailian_ws.send(json.dumps({"type": "input_audio_buffer.finish"}))

            # 2. 接收翻译结果
            translated_audio_b64 = None
            transcript_text = ""

            async for message in bailian_ws:
                event = json.loads(message)
                event_type = event.get("type")

                if event_type == "response.audio.delta":
                    # 收集翻译后的音频数据
                    delta = event.get("delta", "")
                    if delta:
                        translated_audio_b64 = delta  # 实际应拼接多个 delta
                elif event_type == "response.transcript.done":
                    transcript_text = event.get("transcript", "")
                elif event_type == "session.finished":
                    break

            # 3. 将翻译结果发送给目标客户端
            if room_id in rooms and target_client_id in rooms[room_id]["clients"]:
                target_ws = rooms[room_id]["clients"][target_client_id]
                response = {
                    "type": "translation",
                    "from": "speaker",  # 可替换为说话人标识
                    "text": transcript_text,
                    "audio": translated_audio_b64,  # Base64 编码的 PCM 音频
                    "lang": target_lang
                }
                await target_ws.send_text(json.dumps(response))

    except Exception as e:
        logger.error(f"翻译失败: {e}")

async def broadcast_room_status(room_id: str):
    """广播房间状态更新"""
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
            passimport os
import json
import base64
import asyncio
import logging
from typing import Dict, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import websockets
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 配置 ---
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    logger.warning("⚠️ 环境变量 DASHSCOPE_API_KEY 未设置！")

# 百炼 WebSocket 地址 (北京地域)
BAILIAN_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"

# --- 应用状态 ---
# 存储所有房间: { room_id: { "clients": { client_id: websocket }, "languages": { client_id: target_lang } } }
rooms: Dict[str, Dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    logger.info("🚀 同声传译服务器启动")
    yield
    # 关闭时
    logger.info("🛑 服务器关闭")

app = FastAPI(lifespan=lifespan)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    """返回前端页面"""
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

# --- WebSocket 端点 ---
@app.websocket("/ws/{room_id}/{client_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, client_id: str):
    """处理客户端 WebSocket 连接"""
    await websocket.accept()
    logger.info(f"✅ 客户端 {client_id} 加入房间 {room_id}")

    # 初始化房间
    if room_id not in rooms:
        rooms[room_id] = {"clients": {}, "languages": {}}
    rooms[room_id]["clients"][client_id] = websocket

    # 通知其他客户端有新用户加入
    await broadcast_room_status(room_id)

    try:
        while True:
            # 接收客户端消息 (JSON 格式)
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "set_language":
                # 客户端设置目标语言
                target_lang = message.get("target_lang", "en")
                rooms[room_id]["languages"][client_id] = target_lang
                logger.info(f"   {client_id} 目标语言: {target_lang}")
                await broadcast_room_status(room_id)

            elif msg_type == "audio":
                # 收到音频数据 (Base64 编码的 PCM)
                audio_b64 = message.get("audio", "")
                if not audio_b64:
                    continue

                # 获取该房间所有客户端的翻译目标语言
                target_langs = rooms[room_id]["languages"]
                if not target_langs:
                    continue

                # 为每个目标语言调用百炼 API 进行翻译
                # 注意: 实际场景中应优化为同一音频流复用，此处为简化示例
                for target_client_id, target_lang in target_langs.items():
                    # 不翻译给自己
                    if target_client_id == client_id:
                        continue
                    # 异步调用百炼 API
                    asyncio.create_task(
                        translate_and_send(
                            audio_b64,
                            target_lang,
                            target_client_id,
                            room_id
                        )
                    )

    except WebSocketDisconnect:
        logger.info(f"❌ 客户端 {client_id} 断开连接")
    finally:
        # 清理断开连接的客户端
        if room_id in rooms:
            rooms[room_id]["clients"].pop(client_id, None)
            rooms[room_id]["languages"].pop(client_id, None)
            if not rooms[room_id]["clients"]:
                del rooms[room_id]
        await broadcast_room_status(room_id)

# --- 辅助函数 ---
async def translate_and_send(audio_b64: str, target_lang: str, target_client_id: str, room_id: str):
    """调用百炼 API 翻译音频并发送给指定客户端"""
    try:
        # 1. 连接百炼 WebSocket API[reference:2][reference:3]
        headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
        async with websockets.connect(BAILIAN_WS_URL, extra_headers=headers) as bailian_ws:
            # 发送会话配置 (自动识别源语言)
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "input_audio_format": "pcm",
                    "input_audio_transcription": {"model": "qwen3-asr-flash-realtime", "language": "auto"},
                    "output_audio_format": "pcm_24000hz_mono_16bit",
                    "translation": {"target_language": target_lang, "source_language": "auto"},
                    "voice": "Cherry"
                }
            }
            await bailian_ws.send(json.dumps(session_config))

            # 发送音频数据
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            }
            await bailian_ws.send(json.dumps(audio_event))

            # 发送结束信号
            await bailian_ws.send(json.dumps({"type": "input_audio_buffer.finish"}))

            # 2. 接收翻译结果
            translated_audio_b64 = None
            transcript_text = ""

            async for message in bailian_ws:
                event = json.loads(message)
                event_type = event.get("type")

                if event_type == "response.audio.delta":
                    # 收集翻译后的音频数据
                    delta = event.get("delta", "")
                    if delta:
                        translated_audio_b64 = delta  # 实际应拼接多个 delta
                elif event_type == "response.transcript.done":
                    transcript_text = event.get("transcript", "")
                elif event_type == "session.finished":
                    break

            # 3. 将翻译结果发送给目标客户端
            if room_id in rooms and target_client_id in rooms[room_id]["clients"]:
                target_ws = rooms[room_id]["clients"][target_client_id]
                response = {
                    "type": "translation",
                    "from": "speaker",  # 可替换为说话人标识
                    "text": transcript_text,
                    "audio": translated_audio_b64,  # Base64 编码的 PCM 音频
                    "lang": target_lang
                }
                await target_ws.send_text(json.dumps(response))

    except Exception as e:
        logger.error(f"翻译失败: {e}")

async def broadcast_room_status(room_id: str):
    """广播房间状态更新"""
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