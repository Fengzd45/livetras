import os
import json
import asyncio
import logging
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# 房间数据结构: { room_id: { "clients": { client_id: websocket }, "languages": { client_id: lang } } }
rooms: Dict[str, Dict] = {}

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.websocket("/ws/{room_id}/{client_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, client_id: str):
    await websocket.accept()
    logger.info(f"✅ {client_id} 加入房间 {room_id}")

    # 初始化房间
    if room_id not in rooms:
        rooms[room_id] = {"clients": {}, "languages": {}}
    rooms[room_id]["clients"][client_id] = websocket
    rooms[room_id]["languages"][client_id] = "zh"  # 默认中文

    # 🔥 关键修复：立即向新客户端发送当前房间状态
    await send_room_status_to_client(room_id, websocket)

    # 广播给所有客户端（包括新加入的）
    await broadcast_room_status(room_id)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "set_language":
                target_lang = message.get("target_lang", "zh")
                if room_id in rooms:
                    rooms[room_id]["languages"][client_id] = target_lang
                    logger.info(f"   {client_id} 目标语言: {target_lang}")
                    await broadcast_room_status(room_id)

            elif msg_type == "set_region":
                logger.info(f"   {client_id} 地域: {message.get('region')}")

            elif msg_type == "audio":
                audio_b64 = message.get("audio", "")
                if not audio_b64 or room_id not in rooms:
                    continue

                # 把音频广播给其他人
                for cid, ws in rooms[room_id]["clients"].items():
                    if cid != client_id:
                        try:
                            await ws.send_text(json.dumps({
                                "type": "translation",
                                "from": client_id,
                                "text": "[翻译] 收到音频",
                                "audio": audio_b64
                            }))
                        except:
                            pass

    except WebSocketDisconnect:
        logger.info(f"❌ {client_id} 断开连接")
    finally:
        # 清理断开连接的客户端
        if room_id in rooms:
            rooms[room_id]["clients"].pop(client_id, None)
            rooms[room_id]["languages"].pop(client_id, None)
            if not rooms[room_id]["clients"]:
                del rooms[room_id]
        # 广播更新后的状态
        if room_id in rooms:
            await broadcast_room_status(room_id)

async def send_room_status_to_client(room_id: str, websocket: WebSocket):
    """只向指定的客户端发送房间状态"""
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

    try:
        await websocket.send_text(json.dumps(status))
        logger.info(f"   向新客户端发送房间状态: {status['clients']}")
    except:
        pass

async def broadcast_room_status(room_id: str):
    """向房间内所有客户端广播当前状态（人数、姓名列表）"""
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

    # 广播给所有客户端
    for ws in clients.values():
        try:
            await ws.send_text(json.dumps(status))
        except:
            pass
