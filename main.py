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

rooms: Dict[str, Dict] = {}

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.get("/room/{room_id}/status")
async def get_room_status(room_id: str):
    """获取房间状态"""
    if room_id not in rooms:
        return {"count": 0, "clients": []}
    clients = rooms[room_id]["clients"]
    return {
        "count": len(clients),
        "clients": [{"id": cid} for cid in clients.keys()]
    }

@app.websocket("/ws/{room_id}/{client_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, client_id: str):
    # 满员检查
    if room_id in rooms and len(rooms[room_id]["clients"]) >= 4:
        await websocket.accept()
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "会议已满（最多4人）"
        }))
        await websocket.close(code=1008)
        return

    await websocket.accept()
    logger.info(f"✅ {client_id} 加入房间 {room_id}")

    if room_id not in rooms:
        rooms[room_id] = {"clients": {}, "languages": {}}
    rooms[room_id]["clients"][client_id] = websocket
    rooms[room_id]["languages"][client_id] = "zh"

    await asyncio.sleep(0.1)
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
                    await broadcast_room_status(room_id)

            elif msg_type == "set_region":
                logger.info(f"   {client_id} 地域: {message.get('region')}")

            elif msg_type == "audio":
                audio_b64 = message.get("audio", "")
                if not audio_b64 or room_id not in rooms:
                    continue
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
        if room_id in rooms:
            rooms[room_id]["clients"].pop(client_id, None)
            rooms[room_id]["languages"].pop(client_id, None)
            if not rooms[room_id]["clients"]:
                del rooms[room_id]
            else:
                await broadcast_room_status(room_id)

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
