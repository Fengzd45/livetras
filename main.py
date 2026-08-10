import os
import json
import base64
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

@app.websocket("/ws/{room_id}/{client_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, client_id: str):
    await websocket.accept()
    logger.info(f"✅ {client_id} 加入房间 {room_id}")
    
    if room_id not in rooms:
        rooms[room_id] = {"clients": {}}
    rooms[room_id]["clients"][client_id] = websocket
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")
            
            if msg_type == "audio":
                # 简单回显：把收到的音频广播给其他人
                for cid, ws in rooms[room_id]["clients"].items():
                    if cid != client_id:
                        try:
                            await ws.send_text(json.dumps({
                                "type": "translation",
                                "from": client_id,
                                "text": "[翻译测试] 收到你的音频",
                                "audio": message.get("audio", "")
                            }))
                        except:
                            pass
            elif msg_type == "set_language":
                logger.info(f"   {client_id} 设置语言: {message.get('target_lang')}")
            elif msg_type == "set_region":
                logger.info(f"   {client_id} 设置地域: {message.get('region')}")
                
    except WebSocketDisconnect:
        logger.info(f"❌ {client_id} 断开连接")
    finally:
        if room_id in rooms:
            rooms[room_id]["clients"].pop(client_id, None)
            if not rooms[room_id]["clients"]:
                del rooms[room_id]
