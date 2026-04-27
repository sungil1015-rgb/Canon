"""WebSocket router for the Canon inspection dashboard.

프론트엔드(App.tsx)가 기대하는 메시지 형식:

메인 WebSocket  ws://host/ws
  서버 → 클라이언트:
    { "type": "camera_list",  "cameras": ["CAM_01", "CAM_02", ...] }
    { "type": "video_frame",  "cameraId": "CAM_01", "frame": "<base64 jpeg>" }
    [                                          ← 배열 형태로 일괄 전송
      { "cameraId": "CAM_01",  "payload": CameraData },
      { "type": "image_log",   "payload": ImageLog   },
      ...
    ]

카메라 소스 WebSocket  ws://host/ws/source
  클라이언트 → 서버:
    "data:image/jpeg;base64,<base64 jpeg>"   ← MobileSourceView 프레임
  서버 처리:
    수신 프레임을 CAM_WS_SOURCE 카메라로 등록하여 추론 루프에 공급
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    import numpy as np


logger = logging.getLogger(__name__)

router = APIRouter()

# ── 설정 상수 ──────────────────────────────────────────────────────────────────
FRAME_BROADCAST_INTERVAL = 0.1   # 프레임 브로드캐스트 주기 (초), 10fps 목표
STATE_BROADCAST_INTERVAL = 0.5   # 추론 상태 브로드캐스트 주기 (초)
JPEG_QUALITY = 60                 # WebSocket 전송용 JPEG 품질
CAM_WS_SOURCE = "CAM_WS"         # MobileSourceView 전용 가상 카메라 ID


# ── 연결 매니저 ───────────────────────────────────────────────────────────────
class ConnectionManager:
    """메인 /ws 에 연결된 모든 클라이언트를 관리합니다."""

    def __init__(self) -> None:
        self._clients: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.append(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients = [c for c in self._clients if c is not ws]

    async def broadcast(self, data: str) -> None:
        """연결된 모든 클라이언트에게 텍스트 메시지를 전송합니다."""
        async with self._lock:
            dead: list[WebSocket] = []
            for client in self._clients:
                try:
                    await client.send_text(data)
                except Exception:
                    dead.append(client)
            for d in dead:
                self._clients = [c for c in self._clients if c is not d]

    def count(self) -> int:
        return len(self._clients)


manager = ConnectionManager()


# ── 카메라 상태 저장소 ─────────────────────────────────────────────────────────
@dataclass
class CameraState:
    """단일 카메라의 현재 추론 상태를 보관합니다."""

    cam_id: str
    predicted_label: str = "unknown"
    confidence: float = 0.0
    confirmed_state: str = "Idle"
    is_unknown: bool = False
    inference: bool = False
    current_step_index: int = 1
    allowed_transition: bool = True
    system_message: str = "WAITING..."
    last_frame: Any = field(default=None, repr=False)
    last_frame_time: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "cameraId": self.cam_id,
            "payload": {
                "predicted_label": self.predicted_label,
                "confidence": self.confidence,
                "confirmed_state": self.confirmed_state,
                "is_unknown": self.is_unknown,
                "inference": self.inference,
                "logic": {
                    "current_step_index": self.current_step_index,
                    "confirmed_state": self.confirmed_state,
                    "allowed_transition": self.allowed_transition,
                    "confidence": self.confidence,
                },
                "display": {
                    "system_message": self.system_message,
                },
            },
        }


class CameraStateStore:
    """등록된 모든 카메라 상태를 스레드-안전하게 관리합니다."""

    def __init__(self) -> None:
        self._cameras: dict[str, CameraState] = {}
        self._lock = threading.Lock()

    def register(self, cam_id: str) -> CameraState:
        with self._lock:
            if cam_id not in self._cameras:
                self._cameras[cam_id] = CameraState(cam_id=cam_id)
            return self._cameras[cam_id]

    def unregister(self, cam_id: str) -> None:
        with self._lock:
            self._cameras.pop(cam_id, None)

    def update_frame(self, cam_id: str, frame_bgr: Any) -> None:
        with self._lock:
            state = self._cameras.get(cam_id)
            if state:
                state.last_frame = frame_bgr.copy()
                state.last_frame_time = time.monotonic()

    def update_inference(self, cam_id: str, **kwargs: Any) -> None:
        with self._lock:
            state = self._cameras.get(cam_id)
            if state:
                for k, v in kwargs.items():
                    if hasattr(state, k):
                        setattr(state, k, v)

    def camera_ids(self) -> list[str]:
        with self._lock:
            return list(self._cameras.keys())

    def get_all_payloads(self) -> list[dict[str, Any]]:
        with self._lock:
            return [s.to_payload() for s in self._cameras.values()]

    def get_frame(self, cam_id: str) -> Any:
        with self._lock:
            state = self._cameras.get(cam_id)
            if state and state.last_frame is not None:
                return state.last_frame.copy()
            return None


# 전역 상태 저장소 (main.py에서 참조 가능)
camera_store = CameraStateStore()


# ── 프레임 → base64 변환 ───────────────────────────────────────────────────────
def _encode_frame(frame_bgr: Any, quality: int = JPEG_QUALITY) -> str:
    """BGR numpy 배열을 base64 JPEG 문자열로 변환합니다."""
    import cv2
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ── 백그라운드 브로드캐스트 루프 ───────────────────────────────────────────────
async def _broadcast_loop() -> None:
    """등록된 카메라의 최신 프레임과 추론 상태를 주기적으로 브로드캐스트합니다."""
    last_state_time = 0.0

    while True:
        await asyncio.sleep(FRAME_BROADCAST_INTERVAL)

        if manager.count() == 0:
            continue

        cam_ids = camera_store.camera_ids()

        # 카메라 목록 변경은 매번 전송하지 않고 상태 업데이트 주기에 맞춤
        now = time.monotonic()

        # ── 프레임 브로드캐스트 ──
        for cam_id in cam_ids:
            frame = camera_store.get_frame(cam_id)
            if frame is None:
                continue
            try:
                encoded = _encode_frame(frame)
                msg = json.dumps(
                    {"type": "video_frame", "cameraId": cam_id, "frame": encoded}
                )
                await manager.broadcast(msg)
            except Exception as exc:
                logger.warning("frame broadcast error for %s: %s", cam_id, exc)

        # ── 추론 상태 + 카메라 목록 브로드캐스트 (STATE_BROADCAST_INTERVAL 주기) ──
        if now - last_state_time >= STATE_BROADCAST_INTERVAL:
            last_state_time = now

            # 카메라 목록
            if cam_ids:
                list_msg = json.dumps({"type": "camera_list", "cameras": cam_ids})
                await manager.broadcast(list_msg)

            # 추론 상태 배열
            payloads = camera_store.get_all_payloads()
            if payloads:
                state_msg = json.dumps(payloads)
                await manager.broadcast(state_msg)


# ── 메인 WebSocket 엔드포인트 ──────────────────────────────────────────────────
@router.websocket("/ws")
async def ws_main(ws: WebSocket) -> None:
    """대시보드 메인 WebSocket.

    연결 직후 현재 카메라 목록을 전송하고, 이후 broadcast_loop 가 지속 전송합니다.
    """
    await manager.connect(ws)
    logger.info("WS client connected. total=%d", manager.count())

    # 연결 직후 즉시 현재 카메라 목록 전송
    cam_ids = camera_store.camera_ids()
    if cam_ids:
        await ws.send_text(
            json.dumps({"type": "camera_list", "cameras": cam_ids})
        )

    try:
        while True:
            # 클라이언트 메시지 수신 (ping/pong 유지 목적)
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(ws)
        logger.info("WS client disconnected. total=%d", manager.count())


# ── 소스(모바일) WebSocket 엔드포인트 ─────────────────────────────────────────
@router.websocket("/ws/source")
async def ws_source(ws: WebSocket) -> None:
    """모바일 카메라 프레임을 수신하여 CAM_WS 카메라로 등록합니다.

    프론트엔드 MobileSourceView 가 'data:image/jpeg;base64,...' 형태로 전송합니다.
    """
    await ws.accept()
    camera_store.register(CAM_WS_SOURCE)

    # 클라이언트에게 카메라 목록 업데이트 알림
    cam_ids = camera_store.camera_ids()
    await manager.broadcast(
        json.dumps({"type": "camera_list", "cameras": cam_ids})
    )
    logger.info("Mobile source connected → %s", CAM_WS_SOURCE)

    try:
        while True:
            data = await ws.receive_text()
            # data:image/jpeg;base64,<...> 형태 파싱
            if "," in data:
                b64 = data.split(",", 1)[1]
            else:
                b64 = data
            try:
                import cv2
                import numpy as np

                raw = base64.b64decode(b64)
                arr = np.frombuffer(raw, np.uint8)
                frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame_bgr is not None:
                    camera_store.update_frame(CAM_WS_SOURCE, frame_bgr)
            except Exception as exc:
                logger.warning("source frame decode error: %s", exc)
    except WebSocketDisconnect:
        pass
    finally:
        camera_store.unregister(CAM_WS_SOURCE)
        cam_ids = camera_store.camera_ids()
        await manager.broadcast(
            json.dumps({"type": "camera_list", "cameras": cam_ids})
        )
        logger.info("Mobile source disconnected. %s unregistered.", CAM_WS_SOURCE)


# ── 편의 함수: 외부(카메라 루프 등)에서 카메라 등록/해제 ─────────────────────────
def register_camera(cam_id: str) -> None:
    camera_store.register(cam_id)


def unregister_camera(cam_id: str) -> None:
    camera_store.unregister(cam_id)


def push_frame(cam_id: str, frame_bgr: Any) -> None:
    """추론 루프 스레드에서 새 프레임을 저장합니다."""
    camera_store.update_frame(cam_id, frame_bgr)


def push_inference_state(cam_id: str, **kwargs: Any) -> None:
    """추론 결과를 상태 저장소에 반영합니다."""
    camera_store.update_inference(cam_id, **kwargs)


__all__ = [
    "router",
    "manager",
    "camera_store",
    "_broadcast_loop",
    "register_camera",
    "unregister_camera",
    "push_frame",
    "push_inference_state",
    "CAM_WS_SOURCE",
]
