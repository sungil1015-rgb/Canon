"""FastAPI application entry point for the Canon inspection backend.

실행 방법 (backend/ 디렉터리에서):
    uv run uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

또는 직접:
    uv run python -m app.main
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# backend/ 루트를 sys.path 에 추가하여 db.database 를 import 가능하게 합니다.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.api_router import router as api_router
from app.api.ws_router import _broadcast_loop, router as ws_router
from app.core.paths import ensure_project_dirs

# ── 로깅 ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── FastAPI 앱 ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Canon Inspection Backend",
    description="산업 현장 화면 인식 백엔드 API",
    version="0.1.0",
)

# ── CORS 설정 ─────────────────────────────────────────────────────────────────
# 프론트엔드 Vite dev 서버(기본 5173) 및 배포 주소를 허용합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # 배포 시 실제 도메인 추가
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 라우터 등록 ───────────────────────────────────────────────────────────────
app.include_router(api_router)   # REST  /api/*
app.include_router(ws_router)    # WS    /ws, /ws/source


# ── 라이프사이클 이벤트 ───────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup() -> None:
    """서버 시작 시 실행: 디렉터리 생성, DB 초기화, 브로드캐스트 루프 시작."""
    logger.info("Canon backend starting up…")

    # 필수 디렉터리 보장
    ensure_project_dirs()

    # inspection_log DB 초기화 (테이블 없으면 생성)
    from db.database import DB_PATH, initialize as init_db
    init_db(DB_PATH)
    logger.info("Inspection DB initialized: %s", DB_PATH)

    # WebSocket 브로드캐스트 백그라운드 태스크 시작
    asyncio.create_task(_broadcast_loop())
    logger.info("WebSocket broadcast loop started.")

    logger.info("Canon backend ready.  Listening on http://0.0.0.0:8080")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("Canon backend shutting down.")


# ── 헬스체크 ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ── 직접 실행 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )
