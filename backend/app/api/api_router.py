"""REST API router for the Canon inspection dashboard.

Endpoints expected by the frontend (frontend/src/services/api.ts):
    GET  /api/inspection-logs          → inspection log list (paginated)
    POST /api/override                 → override camera state
    POST /api/inspect-image            → upload image(s)/video(s) for on-demand inference
    POST /api/reinspect-log/{log_id}   → re-run inference on a stored log entry

inspect-image 처리 흐름:
    이미지 파일 → YOLO detection + warping + 전체 target(1~4) 분류
    영상 파일   → SequenceService.process_video() 순차 판정 (T1→T2→T3→T4)
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from db.database import DB_PATH, get_log_by_id, get_logs, get_sequence_runs as fetch_sequence_runs, initialize, insert_log, update_log

router = APIRouter(prefix="/api")

# 라우터 로드 시 DB 초기화
initialize(DB_PATH)

# 지원 확장자 분류
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".ts"}

# 기본 target 순서
DEFAULT_TARGET_ORDER = ["target_1", "target_2", "target_3", "target_4"]


# ── Pydantic 스키마 ────────────────────────────────────────────────────────────
class OverrideRequest(BaseModel):
    cam_id: str
    predicted_label: str | None = None
    confidence: float | None = None
    is_unknown: bool | None = None
    logic: dict[str, Any] | None = None
    display: dict[str, Any] | None = None


# ── 내부: 이미지 단일 추론 ────────────────────────────────────────────────────
def _inspect_single_image(image_path: Path) -> list[dict[str, Any]]:
    """YOLO detection → warping → target 1~4 분류. 결과 리스트를 반환합니다."""
    import cv2
    from app.core.paths import yolo_weight_file
    from app.models.warping import YoloScreenWarper
    from app.service.target_service import TargetService

    frame_bgr = cv2.imread(str(image_path))
    if frame_bgr is None:
        return []

    results: list[dict[str, Any]] = []

    try:
        yolo_path = yolo_weight_file()
        if not yolo_path.exists():
            # YOLO 가중치 없으면 raw 분류만
            raise FileNotFoundError("no yolo weights")

        warper = YoloScreenWarper(
            weights=yolo_path,
            device="cpu",
            conf=0.25,
            imgsz=640,
            padding_ratio=0.02,
            output_size=640,
            classes=[0],
        )
        detections = warper.detect(frame_bgr)

        if not detections:
            return [{
                "target_name": "unknown",
                "target_idx": 0,
                "label": "no_detection",
                "score": 0.0,
                "confirmed_state": "NoDetection",
                "anomaly_flag": True,
            }]

        best = max(detections, key=lambda d: d["confidence"])
        warped = warper.warp_detection(frame_bgr, best, index=0)
        warped_bgr = warped.warped_bgr

        svc = TargetService()
        for idx, target_name in enumerate(DEFAULT_TARGET_ORDER, start=1):
            try:
                pred = svc.predict_bgr(target_name, warped_bgr)
                results.append({
                    "target_name": target_name,
                    "target_idx": idx,
                    "label": pred.label,
                    "score": float(pred.score),
                    "confirmed_state": "Yes" if pred.label == "yes" else "No",
                    "anomaly_flag": pred.label != "yes",
                })
            except Exception as e:
                results.append({
                    "target_name": target_name,
                    "target_idx": idx,
                    "label": "error",
                    "score": 0.0,
                    "confirmed_state": "Error",
                    "anomaly_flag": True,
                })

    except FileNotFoundError:
        # YOLO 없으면 raw 이미지로 target_1만 분류 시도
        try:
            from app.service.target_service import TargetService
            svc = TargetService()
            pred = svc.predict_bgr("target_1", frame_bgr)
            results.append({
                "target_name": "target_1",
                "target_idx": 1,
                "label": pred.label,
                "score": float(pred.score),
                "confirmed_state": "Yes" if pred.label == "yes" else "No",
                "anomaly_flag": pred.label != "yes",
            })
        except Exception:
            pass

    return results


# ── 내부: 영상 순차 추론 ───────────────────────────────────────────────────────
def _inspect_video(video_path: Path, run_root: Path) -> dict[str, Any]:
    """SequenceService를 사용해 영상에서 T1→T4 순차 판정합니다."""
    from app.core.paths import OUTPUTS_DIR, yolo_weight_file
    from app.service.sequence_service import SequenceRunConfig, SequenceService

    yolo_path = yolo_weight_file()
    config = SequenceRunConfig(
        source=[video_path],
        target_order=DEFAULT_TARGET_ORDER,
        yolo_weights=yolo_path,
        output_root=run_root,
        device="cpu",
        save_confirmed_frames=False,
    )
    svc = SequenceService(config)
    result = svc.process_video(video_path, run_root)
    return result


# ── GET /api/inspection-logs ───────────────────────────────────────────────────
@router.get("/inspection-logs")
def get_inspection_logs(offset: int = 0, limit: int = 30) -> JSONResponse:
    """검사 이력 목록을 최신순으로 반환합니다."""
    logs = get_logs(offset=offset, limit=limit)
    return JSONResponse(logs)


# ── GET /api/sequence-runs ─────────────────────────────────────────────────────
@router.get("/sequence-runs")
def get_sequence_runs(offset: int = 0, limit: int = 50) -> JSONResponse:
    """sequence_runs.sqlite3 의 결과를 inspection_log 포맷으로 반환합니다.

    프론트엔드 TestMode 테이블과 동일한 필드 구조를 사용합니다:
        id, timestamp, source_type, confirmed_state, predicted_label,
        confidence, anomaly_flag, file_path, cam_id, target_idx
    """
    return JSONResponse(fetch_sequence_runs(offset=offset, limit=limit, total_targets=len(DEFAULT_TARGET_ORDER)))


# ── POST /api/override ─────────────────────────────────────────────────────────
@router.post("/override")
def override_camera(body: OverrideRequest) -> JSONResponse:
    """카메라 상태를 수동으로 덮어씁니다. 변경 이력을 DB에 기록합니다."""
    log_id = insert_log(
        source_type="override",
        confirmed_state=str((body.logic or {}).get("confirmed_state", "Override")),
        predicted_label=body.predicted_label or "",
        confidence=float(body.confidence or 0.0),
        anomaly_flag=bool(body.is_unknown),
        cam_id=body.cam_id,
        target_idx=int((body.logic or {}).get("current_step_index", 0)),
        extra={"logic": body.logic, "display": body.display},
    )
    return JSONResponse({"status": "ok", "cam_id": body.cam_id, "log_id": log_id})


# ── POST /api/inspect-image ────────────────────────────────────────────────────
@router.post("/inspect-image")
async def inspect_image(files: list[UploadFile] = File(...)) -> JSONResponse:
    """이미지 또는 영상 파일을 업로드하여 추론 결과를 반환합니다.

    이미지: YOLO + warping + target 1~4 분류
    영상:   SequenceService 순차 판정 (T1→T4)
    """
    results = []
    loop = asyncio.get_running_loop()

    # 임시 디렉터리에 업로드 파일 저장
    with tempfile.TemporaryDirectory(prefix="canon_inspect_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        for upload in files:
            filename = upload.filename or "upload"
            save_path = tmp_path / filename
            content = await upload.read()
            save_path.write_bytes(content)

            ext = save_path.suffix.lower()
            file_label = filename

            if ext in _VIDEO_EXTS:
                # ── 영상 처리 ──────────────────────────────────────────────
                from app.core.paths import SEQUENCE_VIDEO_RUNS_DIR

                try:
                    video_result = await loop.run_in_executor(
                        None, _inspect_video, save_path, SEQUENCE_VIDEO_RUNS_DIR
                    )
                    targets = video_result.get("targets", [])
                    completed = bool(video_result.get("completed", False))
                    confirmed_targets = int(video_result.get("confirmed_targets", 0))

                    confirmed_state = (
                        f"Complete_{confirmed_targets}of{len(DEFAULT_TARGET_ORDER)}"
                        if completed else
                        f"Partial_{confirmed_targets}of{len(DEFAULT_TARGET_ORDER)}"
                    )
                    last_target = targets[-1] if targets else {}
                    predicted_label = last_target.get("target_name", "unknown")
                    confidence = float(last_target.get("last_score", 0.0))
                    anomaly_flag = not completed

                    log_id = insert_log(
                        source_type="video_upload",
                        confirmed_state=confirmed_state,
                        predicted_label=predicted_label,
                        confidence=confidence,
                        anomaly_flag=anomaly_flag,
                        file_path=file_label,
                        target_idx=confirmed_targets,
                        extra={"video_result": {
                            "completed": completed,
                            "confirmed_targets": confirmed_targets,
                            "targets": [
                                {"name": t.get("target_name"), "completed": t.get("completed")}
                                for t in targets
                            ],
                        }},
                    )
                    results.append({
                        "id": log_id,
                        "file": file_label,
                        "source_type": "video_upload",
                        "predicted_label": predicted_label,
                        "confidence": confidence,
                        "confirmed_state": confirmed_state,
                        "anomaly_flag": anomaly_flag,
                        "targets": [
                            {
                                "target_name": t.get("target_name"),
                                "completed": t.get("completed"),
                                "last_label": t.get("last_label"),
                                "last_score": t.get("last_score"),
                            }
                            for t in targets
                        ],
                    })
                except Exception as e:
                    results.append({
                        "file": file_label,
                        "source_type": "video_upload",
                        "error": str(e),
                        "confirmed_state": "Error",
                        "anomaly_flag": True,
                    })

            elif ext in _IMAGE_EXTS:
                # ── 이미지 처리 ────────────────────────────────────────────
                try:
                    per_target = await loop.run_in_executor(
                        None, _inspect_single_image, save_path
                    )

                    for item in per_target:
                        log_id = insert_log(
                            source_type="image_upload",
                            confirmed_state=item["confirmed_state"],
                            predicted_label=item["label"],
                            confidence=item["score"],
                            anomaly_flag=item["anomaly_flag"],
                            file_path=file_label,
                            target_idx=item["target_idx"],
                            extra={"target_name": item["target_name"]},
                        )
                        results.append({
                            "id": log_id,
                            "file": file_label,
                            "source_type": "image_upload",
                            "target_name": item["target_name"],
                            "target_idx": item["target_idx"],
                            "predicted_label": item["label"],
                            "confidence": item["score"],
                            "confirmed_state": item["confirmed_state"],
                            "anomaly_flag": item["anomaly_flag"],
                        })
                except Exception as e:
                    results.append({
                        "file": file_label,
                        "source_type": "image_upload",
                        "error": str(e),
                        "confirmed_state": "Error",
                        "anomaly_flag": True,
                    })
            else:
                results.append({
                    "file": file_label,
                    "error": f"지원하지 않는 파일 형식: {ext}",
                    "confirmed_state": "Unsupported",
                    "anomaly_flag": True,
                })

    return JSONResponse({"status": "ok", "inspections": results})


# ── POST /api/reinspect-log/{log_id} ──────────────────────────────────────────
@router.post("/reinspect-log/{log_id}")
async def reinspect_log(log_id: int) -> JSONResponse:
    """저장된 로그 항목의 파일 경로로 재추론합니다.

    file_path 가 실제 존재하는 파일이면 다시 추론하고, 없으면 DB 값을 그대로 반환합니다.
    """
    loop = asyncio.get_running_loop()
    log = get_log_by_id(log_id)
    if log is None:
        raise HTTPException(status_code=404, detail="log not found")

    file_path = log.get("file_path", "")
    predicted_label = log.get("predicted_label", "unknown")
    confidence = float(log.get("confidence") or 0.0)
    confirmed_state = log.get("confirmed_state", "Pending")

    src = Path(file_path) if file_path else None

    if src and src.exists():
        ext = src.suffix.lower()
        try:
            if ext in _IMAGE_EXTS:
                per_target = await loop.run_in_executor(None, _inspect_single_image, src)
                if per_target:
                    # target_1 기준으로 대표 값 설정
                    item = per_target[0]
                    predicted_label = item["label"]
                    confidence = item["score"]
                    confirmed_state = item["confirmed_state"]
            elif ext in _VIDEO_EXTS:
                from app.core.paths import SEQUENCE_VIDEO_RUNS_DIR
                result = await loop.run_in_executor(
                    None, _inspect_video, src, SEQUENCE_VIDEO_RUNS_DIR
                )
                targets = result.get("targets", [])
                confirmed_targets = int(result.get("confirmed_targets", 0))
                completed = bool(result.get("completed", False))
                confirmed_state = (
                    f"Complete_{confirmed_targets}of{len(DEFAULT_TARGET_ORDER)}"
                    if completed else
                    f"Partial_{confirmed_targets}of{len(DEFAULT_TARGET_ORDER)}"
                )
                last = targets[-1] if targets else {}
                predicted_label = last.get("target_name", "unknown")
                confidence = float(last.get("last_score", 0.0))
        except Exception as e:
            confirmed_state = f"Error: {e}"

    update_log(
        log_id,
        confirmed_state=confirmed_state,
        predicted_label=predicted_label,
        confidence=confidence,
    )

    return JSONResponse({
        "status": "success",
        "id": log_id,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "confirmed_state": confirmed_state,
    })
