"""Calibration API endpoints."""

import asyncio
import platform
import subprocess
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend.models.system import CalibrationStatus
from backend.services.auto_calibration import AutoCalibrationService
from backend.services.calibration_service import CalibrationService
from backend.services.config_manager import ConfigManager
from backend.services.manual_calibration import ManualCalibrationService, MOTOR_IDS
from backend.services.process_manager import process_manager

router = APIRouter()
calibration_service = CalibrationService()
config_manager = ConfigManager()
auto_cal_service = AutoCalibrationService()
manual_cal_service = ManualCalibrationService()


class CalibrationStartRequest(BaseModel):
    """Request to start calibration."""

    device_type: str  # "robot" or "teleoperator"
    device_id: str
    robot_type: str
    port: str


class CalibrationStartResponse(BaseModel):
    """Response from starting calibration."""

    process_id: str
    message: str


@router.get("/files", response_model=List[str])
async def list_calibration_files(category: str, robot_type: str):
    """List available calibration files for a given category and robot type.

    Args:
        category: "robots" or "teleoperators".
        robot_type: Robot type (e.g., "so101_follower", "bi_so101_follower").
    """
    try:
        return calibration_service.list_calibration_files(category, robot_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list calibration files: {e}")


@router.post("/open-folder")
async def open_calibration_folder(category: str, robot_type: str):
    """Open the calibration folder in the system file manager.

    Args:
        category: "robots" or "teleoperators".
        robot_type: Robot type (e.g., "so101_follower", "so101_leader").
    """
    folder = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / category / robot_type
    folder.mkdir(parents=True, exist_ok=True)

    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", str(folder)])
        elif system == "Windows":
            subprocess.Popen(["explorer", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open folder: {e}")

    return {"message": f"Opened {folder}", "path": str(folder)}


@router.get("/missing", response_model=List[CalibrationStatus])
async def get_missing_calibrations():
    """Get list of devices missing calibration based on current config."""
    try:
        config = config_manager.load_config()
        return calibration_service.list_missing_calibrations(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check calibrations: {e}")


@router.get("/status", response_model=List[CalibrationStatus])
async def get_calibration_status():
    """Get calibration status for all devices in current config."""
    try:
        config = config_manager.load_config()
        statuses = []

        if config.mode == "bimanual":
            # Bimanual sub-arm IDs are derived as {base}_left / {base}_right.
            bi = config.bimanual
            follower_base = bi.follower_id or "bimanual_follower"
            leader_base = bi.leader_id or "bimanual_leader"
            devices = [
                ("robot", f"{follower_base}_left", "bi_so101_follower", bi.left_follower_port),
                ("robot", f"{follower_base}_right", "bi_so101_follower", bi.right_follower_port),
                ("teleoperator", f"{leader_base}_left", "bi_so101_leader", bi.left_leader_port),
                ("teleoperator", f"{leader_base}_right", "bi_so101_leader", bi.right_leader_port),
            ]
        else:
            # Check single arm devices
            devices = [
                ("robot", "single_follower", "so101_follower", config.single_arm.follower_port),
                ("teleoperator", "single_leader", "so101_leader", config.single_arm.leader_port),
            ]

        for device_type, device_id, robot_type, port in devices:
            if port or config.mode == "bimanual":
                status = calibration_service.check_calibration(device_type, device_id, robot_type, port)
                statuses.append(status)

        return statuses

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get calibration status: {e}")


@router.post("/start", response_model=CalibrationStartResponse)
async def start_calibration(request: CalibrationStartRequest):
    """Start calibration process for a device."""
    try:
        command = calibration_service.build_calibration_command(
            request.device_type, request.device_id, request.robot_type, request.port
        )

        process_id = await process_manager.start_process(command, "calibration")

        return CalibrationStartResponse(
            process_id=process_id, message=f"Calibration started for {request.device_id}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start calibration: {e}")


@router.post("/stop/{process_id}")
async def stop_calibration(process_id: str):
    """Stop calibration process."""
    try:
        success = await process_manager.stop_process(process_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Process {process_id} not found")

        return {"message": "Calibration stopped successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop calibration: {e}")


# --- Auto-calibration endpoints ---
# Runs lerobot_measure_feetech_ranges.py as a subprocess via ProcessManager.
# Logs are streamed via the existing /ws/logs/{process_id} WebSocket.


class AutoCalibrationStartRequest(BaseModel):
    """Request to start auto-calibration."""

    port: str
    device_id: str = "left_follower"


class AutoCalibrationStartResponse(BaseModel):
    """Response from starting auto-calibration."""

    process_id: str
    message: str


@router.post("/auto/start", response_model=AutoCalibrationStartResponse)
async def start_auto_calibration(request: AutoCalibrationStartRequest):
    """Start auto-calibration for an entire arm.

    Runs lerobot_measure_feetech_ranges with --save to calibrate all 6 motors
    and persist the result. Logs can be streamed via /ws/logs/{process_id}.
    """
    try:
        process_id = await auto_cal_service.start(
            port=request.port,
            device_id=request.device_id,
        )
        return AutoCalibrationStartResponse(
            process_id=process_id,
            message=f"Auto-calibration started for {request.device_id}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start auto-calibration: {e}")


@router.post("/auto/stop/{process_id}")
async def stop_auto_calibration(process_id: str):
    """Stop a running auto-calibration process."""
    success = await auto_cal_service.stop(process_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Process {process_id} not found")
    return {"message": "Auto-calibration stopped"}


class AutoCalibrationCompleteRequest(BaseModel):
    """Request to complete auto-calibration."""

    category: str = "robots"
    robot_type: str = "so101_follower"


@router.post("/auto/complete/{device_id}")
async def complete_auto_calibration(device_id: str, request: AutoCalibrationCompleteRequest):
    """Post-process after auto-calibration completes.

    Copies the calibration file from so_follower/ to the correct category/type
    directory so the UI and lerobot SO101 wrappers can find it.
    """
    dst = auto_cal_service.copy_to_so101_path(
        device_id,
        category=request.category,
        robot_type=request.robot_type,
    )
    if dst is None:
        raise HTTPException(
            status_code=404,
            detail=f"Calibration file for '{device_id}' not found in so_follower/",
        )
    return {"message": "Calibration file copied", "path": str(dst)}


# --- Manual calibration endpoint ---


@router.websocket("/manual/ws")
async def manual_calibration_ws(websocket: WebSocket):
    """WebSocket endpoint for step-by-step manual calibration.

    Protocol:
    1. Client sends: {"action": "start", "port": "...", "device_type": "...", "robot_type": "...", "device_id": "..."}
       Server responds: {"type": "connected", "motors": [...]}
    2. Client sends: {"action": "set_homing"}
       Server responds: {"type": "homing_done", "offsets": {...}}
    3. Client sends: {"action": "start_recording"}
       Server streams: {"type": "positions", "motors": {"shoulder_pan": {"pos": ..., "min": ..., "max": ...}, ...}}
    4. Client sends: {"action": "stop_recording"}
       Server responds: {"type": "recording_done", "mins": {...}, "maxes": {...}}
       Then saves calibration and responds: {"type": "saved", "path": "..."}
    """
    await websocket.accept()

    bus = None
    port = None
    device_type = None
    robot_type = None
    device_id = None
    homing_offsets = None
    recording = False
    mins = {}
    maxes = {}

    try:
        while True:
            # If recording, check for messages with a short timeout then send positions
            if recording and bus:
                try:
                    msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                except asyncio.TimeoutError:
                    # No message — send position update
                    positions = await asyncio.to_thread(manual_cal_service.read_positions, bus)
                    for motor, pos in positions.items():
                        mins[motor] = min(pos, mins.get(motor, pos))
                        maxes[motor] = max(pos, maxes.get(motor, pos))
                    await websocket.send_json({
                        "type": "positions",
                        "motors": {
                            motor: {"pos": positions[motor], "min": mins[motor], "max": maxes[motor]}
                            for motor in positions
                        },
                    })
                    continue
                except WebSocketDisconnect:
                    break
            else:
                try:
                    msg = await websocket.receive_json()
                except WebSocketDisconnect:
                    break

            action = msg.get("action")

            if action == "start":
                port = msg["port"]
                device_type = msg.get("device_type", "robot")
                robot_type = msg.get("robot_type", "so101_follower")
                device_id = msg.get("device_id", "left_follower")
                bus = await asyncio.to_thread(manual_cal_service.create_bus, port)
                await websocket.send_json({
                    "type": "connected",
                    "motors": list(bus.motors.keys()),
                })

            elif action == "set_homing" and bus:
                homing_offsets = await asyncio.to_thread(manual_cal_service.set_homing, bus)
                await websocket.send_json({
                    "type": "homing_done",
                    "offsets": homing_offsets,
                })

            elif action == "start_recording" and bus:
                # Reset mins/maxes from current positions
                positions = await asyncio.to_thread(manual_cal_service.read_positions, bus)
                mins = dict(positions)
                maxes = dict(positions)
                recording = True
                await websocket.send_json({"type": "recording_started"})

            elif action == "stop_recording":
                recording = False
                await websocket.send_json({
                    "type": "recording_done",
                    "mins": mins,
                    "maxes": maxes,
                })

                # Build calibration data and save
                if homing_offsets and device_type and robot_type and device_id:
                    calibration_data = {}
                    for motor in mins:
                        calibration_data[motor] = {
                            "id": MOTOR_IDS[motor],
                            "drive_mode": 0,
                            "homing_offset": homing_offsets[motor],
                            "range_min": mins[motor],
                            "range_max": maxes[motor],
                        }

                    saved_path = manual_cal_service.save_calibration(
                        calibration_data, device_type, robot_type, device_id,
                    )

                    # Write to motors using a fresh bus connection
                    if port:
                        await asyncio.to_thread(
                            manual_cal_service.write_calibration_to_motors,
                            port, calibration_data,
                        )

                    await websocket.send_json({
                        "type": "saved",
                        "path": str(saved_path),
                    })

            elif action == "disconnect":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if bus:
            try:
                await asyncio.to_thread(bus.disable_torque)
                await asyncio.to_thread(bus.disconnect)
            except Exception:
                pass
        try:
            await websocket.close()
        except Exception:
            pass
