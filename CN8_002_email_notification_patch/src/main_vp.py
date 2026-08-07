#!/usr/bin/env python3
import argparse
import os
import time
import json
from enum import Enum
import sys
import subprocess
import fcntl
import threading
import traceback
import faulthandler
from dataclasses import dataclass

import cv2
import numpy as np

from capture.camera_factory import create_camera_from_hardware_config
from hardware.hardware_config_loader import load_hardware_config
from roi.roi_manager import ROIManager
from roi.roi_editor import ROIEditor
from inspection.inspector import Inspector
from inspection.stabilizer import Stabilizer
from inspection.normalize import normalize_frame

from plc.plc_config_loader import load_plc_config
from plc.plc_controller import create_plc_controller
from plc.plc_error_logger import (
    configure_diagnostics,
    get_diagnostics_config,
    save_normal_reference_log,
    save_plc_error_log,
    save_plc_test_log,
)

from light.light_controller import create_light_controller_from_hardware_config

from ui import overlay_clean as overlay
from typing import Optional, Dict, Any
from ui.hud import draw_mode_indicator, draw_dev_hud
from ui.pose_guide import draw_pose_message
from ui.service_panel import ServicePanel
from runtime.product_profile_loader import load_product_profile
from data_io.sample_capture import (
    handle_sample_keys,
    prune_snapshots,
    save_inspection_capture,
)
from ui.control_bar import render_control_bar, key_to_cmd, button_id_to_cmd
from app.command_executor import execute_command
from inspection.inspect_service import run_inspect_once
from email_notifier import EmailNotifier
from modes.run_renderer import draw_run_tracking
from runtime.runtime_config_loader import load_runtime_config
from runtime.service_soak_test import ServiceSoakTest
from app.app_setup import ensure_dirs
from app.app_paths import (
    PLC_CONFIG_PATH,
    HARDWARE_CONFIG_PATH,
    PRODUCT_PROFILE_PATH,
    PROFILES_DIR,
    TEMPLATE_PATH,
    DATA_DIR,
    ROI_DIR,
    ROI_PATH,
    RECIPES_DIR,
    DEFAULT_RECIPE_PATH,
    RUNTIME_CONFIG_PATH,
    LOGS_ROOT,
    PROJECT_ROOT,
)
from inspection.auto_baseline import AutoBaseline

# =========================
# Commands
# =========================
class CameraRecoveryInProgress(RuntimeError):
    pass


class UICmd(Enum):
    NONE = 0
    TOGGLE_MODE = 1
    INSPECT = 2
    AUTOTUNE = 3
    RELOAD = 4
    SAVE = 5
    NEXT = 6
    CLEAR = 7
    QUIT = 8
    DELETE = 9
    TOGGLE_AUTO_INSPECT = 10  # NEW
    TOGGLE_SERVICE_PANEL = 11

# =========================
# Config
# =========================
DEV_MODE = True

POSE_ROI_ID_STR = "1"          # pose 판단 ROI (문자열 키)
POSE_METRIC_KEY = "blob_count" # pose 판단 metric
POSE_EXPECT = 4                # blob_count == 4

# =========================
# Small utils
# =========================

def roi_label_pos(x, y, w, h, margin=25):
    tx = x
    ty = y - margin
    if ty < 18:
        ty = y + h + 18
    return int(tx), int(ty)


def _resolve_startup_mode(requested_mode: str = "auto") -> str:
    """Resolve boot/manual startup without changing the inspection logic.

    - Explicit --startup-mode or VISION_STARTUP_MODE always wins.
    - Terminal launch defaults to EDIT.
    - Desktop/system autostart (no TTY) defaults to RUN.
    """
    requested = str(requested_mode or "auto").strip().lower()
    env_mode = str(os.environ.get("VISION_STARTUP_MODE", "") or "").strip().lower()

    if requested in ("edit", "run"):
        return requested
    if env_mode in ("edit", "run"):
        return env_mode

    try:
        interactive = bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:
        interactive = False
    return "edit" if interactive else "run"


_INSTANCE_LOCK_FD = None


def _acquire_single_instance_lock(
    lock_path: str = "/tmp/daol_vision_main.lock",
):
    """Prevent two camera clients from opening the same Argus sensor."""
    global _INSTANCE_LOCK_FD

    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            owner = os.read(fd, 128).decode("utf-8", errors="replace").strip()
        except Exception:
            owner = ""
        os.close(fd)
        raise RuntimeError(
            "vision application is already running"
            + (f" ({owner})" if owner else "")
        )

    os.ftruncate(fd, 0)
    os.write(
        fd,
        f"pid={os.getpid()} started={time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode(
            "utf-8"
        ),
    )
    os.fsync(fd)
    _INSTANCE_LOCK_FD = fd
    return fd


_RUNTIME_FAULT_LOG_HANDLE = None


def _enable_runtime_fault_log():
    """Persist Python fatal traces even when the launch terminal is unavailable."""
    global _RUNTIME_FAULT_LOG_HANDLE
    try:
        log_dir = os.path.join(LOGS_ROOT, "terminal")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, "faulthandler.log")
        if os.path.exists(path) and os.path.getsize(path) > (5 * 1024 * 1024):
            backup = path + ".1"
            try:
                os.remove(backup)
            except OSError:
                pass
            os.replace(path, backup)
        _RUNTIME_FAULT_LOG_HANDLE = open(path, "a", encoding="utf-8", buffering=1)
        _RUNTIME_FAULT_LOG_HANDLE.write(
            f"\n===== START {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"pid={os.getpid()} =====\n"
        )
        faulthandler.enable(file=_RUNTIME_FAULT_LOG_HANDLE, all_threads=True)
        return path
    except Exception as e:
        print(f"[RUNTIME LOG] faulthandler setup failed: {e}")
        return ""


def _write_runtime_marker(message: str):
    try:
        if _RUNTIME_FAULT_LOG_HANDLE is not None:
            _RUNTIME_FAULT_LOG_HANDLE.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
            )
            _RUNTIME_FAULT_LOG_HANDLE.flush()
    except Exception:
        pass


class LightCommunicationError(RuntimeError):
    pass
class InspectionResultError(RuntimeError):
    pass
@dataclass
class AppState:
    edit_mode: bool = True
    status: str = "EDIT MODE"
    quit_requested: bool = False
    last_results: Optional[Dict[str, Any]] = None   # {"1": Result, ...}
    last_overall_ok: Optional[bool] = None

    pending_cmd: UICmd = UICmd.NONE

    # pose assist
    pose_bad_cnt: int = 0

    # auto inspect
    auto_inspect: bool = False
    last_auto_inspect_ts: float = 0.0

    # UI
    last_buttons: list = None

    # spot light pre-arm
    spot_armed: bool = False
    spot_armed_ts: float = 0.0
    spot_armed_brightness: int = 0

    tracking_stable: bool = False
    stable_frame_count: int = 0
    run_mode_text: str = "HELD"

    #학습을 위한 저장
    baseline_learning: bool = False
    baseline_target_count: int = 10
    baseline_count: int = 0

    # PLC shutdown
    plc_shutdown_started: bool = False

    # PLC ready
    plc_vision_ready: bool = False
    plc_ready_consumed: bool = False

    # Vision error latch
    camera_error_latched: bool = False
    camera_read_blocked: bool = False
    light_error_latched: bool = False

    camera_error_grace_until: float = 0.0

    # PLC service-panel test state
    test_error_active: bool = False
    test_error_mode: str = ""          # logic / recovery
    test_error_type: str = ""
    test_error_code: int = 0
    test_error_expected_code: int = 0
    test_error_actual_code: int = 0
    test_error_injected_at: str = ""
    test_error_request_id: str = ""
    test_error_safe_logical: bool = True
    test_error_phase: str = "IDLE"
    test_error_result: str = ""
    test_error_health_detail: str = ""
    test_error_block_reason: str = ""
    test_error_log_path: str = ""
    test_error_log_saved: bool = False
    test_error_recovered_at: str = ""
    latest_error_log_path: str = ""

    # Test-only lifecycle context. Production PLC/camera/light logic is not
    # modified by these fields.
    test_error_pre_command: int = 0
    test_error_pre_status: int = 0
    test_error_pre_result: int = 0
    test_error_pre_error_code: int = 0
    test_error_reset_source: str = ""
    test_error_started_monotonic: float = 0.0
    test_error_recovery_elapsed_ms: int = 0


class VisionApp:
    def __init__(self, startup_mode: str = "edit"):
        ensure_dirs(DATA_DIR, ROI_DIR, LOGS_ROOT)

        self.runtime_cfg = load_runtime_config(RUNTIME_CONFIG_PATH)
        self.hardware_cfg = load_hardware_config(HARDWARE_CONFIG_PATH)
        self.plc_cfg = load_plc_config(PLC_CONFIG_PATH)
        self.startup_mode = _resolve_startup_mode(startup_mode)
        configure_diagnostics(
            self.plc_cfg.get("diagnostics", {}) or {},
            LOGS_ROOT,
        )

        profile_name = str(self.runtime_cfg.get("profile_name", "") or "").strip()
        profile_path = PRODUCT_PROFILE_PATH

        if profile_name:
            candidate = os.path.join(PROFILES_DIR, f"{profile_name}_profile.json")
            if os.path.exists(candidate):
                profile_path = candidate

        self.product_profile = load_product_profile(profile_path)
        print("[PROFILE]", profile_path)

        self.cam, self.camera_info = create_camera_from_hardware_config(self.hardware_cfg)

        self.frame_width = int(self.camera_info.get("width", 1280))
        self.frame_height = int(self.camera_info.get("height", 720))

        camera_profile = self.product_profile.get(
            "camera_profile",
            self.camera_info.get("camera_profile", "default"),
        )

        if hasattr(self.cam, "set_profile"):
            self.cam.set_profile(camera_profile)

        print("[MAIN] camera created", self.camera_info.get("name", "camera"))
        print("[MAIN] camera device", self.camera_info.get("device", ""))
        print("[MAIN] camera size", self.frame_width, self.frame_height)

        self.runtime_cfg["_product_profile"] = self.product_profile
        self.runtime_cfg["_hardware_config"] = self.hardware_cfg
        self.runtime_cfg["_camera_info"] = self.camera_info

        self.plc = create_plc_controller(self.plc_cfg)
        self.runtime_cfg["_plc_config"] = self.plc_cfg

        self.light = create_light_controller_from_hardware_config(self.hardware_cfg)
        self.runtime_cfg["_light_state"] = self.light.get_state()
        
        recipe_name = self.product_profile.get("recipe_name", "tape_presence")
        recipe_candidate = os.path.join(RECIPES_DIR, f"{recipe_name}.json")
        selected_recipe_path = recipe_candidate if os.path.exists(recipe_candidate) else DEFAULT_RECIPE_PATH
        self.roi_mgr = ROIManager(frame_size=(self.frame_width, self.frame_height))

        profile_name = str(self.runtime_cfg.get("profile_name", "") or "").strip()

        roi_path = ROI_PATH  # fallback

        if profile_name:
            candidate = os.path.join(PROFILES_DIR, f"{profile_name}_roi.json")
            if os.path.exists(candidate):
                roi_path = candidate

        print("[ROI PATH]", roi_path)

        try:
            self.roi_mgr.load(roi_path)
            if DEV_MODE == True:
                print("[DBG ROI COUNT]", len(getattr(self.roi_mgr, "rois", [])))
        except Exception as e:
            pass

        self.editor = ROIEditor(self.roi_mgr)
        self.inspector = Inspector(
            self.roi_mgr,
            recipe_path=selected_recipe_path,
            logs_root=LOGS_ROOT,
            runtime_cfg=self.runtime_cfg,
        )

        # E-mail notification is isolated from inspection/PLC/light logic.
        # Communication failures only create notifier logs and never stop vision.
        self.email_notifier = None
        try:
            active_light_set = str(
                self.hardware_cfg.get("active_light_set", "") or ""
            )
            light_sets = self.hardware_cfg.get("light_sets", {}) or {}
            active_light_cfg = light_sets.get(active_light_set, {}) or {}

            self.email_notifier = EmailNotifier(
                config_path=os.path.join(
                    DATA_DIR,
                    "config",
                    "email_config.json",
                ),
                logs_root=LOGS_ROOT,
                static_context={
                    "profile_name": profile_name,
                    "recipe_name": recipe_name,
                    "recipe_path": selected_recipe_path,
                    "camera_name": self.camera_info.get("name", ""),
                    "camera_size": f"{self.frame_width}x{self.frame_height}",
                    "plc_enabled": bool(self.plc_cfg.get("enabled", False)),
                    "plc_backend": self.plc_cfg.get("backend", ""),
                    "light_set": active_light_set,
                    "light_backend": active_light_cfg.get("backend", "disabled"),
                },
            )
            self.inspector.email_notifier = self.email_notifier
        except Exception as e:
            print("[EMAIL] notifier setup failed:", e)

        self.editor.on_select_changed = self.inspector.reset_tracker_template

        stab_cfg = self.runtime_cfg.get("stabilizer", {})

        self.stabilizer = Stabilizer(
            window=int(stab_cfg.get("window", 5)),
            move_thresh_px=float(stab_cfg.get("move_thresh_px", 3)),
            alpha=float(stab_cfg.get("alpha", 0.7)),
        )

        self.state = AppState(last_buttons=[])
        self.state.auto_inspect = bool(self.runtime_cfg.get("enable_auto_inspect", True))
        self.state.edit_mode = self.startup_mode != "run"
        run_mode = str(self.runtime_cfg.get("run_mode", "held") or "held").lower()
        if self.state.edit_mode:
            self.state.status = "EDIT MODE"
        else:
            self.state.status = (
                "RUN MODE / STATIC"
                if run_mode == "static"
                else "RUN MODE / HELD"
            )
            self.state.run_mode_text = "STATIC" if run_mode == "static" else "HELD"
        service_cfg = self.plc_cfg.get("service_panel", {}) or {}
        self.service_panel = ServicePanel(service_cfg)
        self.soak_test = ServiceSoakTest(
            service_cfg.get("soak_test", {}) or {},
            LOGS_ROOT,
        )
        self._soak_previous_auto_inspect = None
        self._soak_tracking_wait_started_epoch = 0.0
        self._soak_tracking_wait_log_epoch = 0.0
        print(f"[STARTUP] mode={self.startup_mode.upper()}")

        # load saved tracker template if exists
        self._load_alignment_template()

        self.win = "Static Mode - ROI Setup"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.frame_width, self.frame_height)
        cv2.setMouseCallback(self.win, self._mouse_router)
        self.UICmd = UICmd
        self.baseline_path = os.path.join(ROI_DIR, "baseline_profile.json")
        self.baseline = AutoBaseline(self.baseline_path)
        self.baseline_debug = False
        self._inspect_frame_idx = 0
        self._plc_test_last_request_id = ""
        self._last_good_camera_frame_epoch = 0.0
        self._normal_reference_counter = 0

        # Non-invasive runtime diagnostics. These values do not change the
        # camera/inspection/PLC production paths.
        self._process_started_epoch = time.time()
        self._camera_frame_seq = 0
        self._last_camera_read_ms = 0.0
        self._max_camera_read_ms = 0.0
        self._slow_camera_read_count = 0
        self._loop_fps = 0.0
        self._loop_fps_last_epoch = time.time()
        self._loop_fps_frames = 0

        # Service data is sampled only while the panel is visible. This avoids
        # copying PLC event lists and ROI debug images on every camera frame.
        self._service_cache_epoch = 0.0
        self._service_roi_cache_epoch = 0.0
        self._service_plc_snapshot = {}
        self._service_recent_events = []
        self._service_roi_debug = None

        # Isolated Argus capture-process recovery state.
        self._camera_process_last_state = ""
        self._camera_recovery_plc_status_before = None
        self._camera_recovery_logged_count = 0
        self._camera_recovery_request_id = ""
        self._pending_plc_camera_command = None
        self._camera_recovery_soak_retry = False
        self._plc_auto_reconnect_pending = False

    def _get_primary_anchor_roi_id(self):
        align_cfg = self.product_profile.get("align", {}) or {}
        anchors = align_cfg.get("anchors") or []
        for a in anchors:
            if isinstance(a, dict) and bool(a.get("enabled", True)):
                roi_id = a.get("roi_id")
                if roi_id is not None:
                    return int(roi_id)
        return self.roi_mgr.selected_id
    
    def _update_baseline_from_ok_results(self):
        st = self.state

        if not st.last_overall_ok:
            print("[BASELINE UPDATE] skipped: last result is not OK")
            return False

        if not st.last_results:
            print("[BASELINE UPDATE] skipped: no last_results")
            return False

        updater = AutoBaseline(self.baseline_path)
        if not updater.load():
            print(f"[BASELINE UPDATE] skipped: baseline file not found -> {self.baseline_path}")
            return False

        updated_count = 0
        max_count = int(self.runtime_cfg.get("baseline_max_count", 200))

        for roi_id, res in st.last_results.items():
            metrics = getattr(res, "metrics", None) or {}
            roi_name = f"ROI{roi_id}"

            if roi_name in ("ROI2", "ROI3", "ROI4", "ROI5"):
                v = metrics.get("dark_ratio", None)
                if v is not None:
                    updater.update_from_ok_result(
                        roi_name, "dark_ratio", v, max_count=max_count
                    )
                    updated_count += 1

            elif roi_name == "ROI6":
                v = metrics.get("blob_count", None)
                if v is None:
                    v = metrics.get("blob", None)
                if v is not None:
                    updater.update_from_ok_result(
                        roi_name, "blob_count", v, max_count=max_count
                    )
                    updated_count += 1

        if updated_count <= 0:
            print("[BASELINE UPDATE] skipped: no valid metrics extracted")
            return False

        if not updater.stats:
            print("[BASELINE UPDATE] skipped: updater.stats empty")
            return False

        updater.save()
        self.baseline = updater
        print(f"[BASELINE UPDATE] updated features = {updated_count}")
        return True
    
    def _load_alignment_template(self):
        try:
            aligner = getattr(self.inspector, "aligner", None)
            if aligner is None or not hasattr(aligner, "load_templates_from_disk"):
                print("[WARN] alignment template loader is unavailable")
                return 0

            loaded = int(aligner.load_templates_from_disk() or 0)
            if loaded > 0:
                print(f"[INFO] alignment template loaded: {loaded}")
            else:
                print("[WARN] no alignment template loaded; runtime ROI template will be used")
            return loaded
        except Exception as e:
            print("[WARN] failed to load alignment template:", e)
            return 0

    # -------------------------
    # Input handlers
    # -------------------------
    def _mouse_router(self, event, x, y, flags, param):
        st = self.state

        if (not st.edit_mode) and self.service_panel.handle_mouse(event, x, y):
            action = self.service_panel.pop_action()
            if action:
                self._handle_service_action(action)
            return

        if event == cv2.EVENT_MOUSEMOVE:
            if st.edit_mode:
                self.editor._on_mouse(event, x, y, flags, None)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            for b in (st.last_buttons or []):
                x1, y1, x2, y2 = b.get("rect", (0,0,0,0))
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if b.get("enabled", True):
                        bid = b.get("id")
                        if bid == "quit":
                            st.quit_requested = True
                        else:
                            st.pending_cmd = button_id_to_cmd(bid, UICmd)
                    return

            if st.edit_mode:
                self.editor._on_mouse(event, x, y, flags, None)
            return

        if st.edit_mode:
            self.editor._on_mouse(event, x, y, flags, None)

    # -------------------------
    # Core actions
    # -------------------------
    def _toggle_mode(self):
        st = self.state
        st.edit_mode = not st.edit_mode

        try:
            self.inspector.reset_temporal_filters()
        except Exception as e:
            print("[WARN] failed to reset temporal filters:", e)

        try:
            self.stabilizer.reset()
            st.tracking_stable = False
            st.stable_frame_count = 0
            st._trk_frame_idx = 0
            st._trk_cache = None
            st._trk_smoothed_cache = None
            st._trk_motion_stable = False

            aligner = getattr(self.inspector, "aligner", None)

            if st.edit_mode:
                if aligner is not None:
                    aligner.reset_templates()
            else:
                self._load_alignment_template()
        except Exception as e:
            print("[WARN] failed to reset tracker state on mode change:", e)

        if st.edit_mode:
            self._set_plc_vision_ready(False, "EDIT_MODE")
            if self.soak_test.active:
                self._stop_service_soak_test("EDIT_MODE_ENTERED")
            st.status = "EDIT MODE"
            st.last_overall_ok = None
            st.last_results = None
            self.service_panel.close()
        else:
            run_mode = str(self.runtime_cfg.get("run_mode", "held")).lower()
            st.status = "RUN MODE / STATIC" if run_mode == "static" else "RUN MODE / HELD"
            st.run_mode_text = "STATIC" if run_mode == "static" else "HELD"

    def _save_roi_and_template(self, frame_gray8):
        st = self.state
        try:
            profile_name = str(self.runtime_cfg.get("profile_name", "") or "").strip()
            recipe_name = str(self.product_profile.get("recipe_name", "") or "").strip()

            roi_path = ROI_PATH
            if profile_name:
                roi_path = os.path.join(PROFILES_DIR, f"{profile_name}_roi.json")

            tpl_path = TEMPLATE_PATH
            if recipe_name:
                tpl_path = os.path.join(PROFILES_DIR, f"align_template_{recipe_name}.png")

            self.roi_mgr.save(roi_path)

            anchor_roi_id = self._get_primary_anchor_roi_id()
            ok_tpl = self.roi_mgr.save_alignment_template(
                frame_gray8, tpl_path, roi_id=anchor_roi_id
            )
            if ok_tpl:
                if getattr(self.inspector, "aligner", None) is not None:
                    self.inspector.aligner.reset_templates()
                self._load_alignment_template()
                st.status = f"Saved ROI + Align Template (ROI{anchor_roi_id})"
            else:
                st.status = "Saved ROI"
        except Exception as e:
            st.status = f"Save failed: {e}"

    def _toggle_auto_inspect(self):
        st = self.state
        st.auto_inspect = not st.auto_inspect
        st.last_auto_inspect_ts = 0.0
        run_mode = str(self.runtime_cfg.get("run_mode", "held")).lower()
        mode_text = "STATIC" if run_mode == "static" else "HELD"
        st.status = f"AUTO INSPECT ON / {mode_text}" if st.auto_inspect else f"AUTO INSPECT OFF / {mode_text}"

    def _toggle_service_panel(self):
        if self.state.edit_mode:
            self.service_panel.close()
            self.state.status = "SERVICE PANEL AVAILABLE IN RUN MODE"
            return
        self.service_panel.request_toggle()

    def _handle_service_action(self, action: str):
        action = str(action or "").strip().lower()

        if action == "soak:start":
            ok, message = self._start_service_soak_test()
            self.service_panel.set_message(message, 3.0)
            return

        if action == "soak:stop":
            ok, message = self._stop_service_soak_test("USER_STOP")
            self.service_panel.set_message(message, 3.0)
            return

        if action == "plc:vision_ready:100":
            try:
                snapshot = self.plc.get_state_snapshot()
                serial_open = bool(snapshot.get("serial_open", False))
                comm_fault = bool(snapshot.get("comm_fault_active", False))

                if not serial_open or comm_fault:
                    self.service_panel.set_message(
                        "D204 TEST BLOCKED: PLC LINK NOT READY",
                        3.0,
                    )
                    return

                # SERVICE test re-arms the one-shot ready handshake.
                self.state.plc_ready_consumed = False
                self.plc.set_service_vision_ready()
                self.state.plc_vision_ready = True

                updated = self.plc.get_state_snapshot()
                d201 = int(updated.get("status", -1))
                d202 = int(updated.get("result", -1))
                d204 = int(updated.get("vision_ready", -1))

                try:
                    self.plc.trace_application_event(
                        event="SERVICE_D204_READY_100",
                        summary=(
                            "service panel manual set "
                            f"D201={d201} D202={d202} D204={d204}"
                        ),
                        extra={
                            "D201": d201,
                            "D202": d202,
                            "D204": d204,
                            "source": "SERVICE_UI",
                        },
                    )
                except Exception:
                    pass

                print(
                    f"[PLC READY TEST] D201={d201} D202={d202} D204={d204}"
                )

            except Exception as e:
                print(f"[PLC READY TEST] D204=100 failed: {e}")
                self.service_panel.set_message(
                    f"D204 TEST FAILED: {e}",
                    3.0,
                )
            return

        if action == "test:recovery:camera":
            ok, message = self._start_camera_recovery_demo(
                request_id=f"ui-{int(time.time() * 1000)}",
                source="SERVICE_UI",
            )
            self.service_panel.set_message(message, 4.0)
            return

        if action == "test:local_recover":
            if not self.state.test_error_active:
                self.service_panel.set_message("NO ACTIVE TEST ERROR", 2.0)
                return
            ok = self._handle_plc_test_reset(
                reset_heartbeat=False,
                source="SERVICE_LOCAL",
            )
            self.service_panel.set_message(
                "LOCAL RECOVERY PASS" if ok else "LOCAL RECOVERY FAILED",
                3.0,
            )
            return

        if not action.startswith("test:"):
            return

        parts = action.split(":", 2)
        if len(parts) != 3:
            self.service_panel.set_message("INVALID TEST ACTION", 2.0)
            return

        test_mode = parts[1].strip()
        error_type = parts[2].strip()
        request_id = f"ui-{int(time.time() * 1000)}"
        ok = self._inject_plc_test_error(
            error_type=error_type,
            request_id=request_id,
            custom_message=(
                "Service panel PLC logic test"
                if test_mode == "logic"
                else "Service panel safe recovery test"
            ),
            test_mode=test_mode,
        )

        if ok:
            reset_hint = (
                "D200 ALREADY 3 - USE LOCAL RECOVER OR RE-ISSUE RESET"
                if self.state.test_error_pre_command == 3
                else "WAIT PLC D200=3 OR LOCAL RECOVER"
            )
            self.service_panel.set_message(
                f"{test_mode.upper()} / {error_type.upper()} - {reset_hint}",
                4.0,
            )
        else:
            reason = str(self.state.test_error_block_reason or "TEST REJECTED")
            self.service_panel.set_message(reason, 3.0)

    def _get_service_test_summary(self) -> Dict[str, Any]:
        st = self.state
        test_cfg = self._get_plc_error_test_cfg()
        return {
            "enabled": bool(test_cfg.get("enabled", False)),
            "mode": str(st.test_error_mode),
            "phase": str(st.test_error_phase),
            "type": str(st.test_error_type),
            "expected_code": int(st.test_error_expected_code),
            "actual_code": int(st.test_error_actual_code),
            "result": str(st.test_error_result),
            "health_detail": str(st.test_error_health_detail),
            "block_reason": str(st.test_error_block_reason),
            "log_saved": bool(st.test_error_log_saved),
            "request_id": str(st.test_error_request_id),
            "injected_at": str(st.test_error_injected_at),
            "recovered_at": str(st.test_error_recovered_at),
            "pre_command": int(st.test_error_pre_command),
            "pre_status": int(st.test_error_pre_status),
            "pre_result": int(st.test_error_pre_result),
            "pre_error_code": int(st.test_error_pre_error_code),
            "reset_source": str(st.test_error_reset_source),
            "recovery_elapsed_ms": int(st.test_error_recovery_elapsed_ms),
            "camera_recovery_capable": bool(self._camera_recovery_capable()),
            "camera_process": self._get_camera_process_status(),
        }


    def _get_camera_process_status(self) -> Dict[str, Any]:
        getter = getattr(self.cam, "get_status", None)
        if not callable(getter):
            return {
                "state": "DIRECT",
                "is_opened": bool(
                    getattr(getattr(self.cam, "cap", None), "isOpened", lambda: False)()
                ),
                "recovery_count": 0,
                "recovery_elapsed_sec": 0.0,
                "last_recovery_result": "",
                "detail": "direct in-process camera",
            }
        try:
            return dict(getter() or {})
        except Exception as e:
            return {
                "state": "FAILED",
                "is_opened": False,
                "recovery_count": 0,
                "recovery_elapsed_sec": 0.0,
                "last_recovery_result": "FAIL",
                "detail": f"camera status read failed: {e}",
            }

    def _camera_recovery_capable(self) -> bool:
        return bool(
            getattr(self.cam, "supports_auto_recovery", False)
            and callable(getattr(self.cam, "request_recovery", None))
        )

    def _camera_is_recovering(self) -> bool:
        return str(self._get_camera_process_status().get("state", "")) == "RECOVERING"

    def _camera_frame_healthy(self, max_age_sec: float = 0.35) -> bool:
        status = self._get_camera_process_status()
        if str(status.get("state", "")) == "DIRECT":
            return True
        if str(status.get("state", "")) != "RUNNING":
            return False
        age = status.get("frame_age_sec")
        if age is None:
            return False
        try:
            return float(age) <= max(0.1, float(max_age_sec))
        except Exception:
            return False
        
    def _set_plc_vision_ready(self, ready: bool, reason: str = ""):
        ready = bool(ready)

        if self.state.plc_vision_ready == ready:
            return

        try:
            self.plc.set_vision_ready(ready)
            self.state.plc_vision_ready = ready

            print(
                f"[PLC READY] D204={100 if ready else 0} "
                f"reason={reason or '-'}"
            )

        except Exception as e:
            print(f"[PLC READY] update failed: {e}")

    def _consume_plc_vision_ready(self, reason: str = "D200_PREPARE"):
        """Clear the one-shot D204 ready signal when D200=1 is received."""
        self.state.plc_ready_consumed = True
        self._set_plc_vision_ready(False, reason)

    def _start_camera_recovery_demo(
        self,
        request_id: str,
        source: str = "SERVICE_UI",
    ):
        st = self.state
        if not self._camera_recovery_capable():
            return False, "CAMERA PROCESS RECOVERY IS NOT ENABLED"
        if st.edit_mode:
            return False, "CAMERA RECOVERY TEST REQUIRES RUN MODE"
        if self.soak_test.active:
            return False, "CAMERA RECOVERY TEST BLOCKED: SOAK ACTIVE"
        if st.camera_error_latched or st.light_error_latched:
            return False, "CAMERA RECOVERY TEST BLOCKED: REAL FAULT ACTIVE"
        if st.test_error_active or self._camera_is_recovering():
            return False, "CAMERA RECOVERY TEST ALREADY ACTIVE"

        try:
            snapshot = self.plc.get_state_snapshot()
        except Exception:
            snapshot = {}

        st.test_error_active = True
        st.test_error_mode = "recovery"
        st.test_error_type = "camera"
        st.test_error_code = 11
        st.test_error_expected_code = 0
        st.test_error_actual_code = 0
        st.test_error_injected_at = time.strftime("%Y-%m-%d %H:%M:%S")
        st.test_error_request_id = str(request_id)
        st.test_error_phase = "WAIT_CAMERA_STALL"
        st.test_error_result = ""
        st.test_error_health_detail = "SIMULATED CAPTURE STALL - AUTO RECOVERY WAIT"
        st.test_error_block_reason = ""
        st.test_error_recovered_at = ""
        st.test_error_pre_command = int(snapshot.get("command", 0))
        st.test_error_pre_status = int(snapshot.get("status", 0))
        st.test_error_pre_result = int(snapshot.get("result", 0))
        st.test_error_pre_error_code = int(snapshot.get("error_code", 0))
        st.test_error_reset_source = "AUTO_CAMERA_PROCESS"
        st.test_error_started_monotonic = time.monotonic()
        st.test_error_recovery_elapsed_ms = 0
        self._camera_recovery_request_id = str(request_id)

        ok = bool(
            self.cam.simulate_hang(
                reason=f"camera recovery test source={source} request_id={request_id}"
            )
        )
        if not ok:
            st.test_error_active = False
            st.test_error_phase = "START_FAILED"
            st.test_error_result = "FAIL"
            st.test_error_health_detail = "FAILED TO INJECT SIMULATED CAMERA STALL"
            return False, st.test_error_health_detail

        self._save_plc_test_event(
            phase="CAMERA_STALL_INJECTED",
            result="",
            message=st.test_error_health_detail,
            extra={
                "source": str(source),
                "request_id": str(request_id),
                "plc_error_changed": False,
                "camera_process_isolated": True,
            },
        )
        st.status = "CAMERA RECOVERY TEST: STALL INJECTED"
        return True, "CAMERA STALL INJECTED - WATCH AUTO RECOVERY TIME"

    def _poll_camera_process_request(self):
        path = str(getattr(self.cam, "request_path", "") or "")
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as file:
                request = json.load(file)
        except Exception as e:
            print("[CAM RECOVERY] invalid request:", e)
            try:
                os.remove(path)
            except OSError:
                pass
            return
        try:
            os.remove(path)
        except OSError:
            pass

        action = str(request.get("action", "simulate_hang") or "simulate_hang").lower()
        request_id = str(
            request.get("request_id", f"file-{int(time.time() * 1000)}")
        )
        if request_id == self._camera_recovery_request_id:
            return
        if action == "simulate_hang":
            ok, message = self._start_camera_recovery_demo(
                request_id=request_id,
                source="FILE_REQUEST",
            )
            print(f"[CAM RECOVERY TEST] {message} ok={ok}")
        elif action == "recover":
            self._camera_recovery_request_id = request_id
            ok = bool(self.cam.request_recovery(f"file request {request_id}"))
            print(f"[CAM RECOVERY] manual request={request_id} accepted={ok}")

    def _reset_tracking_after_camera_recovery(self):
        st = self.state
        st.tracking_stable = False
        st.stable_frame_count = 0
        st.run_mode_text = "HELD"
        try:
            self.stabilizer.reset()
            st._trk_frame_idx = 0
            st._trk_cache = None
            st._trk_smoothed_cache = None
            st._trk_motion_stable = False
        except Exception as e:
            print("[CAM RECOVERY] stabilizer reset failed:", e)
        try:
            aligner = getattr(self.inspector, "aligner", None)
            if aligner is not None:
                aligner.reset_templates()
            self._load_alignment_template()
        except Exception as e:
            print("[CAM RECOVERY] tracker reload failed:", e)

    def _handle_camera_recovery_plc_tick(self):
        try:
            snapshot = self.plc.get_state_snapshot()
            command_value = int(snapshot.get("command", 0))
        except Exception:
            return

        if command_value in (3, 8, 9):
            command = self.plc.poll_command()
            if command == "reset":
                self._handle_plc_reset()
            elif command == "shutdown":
                self._handle_plc_shutdown()
            elif command == "emergency":
                self._handle_plc_emergency()
            return

        if command_value in (1, 2):
            command = self.plc.poll_command()
            if command == "prepare":
                self._consume_plc_vision_ready("D200_PREPARE_RECEIVED")
            if command in ("prepare", "inspect"):
                self._pending_plc_camera_command = command
            self.plc.set_busy()

    def _sync_camera_process_state(self) -> Dict[str, Any]:
        status = self._get_camera_process_status()
        state = str(status.get("state", ""))
        previous = str(self._camera_process_last_state or "")
        st = self.state

        if state == "RECOVERING":
            self._set_plc_vision_ready(False, "CAMERA_RECOVERING")
            if previous != "RECOVERING":
                try:
                    snapshot = self.plc.get_state_snapshot()
                    before_status = int(snapshot.get("status", 0))
                except Exception:
                    before_status = 0
                self._camera_recovery_plc_status_before = before_status
                if before_status in (0, 1):
                    self.plc.set_busy()
                st.camera_error_latched = False
                st.camera_read_blocked = False

                recovery_count = int(status.get("recovery_count", 0))
                if recovery_count != self._camera_recovery_logged_count:
                    self._camera_recovery_logged_count = recovery_count
                    try:
                        self.plc.trace_application_event(
                            event="CAMERA_AUTO_RECOVERY_STARTED",
                            summary=(
                                f"count={recovery_count} "
                                f"reason={status.get('last_recovery_reason', '')}"
                            ),
                            extra=status,
                        )
                    except Exception:
                        pass

            elapsed = float(status.get("recovery_elapsed_sec", 0.0) or 0.0)
            st.status = f"CAMERA AUTO RECOVERY: {elapsed:.1f}s"
            if st.test_error_active and st.test_error_type == "camera":
                st.test_error_phase = "RECOVERING"
                st.test_error_health_detail = (
                    f"CAMERA PROCESS RECOVERY IN PROGRESS {elapsed:.1f}s"
                )
                st.test_error_recovery_elapsed_ms = int(elapsed * 1000.0)

        elif state == "RUNNING" and previous == "RECOVERING":
            elapsed = float(status.get("recovery_elapsed_sec", 0.0) or 0.0)
            st.camera_error_latched = False
            st.camera_read_blocked = False
            recovery_cfg = self._get_plc_recovery_cfg()
            st.camera_error_grace_until = time.time() + recovery_cfg["camera_grace_sec"]
            self._reset_tracking_after_camera_recovery()

            try:
                snapshot = self.plc.get_state_snapshot()
                command_value = int(snapshot.get("command", 0))
            except Exception:
                command_value = 0
            if self._camera_recovery_soak_retry and self.soak_test.active:
                self._camera_recovery_soak_retry = False
                self.soak_test.phase = "WAIT_INSPECT"
                self.soak_test.next_action_epoch = time.time() + 0.2
                self.soak_test.record_external_event(
                    "CAMERA_AUTO_RECOVERY_RESUME",
                    {"camera_process": status},
                )
                self.plc.set_idle()
            elif (
                self._camera_recovery_plc_status_before == 0
                and command_value not in (1, 2)
                and not self.soak_test.active
            ):
                self.plc.set_idle()

            if st.test_error_active and st.test_error_type == "camera":
                st.test_error_phase = "RECOVERY_COMPLETE"
                st.test_error_result = "PASS"
                st.test_error_health_detail = (
                    f"CAMERA AUTO RECOVERY OK: {elapsed:.3f}s "
                    f"WORKER#{status.get('generation', 0)}"
                )
                st.test_error_recovery_elapsed_ms = int(elapsed * 1000.0)
                st.test_error_recovered_at = time.strftime("%Y-%m-%d %H:%M:%S")
                st.test_error_active = False
                self._save_plc_test_event(
                    phase="RECOVERY_COMPLETE",
                    result="PASS",
                    message=st.test_error_health_detail,
                    extra={"camera_process": status},
                )

            st.status = f"CAMERA RECOVERED {elapsed:.2f}s - TRACKING REACQUIRE"
            try:
                self.plc.trace_application_event(
                    event="CAMERA_AUTO_RECOVERY_COMPLETED",
                    summary=f"elapsed={elapsed:.3f}s",
                    extra=status,
                )
            except Exception:
                pass
            self._camera_recovery_plc_status_before = None

        elif state == "FAILED":
            if previous != "FAILED":
                detail = str(status.get("detail", "camera process recovery failed"))
                try:
                    plc_snapshot = self.plc.get_state_snapshot()
                    active_error_code = int(plc_snapshot.get("error_code", 0))
                except Exception:
                    active_error_code = 0

                st.camera_error_latched = True
                st.camera_read_blocked = True

                # Initial camera startup recovery failure is already logged as
                # CAMERA_INIT_FAILED(10). Do not overwrite it with runtime code 11.
                if active_error_code == 10:
                    st.status = "CAMERA INIT AUTO RECOVERY FAILED: PLC RESET REQUIRED"
                else:
                    st.status = "CAMERA AUTO RECOVERY FAILED: PLC RESET REQUIRED"
                    self._save_plc_error_event(
                        event_type="CAMERA_AUTO_RECOVERY_FAILED",
                        error_code=11,
                        message="Isolated camera process recovery failed",
                        exception=RuntimeError(detail),
                    )
                    self.plc.set_error(code=11, detail=detail)

                if st.test_error_active and st.test_error_type == "camera":
                    st.test_error_phase = "RECOVERY_FAILED"
                    st.test_error_result = "FAIL"
                    st.test_error_health_detail = detail
                    st.test_error_recovery_elapsed_ms = int(
                        float(status.get("recovery_elapsed_sec", 0.0) or 0.0) * 1000.0
                    )
                    st.test_error_active = False

        self._camera_process_last_state = state
        return status

    def _get_process_rss_mb(self) -> float:
        try:
            with open("/proc/self/statm", "r", encoding="utf-8") as file:
                pages = int(file.read().split()[1])
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return float(pages * page_size) / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    def _get_runtime_health_snapshot(self) -> Dict[str, Any]:
        now = time.time()
        frame_age = (
            max(0.0, now - self._last_good_camera_frame_epoch)
            if self._last_good_camera_frame_epoch > 0
            else None
        )
        try:
            light_state = self.light.get_state() if self.light is not None else {}
        except Exception as e:
            light_state = {"read_failed": True, "error": str(e)}
        return {
            "pid": int(os.getpid()),
            "process_uptime_sec": max(0.0, now - self._process_started_epoch),
            "service_panel_visible": bool(self.service_panel.visible),
            "camera_frame_seq": int(self._camera_frame_seq),
            "camera_last_frame_age_sec": frame_age,
            "camera_read_last_ms": round(float(self._last_camera_read_ms), 3),
            "camera_read_max_ms": round(float(self._max_camera_read_ms), 3),
            "camera_slow_read_count": int(self._slow_camera_read_count),
            "camera_process": self._get_camera_process_status(),
            "loop_fps": round(float(self._loop_fps), 3),
            "rss_mb": round(self._get_process_rss_mb(), 2),
            "tracking_stable": bool(self.state.tracking_stable),
            "stable_frame_count": int(self.state.stable_frame_count),
            "camera_error_latched": bool(self.state.camera_error_latched),
            "camera_read_blocked": bool(self.state.camera_read_blocked),
            "light_error_latched": bool(self.state.light_error_latched),
            "light_state": light_state,
        }

    def _start_service_soak_test(self):
        st = self.state
        if not self.soak_test.enabled:
            return False, "SOAK TEST DISABLED"
        if self.soak_test.active:
            return False, "SOAK TEST ALREADY RUNNING"
        if st.edit_mode:
            return False, "SOAK TEST REQUIRES RUN MODE"
        if st.test_error_active:
            return False, "SOAK TEST BLOCKED: ERROR TEST ACTIVE"
        if st.camera_error_latched or st.light_error_latched:
            return False, "SOAK TEST BLOCKED: VISION FAULT ACTIVE"

        try:
            snapshot = self.plc.get_state_snapshot()
        except Exception as e:
            return False, f"SOAK TEST BLOCKED: PLC SNAPSHOT FAILED ({e})"

        command = int(snapshot.get("command", 0))
        status = int(snapshot.get("status", 0))
        if not bool(snapshot.get("serial_open", False)):
            return False, "SOAK TEST BLOCKED: PLC SERIAL CLOSED"
        if bool(snapshot.get("comm_fault_active", False)):
            return False, "SOAK TEST BLOCKED: PLC COMM FAULT"
        if status == 3 or int(snapshot.get("error_code", 0)) != 0:
            return False, "SOAK TEST BLOCKED: ERROR STATE ACTIVE"
        if status == 8 or command in (8, 9):
            return False, "SOAK TEST BLOCKED: SHUTDOWN/EMERGENCY"
        if status == 1:
            return False, "SOAK TEST BLOCKED: CURRENTLY BUSY"

        # The operator explicitly starts the test. Clear an old DONE/OK/NG
        # state once before the first cycle, using the existing reset path.
        if status == 2 or int(snapshot.get("result", 0)) != 0:
            if not self._handle_plc_reset(reset_heartbeat=False):
                return False, "SOAK TEST START FAILED: INITIAL RESET FAILED"
            snapshot = self.plc.get_state_snapshot()

        self._soak_previous_auto_inspect = bool(st.auto_inspect)
        st.auto_inspect = False
        self.runtime_cfg["_service_soak_active"] = True
        ok = self.soak_test.start(
            plc_snapshot=snapshot,
            vision_snapshot=self._get_vision_state_snapshot(),
        )
        if not ok:
            st.auto_inspect = bool(self._soak_previous_auto_inspect)
            self.runtime_cfg["_service_soak_active"] = False
            self._soak_previous_auto_inspect = None
            return False, "SOAK TEST START FAILED"

        st.status = "SOAK TEST STARTED: 30 SEC CYCLE"
        try:
            self.plc.trace_application_event(
                event="SOAK_TEST_STARTED",
                summary=(
                    f"session={self.soak_test.session_id} "
                    f"interval={self.soak_test.interval_sec:.1f}s"
                ),
                extra={
                    "soak_session_id": self.soak_test.session_id,
                    "soak_log_path": self.soak_test.log_path,
                },
            )
        except Exception:
            pass
        return True, "SOAK TEST STARTED"

    def _stop_service_soak_test(self, reason: str):
        if not self.soak_test.active:
            return False, "SOAK TEST IS NOT RUNNING"
        if bool(getattr(self.state, "spot_armed", False)):
            try:
                self._spot_release(reason="soak_stop")
            except Exception as e:
                print(f"[SOAK TEST] spot release on stop failed: {e}")
        self._soak_tracking_wait_started_epoch = 0.0
        self._soak_tracking_wait_log_epoch = 0.0
        try:
            plc_snapshot = self.plc.get_state_snapshot()
        except Exception as e:
            plc_snapshot = {"snapshot_failed": True, "error": str(e)}
        self.soak_test.stop(
            reason=reason,
            plc_snapshot=plc_snapshot,
            vision_snapshot=self._get_vision_state_snapshot(),
        )
        if self._soak_previous_auto_inspect is not None:
            self.state.auto_inspect = bool(self._soak_previous_auto_inspect)
        self.runtime_cfg["_service_soak_active"] = False
        self._soak_previous_auto_inspect = None
        self.state.status = f"SOAK TEST STOPPED: {reason}"
        try:
            self.plc.trace_application_event(
                event="SOAK_TEST_STOPPED",
                summary=(
                    f"session={self.soak_test.session_id} reason={reason} "
                    f"cycles={self.soak_test.cycle_count} "
                    f"ok={self.soak_test.ok_count} ng={self.soak_test.ng_count} "
                    f"errors={self.soak_test.error_count}"
                ),
                extra={
                    "soak_session_id": self.soak_test.session_id,
                    "soak_log_path": self.soak_test.log_path,
                    "soak_summary_path": self.soak_test.summary_path,
                },
            )
        except Exception:
            pass
        return True, "SOAK TEST STOPPED / LOG FINALIZED"

    def _fail_service_soak_test(self, reason: str, exception=None):
        if bool(getattr(self.state, "spot_armed", False)):
            try:
                self._spot_release(reason="soak_error")
            except Exception as e:
                print(f"[SOAK TEST] spot release on error failed: {e}")
        self._soak_tracking_wait_started_epoch = 0.0
        self._soak_tracking_wait_log_epoch = 0.0
        try:
            plc_snapshot = self.plc.get_state_snapshot()
        except Exception as e:
            plc_snapshot = {"snapshot_failed": True, "error": str(e)}
        self.soak_test.fail(
            reason=reason,
            plc_snapshot=plc_snapshot,
            vision_snapshot=self._get_vision_state_snapshot(),
            exception=exception,
        )
        if self._soak_previous_auto_inspect is not None:
            self.state.auto_inspect = bool(self._soak_previous_auto_inspect)
        self.runtime_cfg["_service_soak_active"] = False
        self._soak_previous_auto_inspect = None
        self.state.status = f"SOAK TEST STOPPED ON ERROR: {reason}"

    def _validate_service_soak_result(self):
        overall_ok = self.state.last_overall_ok
        results = self.state.last_results
        if overall_ok is None:
            raise InspectionResultError("inspection overall result was not generated")
        if not isinstance(results, dict) or not results:
            raise InspectionResultError("inspection ROI results were not generated")
        invalid = []
        for roi_id, result in results.items():
            roi_ok = (
                result.get("ok")
                if isinstance(result, dict)
                else getattr(result, "ok", None)
            )
            if roi_ok is None:
                invalid.append(str(roi_id))
        if invalid:
            raise InspectionResultError(
                "ROI result has no OK/NG value: " + ",".join(invalid)
            )
        return bool(overall_ok)

    def _service_soak_tracking_ready(self) -> bool:
        run_mode = str(self.runtime_cfg.get("run_mode", "held") or "held").lower()
        if run_mode == "static":
            return True
        required = max(
            1,
            int(
                getattr(
                    self.soak_test,
                    "tracking_stable_frames",
                    self.runtime_cfg.get("auto_inspect_stable_frames", 3),
                )
            ),
        )
        return bool(
            self.state.tracking_stable
            and self.state.stable_frame_count >= required
        )

    def _execute_service_soak_inspection(
        self,
        frame_gray8,
        vis_bgr,
        *,
        allow_camera_flush: bool,
    ):
        st = self.state
        try:
            started = time.perf_counter()
            self._run_spot_inspect_once(
                frame_gray8,
                vis_bgr,
                avg5=bool(self.runtime_cfg.get("plc_inspect_avg5", False)),
                trigger="SOAK",
                allow_camera_flush=allow_camera_flush,
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000.0)
            overall_ok = self._validate_service_soak_result()
            self.plc.set_done(ok=overall_ok, elapsed_ms=elapsed_ms)
            self._save_normal_reference_if_due(
                overall_ok=overall_ok,
                elapsed_ms=elapsed_ms,
            )
            st.status = (
                f"SOAK CYCLE {self.soak_test.cycle_count}: OK"
                if overall_ok
                else f"SOAK CYCLE {self.soak_test.cycle_count}: NG"
            )
            self.soak_test.complete_inspection(
                overall_ok=overall_ok,
                elapsed_ms=elapsed_ms,
                plc_snapshot=self.plc.get_state_snapshot(),
                inspection_summary={
                    "overall_ok": overall_ok,
                    "roi_results": self._compact_inspection_result_summary(),
                },
                health=self._get_runtime_health_snapshot(),
            )
        except CameraRecoveryInProgress as e:
            self._camera_recovery_soak_retry = True
            self.plc.set_busy()
            self.soak_test.defer("CAMERA AUTO RECOVERY - RETRY CURRENT CYCLE", 0.2)
            self.soak_test.record_external_event(
                "CAMERA_AUTO_RECOVERY_DEFER",
                {
                    "reason": str(e),
                    "camera_process": self._get_camera_process_status(),
                },
            )
            st.status = "SOAK PAUSED: CAMERA AUTO RECOVERY"
        except LightCommunicationError as e:
            st.light_error_latched = True
            self._save_plc_error_event(
                event_type="SOAK_TEST_RUNTIME_ERROR",
                error_code=21,
                message="Light communication failed during soak inspection",
                exception=e,
            )
            self.plc.set_error(code=21, detail=str(e))
            self._fail_service_soak_test("LIGHT ERROR DURING SOAK", e)
        except InspectionResultError as e:
            self._save_plc_error_event(
                event_type="SOAK_TEST_INSPECTION_ERROR",
                error_code=40,
                message="Soak inspection result generation failed",
                exception=e,
            )
            self.plc.set_error(code=40, detail=str(e))
            self._fail_service_soak_test("INSPECTION RESULT ERROR", e)
        except Exception as e:
            self._save_plc_error_event(
                event_type="SOAK_TEST_INSPECTION_ERROR",
                error_code=41,
                message="Unexpected exception during soak inspection",
                exception=e,
            )
            self.plc.set_error(code=41, detail=str(e))
            self._fail_service_soak_test("INSPECTION EXCEPTION", e)

    def _run_service_soak_tick(self, frame_gray8, vis_bgr):
        if not self.soak_test.active:
            return

        st = self.state
        if st.edit_mode:
            self._stop_service_soak_test("EDIT_MODE_ENTERED")
            return

        try:
            snapshot = self.plc.get_state_snapshot()
        except Exception as e:
            self._fail_service_soak_test("PLC SNAPSHOT FAILED", e)
            return

        status = int(snapshot.get("status", 0))
        command = int(snapshot.get("command", 0))
        error_code = int(snapshot.get("error_code", 0))

        # Preserve unexpected faults for diagnosis. The overnight test does not
        # automatically erase a real camera/light/PLC error.
        if (
            st.camera_error_latched
            or st.light_error_latched
            or status == 3
            or error_code != 0
            or bool(snapshot.get("comm_fault_active", False))
        ):
            self._fail_service_soak_test(
                f"REAL FAULT DETECTED: STATUS={status} ERROR={error_code}"
            )
            return
        if status == 8 or command in (8, 9):
            self._stop_service_soak_test("SHUTDOWN_OR_EMERGENCY")
            return

        now = time.time()
        if now < float(self.soak_test.next_action_epoch):
            return

        if self.soak_test.phase == "WAIT_INSPECT":
            if status == 1:
                self.soak_test.defer("WAITING: PLC/VISION BUSY", 0.5)
                return
            if status != 0 or int(snapshot.get("result", 0)) != 0:
                self.soak_test.defer(
                    f"WAITING FOR READY: D201={status} "
                    f"D202={int(snapshot.get('result', 0))}",
                    0.5,
                )
                return
            if frame_gray8 is None or vis_bgr is None:
                self.soak_test.defer("WAITING FOR CAMERA FRAME", 0.2)
                return

            # Do not require tracking stability at the idle-light level.
            # In a dark installation that creates a permanent wait at 20% and
            # prevents the soak cycle from ever reaching the 90% inspection
            # light. Tracking is validated after prearm and fresh frames below.
            self._soak_tracking_wait_started_epoch = 0.0
            self._soak_tracking_wait_log_epoch = 0.0
            self.soak_test.begin_cycle(
                plc_snapshot=snapshot,
                health=self._get_runtime_health_snapshot(),
            )
            try:
                self.plc.set_busy()
                st.status = f"SOAK CYCLE {self.soak_test.cycle_count}: INSPECT BUSY"
                st.last_overall_ok = None
                st.last_results = None

                spot_cfg = self._get_spot_light_cfg()
                if bool(spot_cfg.get("enabled", False)):
                    # Service soak must not become a nested camera consumer.
                    # Ramp/settle the light, then wait for fresh frames delivered
                    # by the normal main loop before judging the product.
                    self._spot_prearm(
                        trigger="SOAK_PREPARE",
                        flush_frames=False,
                    )
                    required_frames = max(
                        1,
                        int(getattr(self.soak_test, "fresh_frame_count", 4)),
                    )
                    settle_sec = max(
                        0.0,
                        float(spot_cfg.get("inspect_settle_ms", 0)) / 1000.0,
                    )
                    self.soak_test.prepare_fresh_frame_wait(
                        start_frame_seq=self._camera_frame_seq,
                        required_frames=required_frames,
                        settle_sec=settle_sec,
                    )
                    st.status = (
                        f"SOAK CYCLE {self.soak_test.cycle_count}: "
                        f"WAIT FRESH FRAME {required_frames}"
                    )
                else:
                    self._execute_service_soak_inspection(
                        frame_gray8,
                        vis_bgr,
                        allow_camera_flush=False,
                    )
            except LightCommunicationError as e:
                st.light_error_latched = True
                self._save_plc_error_event(
                    event_type="SOAK_TEST_RUNTIME_ERROR",
                    error_code=21,
                    message="Light communication failed during soak prearm",
                    exception=e,
                )
                self.plc.set_error(code=21, detail=str(e))
                self._fail_service_soak_test("LIGHT PREARM ERROR DURING SOAK", e)
            except Exception as e:
                self._save_plc_error_event(
                    event_type="SOAK_TEST_INSPECTION_ERROR",
                    error_code=41,
                    message="Unexpected exception during soak preparation",
                    exception=e,
                )
                self.plc.set_error(code=41, detail=str(e))
                self._fail_service_soak_test("SOAK PREPARATION EXCEPTION", e)
            return

        if self.soak_test.phase == "WAIT_FRESH_FRAME":
            if frame_gray8 is None or vis_bgr is None:
                self.soak_test.defer("WAITING FOR FRESH CAMERA FRAME", 0.1)
                return
            if not bool(getattr(st, "spot_armed", False)):
                self._fail_service_soak_test("SPOT PREARM LOST BEFORE INSPECTION")
                return
            if not self.soak_test.is_fresh_frame_ready(self._camera_frame_seq):
                return

            # The tracker must reacquire under the real inspection brightness,
            # not under the dark 20% idle condition. Keep the 90% light armed,
            # wait without counting NG, and log the condition periodically.
            if not self._service_soak_tracking_ready():
                now = time.time()
                if self._soak_tracking_wait_started_epoch <= 0:
                    self._soak_tracking_wait_started_epoch = now
                log_interval = max(
                    1.0,
                    float(getattr(self.soak_test, "tracking_wait_log_sec", 5.0)),
                )
                if (now - self._soak_tracking_wait_log_epoch) >= log_interval:
                    self._soak_tracking_wait_log_epoch = now
                    self.soak_test.record_external_event(
                        "WAIT_TRACKING_INSPECT_LIGHT",
                        {
                            "wait_sec": round(
                                now - self._soak_tracking_wait_started_epoch,
                                3,
                            ),
                            "tracking_stable": bool(st.tracking_stable),
                            "stable_frame_count": int(st.stable_frame_count),
                            "required_frames": int(
                                getattr(self.soak_test, "tracking_stable_frames", 3)
                            ),
                            "spot_armed": bool(st.spot_armed),
                            "spot_armed_brightness": int(
                                getattr(st, "spot_armed_brightness", 0)
                            ),
                            "health": self._get_runtime_health_snapshot(),
                        },
                    )
                st.status = (
                    "SOAK WAIT: INSPECT LIGHT TRACKING "
                    f"{st.stable_frame_count}/"
                    f"{int(getattr(self.soak_test, 'tracking_stable_frames', 3))}"
                )
                self.soak_test.defer(
                    "WAITING FOR TRACKING AT INSPECTION LIGHT",
                    0.2,
                )
                return

            self._soak_tracking_wait_started_epoch = 0.0
            self._soak_tracking_wait_log_epoch = 0.0
            self._execute_service_soak_inspection(
                frame_gray8,
                vis_bgr,
                allow_camera_flush=False,
            )
            return

        if self.soak_test.phase == "WAIT_RESET":
            # If the real PLC sent D200=3 first, _run_plc_inspect_tick() has
            # already returned D201/D202 to zero. Record that real reset source.
            if status == 0 and int(snapshot.get("result", 0)) == 0:
                self.soak_test.complete_reset(
                    plc_snapshot=snapshot,
                    health=self._get_runtime_health_snapshot(),
                    source="PLC_D200",
                )
            else:
                # Otherwise use the existing production reset function locally.
                # No PLC command parser or hardware driver is changed.
                ok = self._handle_plc_reset(reset_heartbeat=False)
                if not ok:
                    self._fail_service_soak_test("VISION RESET FAILED")
                    return
                self.soak_test.complete_reset(
                    plc_snapshot=self.plc.get_state_snapshot(),
                    health=self._get_runtime_health_snapshot(),
                    source="SERVICE_LOCAL",
                )
            st.status = (
                f"SOAK TEST: {self.soak_test.cycle_count} CYCLES / "
                f"OK {self.soak_test.ok_count} NG {self.soak_test.ng_count}"
            )

    def _save_plc_test_event(
        self,
        phase: str,
        result: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        st = self.state
        try:
            plc_snapshot = self.plc.get_state_snapshot()
        except Exception as e:
            plc_snapshot = {"snapshot_failed": True, "error": str(e)}

        path = save_plc_test_log(
            logs_root=LOGS_ROOT,
            test_id=str(st.test_error_request_id),
            test_type=str(st.test_error_type),
            phase=str(phase),
            result=str(result),
            error_code=int(st.test_error_code),
            message=str(message),
            plc_snapshot=plc_snapshot,
            vision_snapshot=self._get_vision_state_snapshot(),
            extra=extra or {},
        )
        if path:
            st.test_error_log_path = str(path)
            st.test_error_log_saved = True
        return path

    def _inspection_result_summary(self) -> Dict[str, Any]:
        summary = {}
        for roi_id, result in (self.state.last_results or {}).items():
            if isinstance(result, dict):
                ok = result.get("ok")
                reason = result.get("reason")
                metrics = result.get("metrics") or {}
            else:
                ok = getattr(result, "ok", None)
                reason = getattr(result, "reason", None)
                metrics = getattr(result, "metrics", None) or {}
            summary[str(roi_id)] = {
                "ok": ok,
                "reason": reason,
                "metrics": metrics,
            }
        return summary

    def _compact_inspection_result_summary(self) -> Dict[str, Any]:
        """Keep soak logs small and JSON-friendly.

        Debug images and private runtime objects are intentionally omitted; NG
        image artifacts continue to be handled by the existing inspection log.
        """
        compact = {}
        for roi_id, result in (self.state.last_results or {}).items():
            if isinstance(result, dict):
                ok = result.get("ok")
                reason = result.get("reason")
                metrics = result.get("metrics") or {}
            else:
                ok = getattr(result, "ok", None)
                reason = getattr(result, "reason", None)
                metrics = getattr(result, "metrics", None) or {}

            safe_metrics = {}
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    key_text = str(key)
                    if key_text.startswith("_"):
                        continue
                    if value is None or isinstance(value, (str, int, float, bool)):
                        safe_metrics[key_text] = value
                    elif isinstance(value, (list, tuple)) and len(value) <= 12:
                        if all(
                            item is None or isinstance(item, (str, int, float, bool))
                            for item in value
                        ):
                            safe_metrics[key_text] = list(value)

            compact[str(roi_id)] = {
                "ok": ok,
                "reason": reason,
                "metrics": safe_metrics,
            }
        return compact

    def _save_normal_reference_if_due(
        self,
        overall_ok: bool,
        elapsed_ms: int,
    ) -> Optional[str]:
        if not bool(overall_ok):
            return None

        diag_cfg = get_diagnostics_config()
        if not bool(diag_cfg.get("enabled", True)):
            return None

        self._normal_reference_counter += 1
        every = max(1, int(diag_cfg.get("normal_sample_every", 20)))
        if self._normal_reference_counter != 1 and (
            self._normal_reference_counter % every
        ) != 0:
            return None

        try:
            plc_snapshot = self.plc.get_state_snapshot()
        except Exception as e:
            plc_snapshot = {"snapshot_failed": True, "error": str(e)}

        path = save_normal_reference_log(
            logs_root=LOGS_ROOT,
            plc_snapshot=plc_snapshot,
            vision_snapshot=self._get_vision_state_snapshot(),
            inspection_summary={
                "overall_ok": True,
                "elapsed_ms": int(elapsed_ms),
                "roi_results": self._inspection_result_summary(),
            },
        )
        if path:
            print(f"[NORMAL REFERENCE] saved: {path}")
        return path

    def _run_auto_inspect_tick(self, frame_gray8, vis_bgr):
        st = self.state
        cfg = self.runtime_cfg

        # spot 조명 검사 모드에서는 auto inspect 금지
        # 수동/PLC pre-arm 테스트 중 의도치 않은 추가 검사 방지
        try:
            spot_cfg = self._get_spot_light_cfg()
            if bool(spot_cfg.get("enabled", False)):
                return
        except Exception:
            pass

        if not (st.auto_inspect and self.product_profile["modules"].get("auto_inspect", True)):
            return

        now = time.time()
        interval = float(cfg.get("auto_inspect_interval", 0.5))
        avg5 = bool(cfg.get("auto_inspect_avg5", False))
        stable_required = int(cfg.get("auto_inspect_stable_frames", 3))
        run_mode = str(cfg.get("run_mode", "held")).lower()

        allow_inspect = False
        if run_mode == "static":
            allow_inspect = True
        else:
            allow_inspect = st.tracking_stable and st.stable_frame_count >= stable_required

        if allow_inspect and (now - st.last_auto_inspect_ts) >= interval:
            st.last_auto_inspect_ts = now
            run_inspect_once(
                cam=self.cam,
                inspector=self.inspector,
                runtime_cfg=self.runtime_cfg,
                state=st,
                frame_gray8=frame_gray8,
                vis_bgr=vis_bgr,
                avg5=avg5,
                use_cache=True,
                cache_every_n=int(cfg.get("auto_inspect_every_n", 3)),
            )

    def _get_spot_light_cfg(self):
        light_sets = self.hardware_cfg.get("light_sets", {}) or {}
        active_light_set = str(self.hardware_cfg.get("active_light_set", "") or "").strip()
        light_cfg = light_sets.get(active_light_set, {}) or {}
        spot_cfg = light_cfg.get("spot_inspect", {}) or {}

        return {
            "enabled": bool(spot_cfg.get("enabled", False)),
            "light_id": str(spot_cfg.get("light_id", "light1")),

            "idle_brightness": int(spot_cfg.get("idle_brightness", 20)),
            "inspect_brightness": int(spot_cfg.get("inspect_brightness", 90)),

            "ramp_enabled": bool(spot_cfg.get("ramp_enabled", True)),
            "ramp_steps": list(spot_cfg.get("ramp_steps", [20, 40, 60, 80, 90])),
            "ramp_step_ms": int(spot_cfg.get("ramp_step_ms", 200)),

            "pre_ready_settle_ms": int(spot_cfg.get("pre_ready_settle_ms", 400)),
            "pre_ready_flush_frames": int(spot_cfg.get("pre_ready_flush_frames", 8)),

            "inspect_settle_ms": int(spot_cfg.get("inspect_settle_ms", 50)),
            "inspect_flush_frames": int(spot_cfg.get("inspect_flush_frames", 2)),

            "armed_timeout_ms": int(spot_cfg.get("armed_timeout_ms", 3000)),
            "restore_after_inspect": bool(spot_cfg.get("restore_after_inspect", True)),
        }

    def _flush_camera_frames(self, n):
        last_gray = None
        last_vis = None

        for _ in range(max(0, int(n))):
            frame = self._read_frame()
            if frame is None:
                continue

            g, v = self._prepare_frame(frame)
            if g is not None and v is not None:
                last_gray = g
                last_vis = v

        return last_gray, last_vis

    def _set_light_brightness_checked(
        self,
        light_id: str,
        brightness: int,
        action: str,
    ):
        ok = self.light.set_brightness(
            light_id,
            brightness,
        )

        light_state = self.light.get_state() or {}
        self.runtime_cfg["_light_state"] = light_state

        info = light_state.get(light_id, {})
        state_ok = (
            info.get("ok")
            if isinstance(info, dict)
            else None
        )
        reason = (
            info.get("reason", "")
            if isinstance(info, dict)
            else ""
        )

        if ok is not True or state_ok is False:
            raise LightCommunicationError(
                f"{action} failed: "
                f"light_id={light_id}, "
                f"brightness={brightness}, "
                f"reason={reason or 'UNKNOWN'}"
            )

        return True

    def _spot_prearm(self, trigger="PLC", flush_frames: bool = True):
        st = self.state
        cfg = self._get_spot_light_cfg()

        if not bool(cfg.get("enabled", False)):
            return False

        light_id = cfg["light_id"]
        inspect_brightness = int(cfg["inspect_brightness"])

        print(
            f"[LIGHT PREARM] trigger={trigger} "
            f"idle={cfg['idle_brightness']}% inspect={inspect_brightness}% "
            f"timeout={cfg['armed_timeout_ms']}ms"
        )

        if bool(cfg["ramp_enabled"]):
            steps = cfg["ramp_steps"]
            if not steps:
                steps = [inspect_brightness]

            for b in steps:
                b = int(max(0, min(100, int(b))))
                self._set_light_brightness_checked(
                    light_id=light_id,
                    brightness=b,
                    action="SPOT_RAMP",
                )
                time.sleep(float(cfg["ramp_step_ms"]) / 1000.0)
        else:
            self._set_light_brightness_checked(
                light_id=light_id,
                brightness=inspect_brightness,
                action="SPOT_INSPECT_BRIGHTNESS",
            )

        if int(cfg["pre_ready_settle_ms"]) > 0:
            time.sleep(float(cfg["pre_ready_settle_ms"]) / 1000.0)

        if bool(flush_frames):
            self._flush_camera_frames(int(cfg["pre_ready_flush_frames"]))

        st.spot_armed = True
        st.spot_armed_ts = time.time()
        st.spot_armed_brightness = inspect_brightness
        st.status = f"{trigger} SPOT READY: {inspect_brightness}%"

        print(f"[LIGHT PREARM] ready brightness={inspect_brightness}%")
        return True

    def _spot_release(self, reason="release"):
        st = self.state
        cfg = self._get_spot_light_cfg()

        if not bool(cfg.get("enabled", False)):
            st.spot_armed = False
            return

        light_id = cfg["light_id"]
        idle_brightness = int(cfg["idle_brightness"])

        self._set_light_brightness_checked(
            light_id=light_id,
            brightness=idle_brightness,
            action="SPOT_IDLE_RESTORE",
        )

        st.spot_armed = False
        st.spot_armed_ts = 0.0
        st.spot_armed_brightness = 0

        print(f"[LIGHT PREARM] restored idle={idle_brightness}% reason={reason}")

    def _spot_timeout_tick(self):
        st = self.state
        cfg = self._get_spot_light_cfg()

        if not bool(cfg.get("enabled", False)):
            return

        if not bool(getattr(st, "spot_armed", False)):
            return

        # During a service soak cycle the inspection light must remain armed
        # while fresh frames arrive and the tracker reacquires under the actual
        # inspection brightness. The normal 3-second safety timeout would
        # otherwise restore the light to idle before tracking can stabilize.
        if (
            self.soak_test.active
            and self.soak_test.phase == "WAIT_FRESH_FRAME"
        ):
            return

        timeout_ms = int(cfg.get("armed_timeout_ms", 3000))
        if timeout_ms <= 0:
            return

        elapsed_ms = int((time.time() - float(st.spot_armed_ts)) * 1000.0)

        if elapsed_ms > timeout_ms:
            try:
                self._spot_release(reason="timeout")
                st.status = "SPOT READY TIMEOUT"

            except Exception as e:
                st.spot_armed = False
                st.spot_armed_ts = 0.0
                st.spot_armed_brightness = 0

                if not st.light_error_latched:
                    st.light_error_latched = True

                    self._save_plc_error_event(
                        event_type="VISION_RUNTIME_ERROR",
                        error_code=21,
                        message="Light communication failed during spot timeout restore",
                        exception=e,
                    )

                self.plc.set_error(
                    code=21,
                    detail=str(e),
                )

                st.status = "LIGHT COMM ERROR: RESET REQUIRED"

    def _run_spot_inspect_once(
        self,
        frame_gray8,
        vis_bgr,
        avg5=False,
        trigger="PLC",
        allow_camera_flush: bool = True,
    ):
        st = self.state
        cfg = self._get_spot_light_cfg()

        use_spot = bool(cfg.get("enabled", False))

        inspect_frame_gray8 = frame_gray8
        inspect_vis_bgr = vis_bgr

        try:
            if use_spot:
                if not bool(getattr(st, "spot_armed", False)):
                    self._spot_prearm(
                        trigger=f"{trigger}_DIRECT",
                        flush_frames=allow_camera_flush,
                    )

                if bool(allow_camera_flush):
                    if int(cfg["inspect_settle_ms"]) > 0:
                        time.sleep(float(cfg["inspect_settle_ms"]) / 1000.0)

                    g, v = self._flush_camera_frames(int(cfg["inspect_flush_frames"]))
                    if g is not None and v is not None:
                        inspect_frame_gray8 = g
                        inspect_vis_bgr = v

            if (
                inspect_frame_gray8 is None
                or inspect_vis_bgr is None
                or not self._camera_frame_healthy(max_age_sec=0.35)
            ):
                raise CameraRecoveryInProgress(
                    "camera frame is not healthy; inspection deferred for auto recovery"
                )

            run_inspect_once(
                cam=self.cam,
                inspector=self.inspector,
                runtime_cfg=self.runtime_cfg,
                state=st,
                frame_gray8=inspect_frame_gray8,
                vis_bgr=inspect_vis_bgr,
                avg5=bool(avg5),
                use_cache=False,
                cache_every_n=1,
            )
                    
            return st.last_overall_ok

        finally:
            if use_spot and bool(cfg.get("restore_after_inspect", True)):
                self._spot_release(reason=f"{trigger}_done")

    def _get_light_failures(self) -> Dict[str, str]:
        failures = {}

        try:
            light_state = self.light.get_state() or {}
        except Exception as e:
            return {
                "_controller": f"STATE_READ_FAILED:{e}"
            }

        for light_id, info in light_state.items():
            if not isinstance(info, dict):
                continue

            if info.get("ok") is False:
                failures[str(light_id)] = str(
                    info.get("reason", "UNKNOWN")
                )

        return failures
    
    def _get_vision_state_snapshot(self) -> Dict[str, Any]:
        camera_open = False

        try:
            cap = getattr(self.cam, "cap", None)
            camera_open = bool(cap is not None and cap.isOpened())
        except Exception:
            camera_open = False

        try:
            light_state = self.light.get_state()
        except Exception as e:
            light_state = {
                "read_failed": True,
                "error": str(e),
            }

        return {
            "app_status": str(self.state.status),
            "edit_mode": bool(self.state.edit_mode),
            "quit_requested": bool(self.state.quit_requested),

            "camera_open": camera_open,
            "camera_device": str(
                self.camera_info.get("device", "")
            ),
            "camera_frame_seq": int(self._camera_frame_seq),
            "camera_last_good_frame_epoch": float(
                self._last_good_camera_frame_epoch
            ),
            "camera_last_read_ms": round(float(self._last_camera_read_ms), 3),
            "camera_max_read_ms": round(float(self._max_camera_read_ms), 3),
            "camera_slow_read_count": int(self._slow_camera_read_count),
            "loop_fps": round(float(self._loop_fps), 3),
            "process_pid": int(os.getpid()),
            "process_started_epoch": float(self._process_started_epoch),
            "process_rss_mb": round(self._get_process_rss_mb(), 2),
            "service_panel_visible": bool(self.service_panel.visible),
            "soak_test": self.soak_test.summary(),

            "light_state": light_state,

            "last_overall_ok": self.state.last_overall_ok,
            "last_result_count": len(
                self.state.last_results or {}
            ),

            "tracking_stable": bool(
                self.state.tracking_stable
            ),
            "stable_frame_count": int(
                self.state.stable_frame_count
            ),

            "spot_armed": bool(self.state.spot_armed),
            "spot_armed_brightness": int(
                self.state.spot_armed_brightness
            ),

            "test_error_active": bool(
                self.state.test_error_active
            ),
            "test_error_mode": str(
                self.state.test_error_mode
            ),
            "test_error_type": str(
                self.state.test_error_type
            ),
            "test_error_code": int(
                self.state.test_error_code
            ),
            "test_error_expected_code": int(
                self.state.test_error_expected_code
            ),
            "test_error_actual_code": int(
                self.state.test_error_actual_code
            ),
            "test_error_injected_at": str(
                self.state.test_error_injected_at
            ),
            "test_error_request_id": str(
                self.state.test_error_request_id
            ),
            "test_error_safe_logical": bool(
                self.state.test_error_safe_logical
            ),
            "test_error_phase": str(
                self.state.test_error_phase
            ),
            "test_error_result": str(
                self.state.test_error_result
            ),
            "test_error_health_detail": str(
                self.state.test_error_health_detail
            ),
            "test_error_log_path": str(
                self.state.test_error_log_path
            ),
            "test_error_pre_state": {
                "command": int(self.state.test_error_pre_command),
                "status": int(self.state.test_error_pre_status),
                "result": int(self.state.test_error_pre_result),
                "error_code": int(self.state.test_error_pre_error_code),
            },
            "test_error_reset_source": str(
                self.state.test_error_reset_source
            ),
            "test_error_recovery_elapsed_ms": int(
                self.state.test_error_recovery_elapsed_ms
            ),
        }

    def _save_plc_error_event(
        self,
        event_type: str,
        error_code: int,
        message: str,
        exception: Optional[BaseException] = None,
    ):
        try:
            plc_snapshot = self.plc.get_state_snapshot()
        except Exception as e:
            plc_snapshot = {
                "snapshot_failed": True,
                "error": str(e),
            }

        saved_path = save_plc_error_log(
            logs_root=LOGS_ROOT,
            event_type=event_type,
            error_code=error_code,
            message=message,
            plc_snapshot=plc_snapshot,
            vision_snapshot=self._get_vision_state_snapshot(),
            exception=exception,
        )

        if saved_path:
            self.state.latest_error_log_path = str(saved_path)
            try:
                self.plc.trace_application_event(
                    event="ERROR_LOG_SAVED",
                    summary=(
                        f"code={int(error_code)} "
                        f"event_type={event_type} "
                        f"path={saved_path}"
                    ),
                    extra={
                        "error_code": int(error_code),
                        "error_event_type": str(event_type),
                        "error_log_path": str(saved_path),
                    },
                )
            except Exception:
                pass

        return saved_path

    def _poll_plc_comm_events(self):
        while True:
            try:
                event = self.plc.poll_comm_event()
            except AttributeError:
                return
            except Exception as e:
                print("[PLC] communication event read failed:", e)
                return

            if event is None:
                return

            event_type = str(event.get("event_type", "fault") or "fault")
            error_code = int(event.get("error_code", 71))
            message = str(
                event.get("message", "PLC communication event")
            )
            reset_required = bool(event.get("reset_required", False))

            if event_type == "fault":
                self._plc_auto_reconnect_pending = False
                self._set_plc_vision_ready(False, "PLC_COMM_RECOVERING")
                self._save_plc_error_event(
                    event_type="PLC_COMMUNICATION_ERROR",
                    error_code=error_code,
                    message=message,
                )
                self.plc.set_recovering(
                    code=error_code,
                    detail=message,
                )
                self.state.status = (
                    "PLC COMM RECOVERY: RECONNECTING / D201=BUSY"
                )
                continue

            if event_type == "reconnected_reset_required" or reset_required:
                self._plc_auto_reconnect_pending = False
                self.plc.set_busy()
                self.state.status = (
                    "PLC COMM RESTORED: D200=3 RESET REQUIRED"
                )
                continue

            if event_type == "reconnected_existing_error":
                self._plc_auto_reconnect_pending = False
                self.state.status = (
                    "PLC COMM RESTORED: EXISTING VISION ERROR REMAINS"
                )
                continue

            if event_type in ("reconnected", "link_verified"):
                try:
                    current_snapshot = self.plc.get_state_snapshot()
                    current_status = int(current_snapshot.get("status", 0))
                    current_error_code = int(
                        current_snapshot.get("error_code", 0)
                    )
                except Exception:
                    current_status = 0
                    current_error_code = 0

                if (
                    current_status == 3
                    and current_error_code not in (0, 70, 71)
                ):
                    self._plc_auto_reconnect_pending = False
                    self.state.status = (
                        "PLC COMM RESTORED: EXISTING VISION ERROR REMAINS"
                    )
                    continue

                self._plc_auto_reconnect_pending = True
                self.plc.set_busy()
                self.state.status = (
                    "PLC COMM RESTORED: VISION HEALTH CHECK"
                )

    def _complete_auto_plc_reconnect_if_ready(self):
        if not self._plc_auto_reconnect_pending:
            return

        st = self.state
        try:
            snapshot = self.plc.get_state_snapshot()
        except Exception:
            return

        if not bool(snapshot.get("serial_open", False)):
            return
        if not bool(snapshot.get("link_verified", False)):
            return
        if bool(snapshot.get("comm_reset_required", False)):
            self._plc_auto_reconnect_pending = False
            return
        if st.camera_error_latched or st.light_error_latched:
            return
        if st.plc_shutdown_started or self._camera_is_recovering():
            return
        if not self._camera_frame_healthy(max_age_sec=1.0):
            return

        try:
            if not self.plc.complete_comm_recovery():
                return
            self.plc.reset_to_ready(reset_heartbeat=False)
            self._plc_auto_reconnect_pending = False
            st.status = "PLC COMM AUTO RECOVERY: READY"
            self.plc.trace_application_event(
                event="PLC_COMM_AUTO_RECOVERY_COMPLETED",
                summary="serial reconnected and vision health verified",
            )
        except Exception as e:
            print("[PLC] automatic communication recovery completion failed:", e)

    def _get_plc_error_test_cfg(self) -> Dict[str, Any]:
        cfg = self.plc_cfg.get("error_test", {}) or {}

        request_path = str(
            cfg.get(
                "request_path",
                "data/runtime/plc_error_test_request.json",
            )
        ).strip()

        if not os.path.isabs(request_path):
            request_path = os.path.join(PROJECT_ROOT, request_path)

        allowed_types = {
            str(item).strip().lower()
            for item in (
                cfg.get(
                    "allowed_types",
                    ["camera", "light", "inspection", "plc_comm"],
                )
                or []
            )
            if str(item).strip()
        }
        allowed_modes = {
            str(item).strip().lower()
            for item in (cfg.get("allowed_modes", ["logic", "recovery"]) or [])
            if str(item).strip()
        }

        return {
            "enabled": bool(cfg.get("enabled", False)),
            "request_path": os.path.abspath(request_path),
            "delete_after_read": bool(cfg.get("delete_after_read", True)),
            "allowed_types": allowed_types,
            "allowed_modes": allowed_modes,
            "camera_health_frames": max(1, int(cfg.get("camera_health_frames", 3))),
            "camera_health_timeout_sec": max(
                0.3,
                float(cfg.get("camera_health_timeout_sec", 1.5)),
            ),
            # Hard-fault injection is intentionally unsupported from the UI.
            "allow_hardware_fault_tests": False,
        }

    def _inject_plc_test_error(
        self,
        error_type: str,
        request_id: str,
        custom_message: str = "",
        test_mode: str = "logic",
    ) -> bool:
        st = self.state
        error_type = str(error_type or "").strip().lower()
        test_mode = str(test_mode or "logic").strip().lower()
        test_cfg = self._get_plc_error_test_cfg()
        st.test_error_block_reason = ""

        if not bool(test_cfg.get("enabled", False)):
            st.test_error_block_reason = "TEST BLOCKED: ERROR TEST IS DISABLED"
            print(f"[PLC TEST] {st.test_error_block_reason}")
            return False
        if test_mode not in test_cfg.get("allowed_modes", set()):
            st.test_error_block_reason = f"TEST BLOCKED: MODE {test_mode} NOT ALLOWED"
            print(f"[PLC TEST] {st.test_error_block_reason}")
            return False
        if error_type not in test_cfg.get("allowed_types", set()):
            st.test_error_block_reason = f"TEST BLOCKED: TYPE {error_type} NOT ALLOWED"
            print(f"[PLC TEST] {st.test_error_block_reason}")
            return False

        definitions = {
            "camera": (11, "CAMERA_FRAME_TIMEOUT"),
            "light": (21, "LIGHT_COMMUNICATION_FAILED"),
            "inspection": (40, "INSPECTION_FAILED"),
            "plc_comm": (71, "PLC_SERIAL_COMMUNICATION_LOST"),
        }
        definition = definitions.get(error_type)
        if definition is None:
            st.test_error_block_reason = f"TEST BLOCKED: UNSUPPORTED TYPE {error_type}"
            return False

        try:
            snapshot = self.plc.get_state_snapshot()
        except Exception as e:
            st.test_error_block_reason = f"TEST BLOCKED: PLC SNAPSHOT FAILED ({e})"
            return False

        command = int(snapshot.get("command", 0))
        status = int(snapshot.get("status", 0))
        active_error_code = int(snapshot.get("error_code", 0))
        serial_open = bool(snapshot.get("serial_open", False))
        comm_fault = bool(snapshot.get("comm_fault_active", False))

        blocked = ""
        if st.edit_mode:
            blocked = "TEST BLOCKED: SERVICE TEST REQUIRES RUN MODE"
        elif self.soak_test.active:
            blocked = "TEST BLOCKED: OVERNIGHT SOAK TEST IS ACTIVE"
        elif st.test_error_active:
            blocked = "TEST BLOCKED: ANOTHER TEST IS ACTIVE"
        elif status == 8 or command in (8, 9):
            blocked = "TEST BLOCKED: SHUTDOWN/EMERGENCY IS ACTIVE"
        elif status == 3 or active_error_code != 0:
            blocked = (
                f"TEST BLOCKED: REAL ERROR IS ACTIVE "
                f"(STATUS={status}, ERROR={active_error_code})"
            )
        elif not serial_open or comm_fault:
            blocked = "TEST BLOCKED: PLC LINK IS NOT HEALTHY"
        elif st.camera_error_latched or st.light_error_latched:
            blocked = "TEST BLOCKED: REAL VISION FAULT IS ACTIVE"

        if blocked:
            st.test_error_block_reason = blocked
            print(f"[PLC TEST] {blocked}")
            return False

        error_code, error_name = definition
        message = str(custom_message or error_name)
        injected_at = time.strftime("%Y-%m-%d %H:%M:%S")

        st.test_error_active = True
        st.test_error_mode = test_mode
        st.test_error_type = error_type
        st.test_error_code = int(error_code)
        st.test_error_expected_code = int(error_code)
        st.test_error_actual_code = int(error_code)
        st.test_error_injected_at = injected_at
        st.test_error_request_id = str(request_id)
        st.test_error_safe_logical = True
        st.test_error_phase = "WAITING_RESET"
        st.test_error_result = ""
        st.test_error_block_reason = ""
        st.test_error_log_path = ""
        st.test_error_log_saved = False
        st.test_error_recovered_at = ""
        st.test_error_pre_command = command
        st.test_error_pre_status = status
        st.test_error_pre_result = int(snapshot.get("result", 0))
        st.test_error_pre_error_code = active_error_code
        st.test_error_reset_source = ""
        st.test_error_started_monotonic = time.monotonic()
        st.test_error_recovery_elapsed_ms = 0

        if command == 3:
            st.test_error_health_detail = (
                "FAULT DETECTED - D200 ALREADY 3; "
                "USE LOCAL RECOVER OR RE-ISSUE RESET AFTER COMMAND CHANGE"
            )
        else:
            st.test_error_health_detail = (
                "FAULT DETECTED - WAIT PLC D200=3 OR LOCAL RECOVER"
            )

        # SAFE RECOVERY TEST sets only internal latches. It never disconnects
        # camera, light or PLC hardware from the service UI.
        if test_mode == "recovery":
            if error_type == "camera":
                st.camera_error_latched = True
            elif error_type == "light":
                st.light_error_latched = True

        detail = (
            f"[TEST:{test_mode.upper()}:{error_type}] {message} "
            f"request_id={request_id}"
        )
        self.plc.set_error(code=error_code, detail=detail)

        event_type = f"PLC_TEST_{test_mode.upper()}_{error_type.upper()}_ERROR"
        saved_path = self._save_plc_error_event(
            event_type=event_type,
            error_code=error_code,
            message=detail,
            exception=RuntimeError(detail),
        )
        st.latest_error_log_path = str(saved_path or "")
        self._save_plc_test_event(
            phase="FAULT_DETECTED",
            result="",
            message=detail,
            extra={
                "test_mode": test_mode,
                "safe_recovery": test_mode == "recovery",
                "hardware_disconnected": False,
                "error_log_path": str(saved_path or ""),
                "pre_state": {
                    "command": command,
                    "status": status,
                    "result": int(snapshot.get("result", 0)),
                    "error_code": active_error_code,
                },
            },
        )

        try:
            self.plc.trace_application_event(
                event="TEST_ERROR_INJECTED",
                summary=(
                    f"mode={test_mode} type={error_type} code={error_code} "
                    f"request_id={request_id}"
                ),
                extra={
                    "test_mode": test_mode,
                    "test_error_type": error_type,
                    "test_error_code": error_code,
                    "test_request_id": request_id,
                    "error_log_path": str(saved_path or ""),
                },
            )
        except Exception:
            pass

        st.status = (
            f"{test_mode.upper()} TEST ERROR {error_code} "
            f"{error_type.upper()}: D200=3 RESET REQUIRED"
        )
        print(
            f"[PLC TEST] injected mode={test_mode} type={error_type} "
            f"code={error_code} log={saved_path}"
        )
        return True

    def _poll_plc_error_test_request(self):
        cfg = self._get_plc_error_test_cfg()
        if not cfg["enabled"]:
            return

        request_path = cfg["request_path"]
        if not os.path.exists(request_path):
            return

        request = None
        read_error = None

        try:
            with open(
                request_path,
                "r",
                encoding="utf-8-sig",
            ) as file:
                request = json.load(file)
        except Exception as e:
            read_error = e
        finally:
            if cfg["delete_after_read"]:
                try:
                    os.remove(request_path)
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(
                        f"[PLC TEST] request cleanup failed: {e}"
                    )

        if read_error is not None:
            print(
                f"[PLC TEST] invalid request file: {read_error}"
            )
            return

        if not isinstance(request, dict):
            print("[PLC TEST] request must be a JSON object")
            return

        request_id = str(
            request.get(
                "request_id",
                f"req-{time.time_ns()}",
            )
        )
        if request_id == self._plc_test_last_request_id:
            return

        self._plc_test_last_request_id = request_id

        error_type = str(
            request.get("type", "")
        ).strip().lower()

        if error_type not in cfg["allowed_types"]:
            print(
                f"[PLC TEST] type not allowed: {error_type}"
            )
            return

        self._inject_plc_test_error(
            error_type=error_type,
            request_id=request_id,
            custom_message=str(request.get("message", "") or ""),
            test_mode=str(request.get("mode", "logic") or "logic"),
        )

    def _clear_spot_runtime_state(self):
        st = self.state
        st.spot_armed = False
        st.spot_armed_ts = 0.0
        st.spot_armed_brightness = 0

    def _clear_vision_runtime_state(self):
        st = self.state
        errors = []

        st.last_results = None
        st.last_overall_ok = None
        if hasattr(st, "last_overall_info"):
            st.last_overall_info = None

        st.pose_bad_cnt = 0
        st.tracking_stable = False
        st.stable_frame_count = 0
        st.last_auto_inspect_ts = 0.0

        self._clear_spot_runtime_state()

        try:
            self.inspector.reset_temporal_filters()
        except Exception as e:
            errors.append(f"mean filters: {e}")

        try:
            self.stabilizer.reset()
            st._trk_frame_idx = 0
            st._trk_cache = None
            st._trk_smoothed_cache = None
            st._trk_motion_stable = False
        except Exception as e:
            errors.append(f"stabilizer: {e}")

        try:
            aligner = getattr(self.inspector, "aligner", None)
            if aligner is not None:
                aligner.reset_templates()
            self._load_alignment_template()
        except Exception as e:
            errors.append(f"tracker: {e}")

        if errors:
            raise RuntimeError("; ".join(errors))

    def _get_plc_recovery_cfg(self):
        cfg = self.plc_cfg.get("recovery", {}) or {}
        return {
            "camera_open_timeout_sec": max(
                0.5,
                float(cfg.get("camera_open_timeout_sec", 2.0)),
            ),
            "camera_retry_interval_sec": max(
                0.02,
                float(cfg.get("camera_retry_interval_sec", 0.1)),
            ),
            "camera_grace_sec": max(
                0.0,
                float(cfg.get("camera_grace_sec", 2.0)),
            ),
            "frame_timeout_sec": max(
                0.1,
                float(cfg.get("frame_timeout_sec", 1.0)),
            ),
        }

    def _restart_camera_checked(self):
        recovery_cfg = self._get_plc_recovery_cfg()

        recover_blocking = getattr(self.cam, "recover_blocking", None)
        if callable(recover_blocking):
            recovered_frame = recover_blocking(
                reason="PLC reset camera recovery",
                timeout_sec=max(20.0, recovery_cfg["camera_open_timeout_sec"] + 15.0),
            )
        else:
            self.cam.release()
            time.sleep(0.2)
            self.cam.open()

            recovered_frame = None
            recovery_deadline = (
                time.time()
                + recovery_cfg["camera_open_timeout_sec"]
            )

            while time.time() < recovery_deadline:
                recovered_frame = self.cam.read()
                if recovered_frame is not None:
                    break
                time.sleep(recovery_cfg["camera_retry_interval_sec"])

            if recovered_frame is None:
                raise RuntimeError("camera reopened but no frame was received")

        self.state.camera_read_blocked = False
        self.state.camera_error_grace_until = (
            time.time()
            + recovery_cfg["camera_grace_sec"]
        )
        return recovered_frame

    def _restart_light_checked(self):
        self.light.start()

        light_state = self.light.get_state() or {}
        self.runtime_cfg["_light_state"] = light_state

        failed_lights = {}
        for light_id, info in light_state.items():
            if not isinstance(info, dict):
                continue
            if info.get("ok") is False:
                failed_lights[str(light_id)] = str(
                    info.get("reason", "UNKNOWN")
                )

        if failed_lights:
            raise LightCommunicationError(
                f"light communication failed: {failed_lights}"
            )

        return light_state

    def _handle_plc_shutdown(self):
        st = self.state

        if bool(getattr(st, "plc_shutdown_started", False)):
            return

        st.plc_shutdown_started = True
        st.status = "PLC SHUTDOWN"
        self.plc.set_shutdown()  # D201=8 유지

        try:
            self._spot_release(reason="plc_shutdown")
        except Exception as e:
            print("[PLC] shutdown spot release failed:", e)

        try:
            self.light.stop()
        except Exception as e:
            print("[PLC] light stop before shutdown failed:", e)

        shutdown_cfg = self.plc_cfg.get("shutdown", {}) or {}
        hold_sec = float(shutdown_cfg.get("ready_hold_sec", 2.0))
        command = str(
            shutdown_cfg.get(
                "command",
                "sudo /sbin/shutdown -h now",
            )
        )

        if hold_sec > 0:
            time.sleep(hold_sec)

        try:
            shutdown_process = subprocess.Popen(command, shell=True)
            time.sleep(0.2)

            return_code = shutdown_process.poll()
            if return_code not in (None, 0):
                raise RuntimeError(
                    f"shutdown command exited with code {return_code}"
                )

            st.status = "PLC SHUTDOWN COMMAND SENT"
            st.quit_requested = True

        except Exception as e:
            print("[PLC] shutdown command failed:", e)

            self._save_plc_error_event(
                event_type="VISION_ERROR",
                error_code=90,
                message="Jetson shutdown command failed",
                exception=e,
            )

            self.plc.set_error(
                code=90,
                detail=str(e),
            )

            st.status = f"PLC SHUTDOWN ERROR: {e}"
            st.plc_shutdown_started = False

    def _handle_plc_emergency(self):
        st = self.state

        # 초기화 전에 현재 상태 저장
        # heartbeat, 내부 오류 코드, 마지막 검사 시각/소요시간 포함
        self._save_plc_error_event(
            event_type="EMERGENCY_STOP",
            error_code=51,
            message="D200=9 emergency stop received",
        )

        reset_ok = self._handle_plc_reset(
            reset_heartbeat=True,
        )

        if not reset_ok:
            # 복구 실패 원인은 _handle_plc_reset()에서 이미 로그 저장
            # D201=3 유지
            st.status = "PLC EMERGENCY: VISION ERROR"
            return

        # 정상 복구
        # D201=0, D202=0, D203=0
        st.status = "PLC EMERGENCY RESET: READY"

    def _verify_camera_stream_checked(self) -> str:
        # The service test must never become a second camera consumer.
        # Health is verified from the frame already received by the main loop.
        cfg = self._get_plc_error_test_cfg()
        max_age = float(cfg.get("camera_health_timeout_sec", 1.5))
        if self._last_good_camera_frame_epoch <= 0 or self._camera_frame_seq <= 0:
            raise RuntimeError("camera health check failed: no main-loop frame")
        age = max(0.0, time.time() - self._last_good_camera_frame_epoch)
        if age > max_age:
            raise RuntimeError(
                f"camera health check failed: last frame age={age:.3f}s"
            )
        return (
            f"CAMERA HEALTH OK: MAIN FRAME#{self._camera_frame_seq} "
            f"AGE={age:.3f}s"
        )

    def _verify_light_recovery_checked(self) -> str:
        light_state = self._restart_light_checked() or {}
        cfg = self._get_spot_light_cfg()
        light_id = str(cfg.get("light_id", "light1"))
        idle = int(cfg.get("idle_brightness", 20))
        info = light_state.get(light_id, {}) if isinstance(light_state, dict) else {}
        brightness = int(info.get("brightness", -1)) if isinstance(info, dict) else -1
        ok = info.get("ok") if isinstance(info, dict) else None
        if ok is False or brightness != idle:
            raise LightCommunicationError(
                f"light health check failed: id={light_id}, brightness={brightness}, idle={idle}, ok={ok}"
            )
        return f"LIGHT HEALTH OK: {light_id}={brightness}%"

    def _verify_plc_comm_health_checked(self) -> str:
        snapshot = self.plc.get_state_snapshot()
        if not bool(snapshot.get("serial_open", False)):
            raise RuntimeError("PLC serial is not open")
        if not bool(snapshot.get("link_verified", False)):
            raise RuntimeError("PLC link has not received a valid Modbus request")
        if bool(snapshot.get("comm_fault_active", False)):
            raise RuntimeError("PLC communication fault is still active")
        return (
            f"PLC COMM HEALTH OK: RX#{int(snapshot.get('rx_count', 0))} "
            f"TX#{int(snapshot.get('tx_count', 0))}"
        )

    def _complete_plc_test_recovery(
        self,
        reset_heartbeat: bool,
        health_detail: str,
    ) -> bool:
        st = self.state
        recovered_mode = str(st.test_error_mode)
        recovered_type = str(st.test_error_type)
        recovered_code = int(st.test_error_code)
        recovered_id = str(st.test_error_request_id)

        self.plc.reset_to_ready(reset_heartbeat=reset_heartbeat)
        st.test_error_active = False
        st.test_error_phase = "RECOVERED"
        st.test_error_result = "PASS"
        st.test_error_health_detail = str(health_detail)
        st.test_error_recovered_at = time.strftime("%Y-%m-%d %H:%M:%S")
        st.test_error_recovery_elapsed_ms = max(
            0,
            int((time.monotonic() - st.test_error_started_monotonic) * 1000.0),
        ) if st.test_error_started_monotonic > 0 else 0
        st.test_error_block_reason = "READY FOR NEXT TEST"

        self._save_plc_test_event(
            phase="RECOVERED",
            result="PASS",
            message=(
                f"D200=3 recovery completed mode={recovered_mode} "
                f"type={recovered_type} code={recovered_code} request_id={recovered_id}"
            ),
            extra={
                "test_mode": recovered_mode,
                "reset_heartbeat": bool(reset_heartbeat),
                "health_detail": str(health_detail),
                "hardware_disconnected": False,
                "reset_source": str(st.test_error_reset_source),
                "recovery_elapsed_ms": int(st.test_error_recovery_elapsed_ms),
                "pre_state": {
                    "command": int(st.test_error_pre_command),
                    "status": int(st.test_error_pre_status),
                    "result": int(st.test_error_pre_result),
                    "error_code": int(st.test_error_pre_error_code),
                },
            },
        )
        try:
            self.plc.trace_application_event(
                event="TEST_ERROR_RECOVERED",
                summary=(
                    f"mode={recovered_mode} type={recovered_type} "
                    f"code={recovered_code} health={health_detail}"
                ),
                extra={
                    "test_mode": recovered_mode,
                    "test_error_type": recovered_type,
                    "test_error_code": recovered_code,
                    "recovery_ok": True,
                    "health_detail": str(health_detail),
                },
            )
        except Exception:
            pass

        st.status = f"PLC TEST PASS: {recovered_mode.upper()} / {recovered_type.upper()}"
        return True

    def _fail_plc_test_recovery(self, message: str, exception: BaseException) -> bool:
        st = self.state
        code = int(st.test_error_code or 60)
        st.test_error_phase = "RECOVERY_FAILED"
        st.test_error_result = "FAIL"
        st.test_error_health_detail = str(message)
        self._save_plc_test_event(
            phase="RECOVERY_FAILED",
            result="FAIL",
            message=str(message),
            extra={
                "test_mode": str(st.test_error_mode),
                "exception_type": type(exception).__name__,
                "exception": str(exception),
            },
        )
        self._save_plc_error_event(
            event_type="PLC_TEST_RECOVERY_FAILED",
            error_code=code,
            message=str(message),
            exception=exception,
        )
        self.plc.set_error(code=code, detail=str(message))
        st.status = f"PLC TEST RECOVERY FAILED: {st.test_error_type.upper()}"
        return False

    def _handle_plc_test_reset(
        self,
        reset_heartbeat: bool = False,
        source: str = "PLC_D200",
    ) -> bool:
        st = self.state
        mode = str(st.test_error_mode or "logic").lower()
        error_type = str(st.test_error_type or "").lower()
        st.test_error_phase = "RECOVERING"
        st.test_error_result = ""
        st.test_error_reset_source = str(source or "UNKNOWN")
        st.test_error_health_detail = (
            f"RESET RECEIVED ({st.test_error_reset_source}) - RECOVERY IN PROGRESS"
        )

        self.plc.prepare_reset(reset_heartbeat=reset_heartbeat)
        self.plc.set_busy()
        self._save_plc_test_event(
            phase="RECOVERING",
            result="",
            message=(
                f"reset received source={st.test_error_reset_source} "
                f"mode={mode} type={error_type}"
            ),
            extra={
                "test_mode": mode,
                "reset_source": st.test_error_reset_source,
            },
        )

        try:
            if mode == "logic":
                if (
                    st.camera_error_latched
                    or st.light_error_latched
                    or bool(self.plc.get_state_snapshot().get("comm_fault_active", False))
                ):
                    raise RuntimeError(
                        "real hardware fault detected during PLC logic test"
                    )

                # Test-only safe cleanup. This mirrors the non-hardware part of
                # the normal reset path without changing production PLC logic.
                if bool(st.spot_armed):
                    self._spot_release(reason="plc_logic_test_reset")
                self._clear_vision_runtime_state()
                return self._complete_plc_test_recovery(
                    reset_heartbeat,
                    "PLC LOGIC RESET OK: STATUS/RESULT/RUNTIME CLEARED",
                )

            if mode != "recovery":
                raise RuntimeError(f"unsupported test mode: {mode}")

            # Common test-only runtime recovery path. Hardware is never
            # disconnected by the service panel.
            if bool(st.spot_armed):
                self._spot_release(reason="plc_recovery_test_reset")
            self._clear_vision_runtime_state()

            if error_type == "inspection":
                if st.last_results is not None or st.last_overall_ok is not None:
                    raise RuntimeError("inspection runtime state was not cleared")
                health = "INSPECTION RECOVERY OK: RUNTIME STATE CLEARED"

            elif error_type == "light":
                health = self._verify_light_recovery_checked()
                st.light_error_latched = False

            elif error_type == "camera":
                # Safe recovery test: keep Argus pipeline open and verify frames.
                # No release/open and no physical camera disconnection are performed.
                health = self._verify_camera_stream_checked()
                st.camera_error_latched = False

            elif error_type == "plc_comm":
                complete_comm_recovery = getattr(
                    self.plc, "complete_comm_recovery", None
                )
                if callable(complete_comm_recovery):
                    complete_comm_recovery()
                health = self._verify_plc_comm_health_checked()

            else:
                raise RuntimeError(f"unsupported recovery test type: {error_type}")

            return self._complete_plc_test_recovery(reset_heartbeat, health)

        except Exception as e:
            return self._fail_plc_test_recovery(
                f"{mode.upper()} / {error_type.upper()} recovery failed: {e}",
                e,
            )

    def _handle_plc_reset(
        self,
        reset_heartbeat: bool = False,
    ):
        st = self.state

        if st.test_error_active:
            return self._handle_plc_test_reset(
                reset_heartbeat=reset_heartbeat,
                source="PLC_D200",
            )

        try:
            before_state = self.plc.get_state_snapshot()
            current_status = int(before_state.get("status", 0))
            current_error_code = int(before_state.get("error_code", 0))
        except Exception:
            current_status = 0
            current_error_code = 0

        camera_error_codes = {10, 11}
        light_error_codes = {20, 21}
        plc_comm_error_codes = {70, 71}

        need_camera_restart = (
            st.camera_error_latched
            or current_error_code in camera_error_codes
        )
        need_light_restart = (
            st.light_error_latched
            or current_error_code in light_error_codes
        )


        # 이전 Reset 자체가 실패한 경우에는 하드웨어 전체 복구를 다시 시도한다.
        if current_status == 3 and current_error_code == 60:
            need_camera_restart = True
            need_light_restart = True

        # PLC 통신 오류는 비전 하드웨어 재시작 대상이 아니다.
        if current_error_code in plc_comm_error_codes:
            need_camera_restart = bool(st.camera_error_latched)
            need_light_restart = bool(st.light_error_latched)

        # D200=3 입력 즉시 D202를 0으로 초기화한다.
        # D200=9에서는 D203도 0으로 초기화한다.
        self.plc.prepare_reset(
            reset_heartbeat=reset_heartbeat
        )
        self.plc.set_busy()
        st.status = "PLC RESET RECOVERY: BUSY"

        # 조명 오류가 이미 발생한 상태에서는 통신 명령을 먼저 보내지 않는다.
        # 조명 재시작 자체가 idle 밝기 복구를 수행한다.
        if need_light_restart:
            self._clear_spot_runtime_state()
        else:
            try:
                self._spot_release(reason="plc_reset")
            except Exception as e:
                need_light_restart = True
                st.light_error_latched = True
                self._clear_spot_runtime_state()

                self._save_plc_error_event(
                    event_type="VISION_RESET_RECOVERY",
                    error_code=21,
                    message=(
                        "Light idle restore failed during PLC reset; "
                        "light controller restart will be attempted"
                    ),
                    exception=e,
                )

        try:
            self._clear_vision_runtime_state()
        except Exception as e:
            self._save_plc_error_event(
                event_type="VISION_RESET_ERROR",
                error_code=60,
                message="Vision runtime state reset failed",
                exception=e,
            )
            self.plc.set_error(code=60, detail=str(e))
            st.status = "PLC RESET ERROR: RUNTIME"
            return False

        if need_camera_restart:
            try:
                self._restart_camera_checked()
                st.camera_error_latched = False
                st.camera_read_blocked = False
            except Exception as e:
                st.camera_error_latched = True

                self._save_plc_error_event(
                    event_type="VISION_RESET_ERROR",
                    error_code=10,
                    message="Camera restart failed during PLC reset",
                    exception=e,
                )
                self.plc.set_error(code=10, detail=str(e))
                st.status = "PLC RESET ERROR: CAMERA"
                return False

        if need_light_restart:
            try:
                self._restart_light_checked()
                st.light_error_latched = False
            except Exception as e:
                st.light_error_latched = True

                self._save_plc_error_event(
                    event_type="VISION_RESET_ERROR",
                    error_code=21,
                    message="Light restart failed during PLC reset",
                    exception=e,
                )
                self.plc.set_error(code=21, detail=str(e))
                st.status = "PLC RESET ERROR: LIGHT"
                return False

        st.camera_error_latched = False
        st.camera_read_blocked = False
        st.light_error_latched = False

        complete_comm_recovery = getattr(
            self.plc, "complete_comm_recovery", None
        )
        if callable(complete_comm_recovery):
            try:
                snapshot = self.plc.get_state_snapshot()
                if bool(snapshot.get("comm_fault_active", False)):
                    if not complete_comm_recovery():
                        raise RuntimeError(
                            "PLC communication is not connected during reset"
                        )
            except Exception as e:
                self._save_plc_error_event(
                    event_type="VISION_RESET_ERROR",
                    error_code=71,
                    message="PLC communication recovery failed during D200=3 reset",
                    exception=e,
                )
                self.plc.set_error(code=71, detail=str(e))
                st.status = "PLC RESET ERROR: COMMUNICATION"
                return False

        self._plc_auto_reconnect_pending = False
        self.plc.reset_to_ready(
            reset_heartbeat=reset_heartbeat
        )

        st.status = "PLC RESET: READY"
        return True

    def _log_plc_command_sequence_error(
        self,
        command_name: str,
        current_status: int,
        reason: str,
    ):
        status_names = {
            0: "READY",
            1: "BUSY",
            2: "DONE",
            3: "VISION_ERROR",
            8: "SHUTDOWN",
        }

        message = (
            f"PLC command sequence rejected: "
            f"command={command_name}, "
            f"status={current_status}"
            f"({status_names.get(current_status, 'UNKNOWN')}), "
            f"reason={reason}"
        )

        self._save_plc_error_event(
            event_type="PLC_COMMAND_SEQUENCE_ERROR",
            error_code=50,
            message=message,
        )

        print(f"[PLC] {message}")

    def _run_plc_inspect_tick(
        self,
        frame_gray8,
        vis_bgr,
        forced_cmd: Optional[str] = None,
    ):
        st = self.state
        is_camera_retry = forced_cmd is not None

        cmd = forced_cmd if is_camera_retry else self.plc.poll_command()
        if cmd is None:
            return

        if cmd in ("prepare", "inspect"):
            reason = (
                "D200_PREPARE_RECEIVED"
                if cmd == "prepare"
                else "D200_INSPECT_RECEIVED"
            )
            self._consume_plc_vision_ready(reason)

        if cmd == "reset":
            self._handle_plc_reset()
            return

        if cmd == "shutdown":
            self._handle_plc_shutdown()
            return

        if cmd == "emergency":
            self._handle_plc_emergency()
            return

        try:
            plc_snapshot = self.plc.get_state_snapshot()
            current_status = int(
                plc_snapshot.get("status", 0)
            )
        except Exception:
            current_status = 0

        # 종료가 시작된 상태에서는 다른 명령을 처리하지 않는다.
        if current_status == 8:
            self._log_plc_command_sequence_error(
                command_name=str(cmd),
                current_status=current_status,
                reason="vision shutdown is already in progress",
            )
            return

        # 검사 결과가 남아 있는 동안에는 D200=3만 정상 초기화 명령이다.
        if current_status == 2 and cmd in ("prepare", "inspect"):
            self._log_plc_command_sequence_error(
                command_name=str(cmd),
                current_status=current_status,
                reason="previous inspection result has not been reset by D200=3",
            )

            # D201=2, D202=OK/NG 그대로 유지
            st.status = "PLC COMMAND REJECTED: RESULT RESET REQUIRED"
            return

        # 비전 오류 상태에서는 Reset, 종료, 비상정지만 허용한다.
        if current_status == 3 and cmd in ("prepare", "inspect"):
            self._log_plc_command_sequence_error(
                command_name=str(cmd),
                current_status=current_status,
                reason="vision error must be recovered by D200=3",
            )

            # D201=3 유지
            st.status = "PLC COMMAND REJECTED: VISION RESET REQUIRED"
            return

        # Busy 중 중복 명령 차단
        if (
            current_status == 1
            and cmd in ("prepare", "inspect")
            and not is_camera_retry
        ):
            self._log_plc_command_sequence_error(
                command_name=str(cmd),
                current_status=current_status,
                reason="vision is already processing another command",
            )

            # D201=1 유지
            st.status = "PLC COMMAND REJECTED: VISION BUSY"
            return
        
        if st.camera_error_latched or st.light_error_latched:
            st.status = "PLC COMMAND REJECTED: VISION ERROR"
            return
        
        if cmd == "command_error":
            try:
                snapshot = self.plc.get_state_snapshot()
                command_value = int(
                    snapshot.get("command", 0)
                )
                current_status = int(
                    snapshot.get("status", 0)
                )
            except Exception:
                command_value = -1
                current_status = 0

            self._log_plc_command_sequence_error(
                command_name=f"UNKNOWN_D200_{command_value}",
                current_status=current_status,
                reason="unsupported D200 command value",
            )

            # PLC 측 잘못된 값이므로 D201=3으로 변경하지 않는다.
            st.status = (
                f"PLC UNKNOWN COMMAND: D200={command_value}"
            )
            return

        if cmd == "prepare":
            if st.edit_mode:
                self._log_plc_command_sequence_error(
                    command_name="prepare",
                    current_status=current_status,
                    reason="vision is in EDIT mode",
                )
                st.status = "PLC PREPARE REJECTED: EDIT MODE"
                return

            if frame_gray8 is None or vis_bgr is None:
                if self._camera_is_recovering():
                    self._pending_plc_camera_command = "prepare"
                    self.plc.set_busy()
                    st.status = "PLC PREPARE PAUSED: CAMERA AUTO RECOVERY"
                    return

                error = RuntimeError("camera frame is unavailable")
                st.camera_error_latched = True
                self._save_plc_error_event(
                    event_type="VISION_RUNTIME_ERROR",
                    error_code=11,
                    message="D200=1 received but camera frame is unavailable",
                    exception=error,
                )
                self.plc.set_error(code=11, detail=str(error))
                st.status = "PLC PREPARE ERROR: NO CAMERA FRAME"
                return

            try:
                self.plc.set_busy()
                st.status = "PLC PREPARE BUSY"

                armed = self._spot_prearm(trigger="PLC")
                if not self._camera_frame_healthy(max_age_sec=0.35):
                    raise CameraRecoveryInProgress(
                        "camera became unavailable during PLC prepare"
                    )

                # 준비 완료 후 Ready
                # D202 값은 변경하지 않음
                self.plc.set_idle()
                self._pending_plc_camera_command = None

                st.status = (
                    "PLC READY: SPOT ARMED"
                    if armed
                    else "PLC READY"
                )

            except CameraRecoveryInProgress as e:
                self._pending_plc_camera_command = "prepare"
                self.plc.set_busy()
                st.status = "PLC PREPARE PAUSED: CAMERA AUTO RECOVERY"
                print("[PLC] prepare deferred:", e)

            except LightCommunicationError as e:
                st.light_error_latched = True

                self._save_plc_error_event(
                    event_type="VISION_RUNTIME_ERROR",
                    error_code=21,
                    message="Light communication failed during PLC prepare",
                    exception=e,
                )

                self.plc.set_error(
                    code=21,
                    detail=str(e),
                )

                st.status = "PLC PREPARE LIGHT ERROR: RESET REQUIRED"

            except Exception as e:
                self._save_plc_error_event(
                    event_type="VISION_RUNTIME_ERROR",
                    error_code=40,
                    message="PLC inspection preparation failed",
                    exception=e,
                )

                self.plc.set_error(
                    code=40,
                    detail=str(e),
                )

                st.status = "PLC PREPARE ERROR: RESET REQUIRED"

            return

        if cmd != "inspect":
            return

        if st.edit_mode:
            self._log_plc_command_sequence_error(
                command_name="inspect",
                current_status=current_status,
                reason="vision is in EDIT mode",
            )

            st.status = "PLC INSPECT REJECTED: EDIT MODE"
            return

        if frame_gray8 is None or vis_bgr is None:
            if self._camera_is_recovering():
                self._pending_plc_camera_command = "inspect"
                self.plc.set_busy()
                st.status = "PLC INSPECT PAUSED: CAMERA AUTO RECOVERY"
                return

            error = RuntimeError("camera frame is unavailable")
            st.camera_error_latched = True
            self._save_plc_error_event(
                event_type="VISION_RUNTIME_ERROR",
                error_code=11,
                message="D200=2 received but camera frame is unavailable",
                exception=error,
            )
            self.plc.set_error(code=11, detail=str(error))
            st.status = "PLC INSPECT ERROR: NO CAMERA FRAME"
            return

        try:
            self.plc.set_busy()
            st.status = "PLC INSPECT BUSY"

            # 이전 내부 검사 결과 제거
            # PLC D202 값은 변경하지 않음
            st.last_overall_ok = None
            st.last_results = None

            started_at = time.perf_counter()

            overall_ok = self._run_spot_inspect_once(
                frame_gray8,
                vis_bgr,
                avg5=bool(
                    self.runtime_cfg.get(
                        "plc_inspect_avg5",
                        False,
                    )
                ),
                trigger="PLC",
            )

            elapsed_ms = int(
                (time.perf_counter() - started_at) * 1000.0
            )

            results = st.last_results

            if overall_ok is None:
                raise InspectionResultError(
                    "inspection overall result was not generated"
                )

            if not isinstance(results, dict) or not results:
                raise InspectionResultError(
                    "inspection ROI results were not generated"
                )

            invalid_roi_ids = []

            for roi_id, result in results.items():
                if isinstance(result, dict):
                    roi_ok = result.get("ok")
                else:
                    roi_ok = getattr(result, "ok", None)

                if roi_ok is None:
                    invalid_roi_ids.append(str(roi_id))

            if invalid_roi_ids:
                raise InspectionResultError(
                    "ROI result has no OK/NG value: "
                    + ",".join(invalid_roi_ids)
                )

            self.plc.set_done(
                ok=bool(overall_ok),
                elapsed_ms=elapsed_ms,
            )
            self._pending_plc_camera_command = None
            self._save_normal_reference_if_due(
                overall_ok=bool(overall_ok),
                elapsed_ms=elapsed_ms,
            )

            st.status = (
                "PLC INSPECT DONE: OK"
                if overall_ok
                else "PLC INSPECT DONE: NG"
            )

        except CameraRecoveryInProgress as e:
            self._pending_plc_camera_command = "inspect"
            self.plc.set_busy()
            st.status = "PLC INSPECT PAUSED: CAMERA AUTO RECOVERY"
            print("[PLC] inspection deferred:", e)

        except LightCommunicationError as e:
            st.light_error_latched = True

            self._save_plc_error_event(
                event_type="VISION_RUNTIME_ERROR",
                error_code=21,
                message="Light communication failed during inspection",
                exception=e,
            )

            self.plc.set_error(
                code=21,
                detail=str(e),
            )

            st.status = "PLC INSPECT LIGHT ERROR: RESET REQUIRED"

        except InspectionResultError as e:
            self._save_plc_error_event(
                event_type="VISION_INSPECTION_ERROR",
                error_code=40,
                message="Inspection result generation failed",
                exception=e,
            )

            self.plc.set_error(
                code=40,
                detail=str(e),
            )

            st.status = "PLC INSPECT RESULT ERROR: RESET REQUIRED"

        except Exception as e:
            print("[PLC] inspect exception:", e)

            self._save_plc_error_event(
                event_type="VISION_INSPECTION_ERROR",
                error_code=41,
                message="Unexpected exception during inspection",
                exception=e,
            )

            self.plc.set_error(
                code=41,
                detail=str(e),
            )

            st.status = "PLC INSPECT EXCEPTION: RESET REQUIRED"

    def _render_run_frame(self, vis, frame_gray8):
        st = self.state
        run_mode = str(self.runtime_cfg.get("run_mode", "held")).lower()

        if run_mode == "static":

            overlay.draw_rois(
                vis,
                rois=[
                    {
                        "id": r.get("id"),
                        "label": r.get("name"),
                        "rect": (
                            int(r.get("x", 0)),
                            int(r.get("y", 0)),
                            int(r.get("w", 0)),
                            int(r.get("h", 0)),
                        ),
                        "angle": float(r.get("angle", 0.0)),
                    }
                    for r in getattr(self.roi_mgr, "rois", [])
                ],
                active_id=self.roi_mgr.selected_id,
                roi_results=st.last_results,
                show_metrics=bool(self.runtime_cfg.get("dev_mode", False))
                and bool(self.runtime_cfg.get("dev_overlay_metrics", False)),
            )
            st.tracking_stable = True
            st.stable_frame_count = 999

        else:

            draw_run_tracking(
                vis,
                frame_gray8,
                runtime_cfg=self.runtime_cfg,
                product_profile=self.product_profile,
                state=st,
                roi_mgr=self.roi_mgr,
                inspector=self.inspector,
                stabilizer=self.stabilizer,
                data_dir=DATA_DIR,
                snapshot_cooldown=float(self.runtime_cfg.get("snapshot_cooldown", 5.0)),
                snapshot_keep=int(self.runtime_cfg.get("snapshot_keep", 200)),
                prune_snapshots=prune_snapshots,
                roi_label_pos=roi_label_pos,
                show_metrics=bool(self.runtime_cfg.get("dev_mode", False))
                and bool(self.runtime_cfg.get("dev_overlay_metrics", False)),
            )
            
    def _handle_key_input(self, key, frame_gray8, vis_bgr):
        st = self.state

        if (not st.edit_mode) and self.service_panel.handle_key(key):
            return

        if (not st.edit_mode) and key in (ord('v'), ord('V')):
            self._toggle_service_panel()
            return

        if key not in (-1, 255):
            print(f"[DBG KEY] raw={key} chr={repr(chr(key)) if 32 <= key <= 126 else 'NONPRINT'}")

        if key in (ord('d'), ord('D')):   # delete OK/NG Dataset reset
            print("[DBG KEY] DATASET RESET HOTKEY")
            self._reset_dataset()
            return

        if key in (ord('b'), ord('B')):   # baseline_profile.json reset
            print("[DBG KEY] BASELINE RESET HOTKEY")
            self._reset_baseline()
            return
        
        if key in (ord('u'), ord('U')):
            print("[DBG KEY] BASELINE UPDATE HOTKEY")
            if self._update_baseline_from_ok_results():
                st.status = "Baseline updated from OK result"
            else:
                st.status = "Baseline update skipped"
            return
        
        if key in (ord('l'), ord('L')):
            self._handle_baseline_key(frame_gray8)
            print("[DBG KEY] BASELINE LEARN HOTKEY")
            return
        
        if (not st.edit_mode) and key in (ord('t'), ord('T')):
            self._save_roi_and_template(frame_gray8)
            return
        
        consumed, sample_msg = handle_sample_keys(
            key,
            frame_gray8,
            vis_bgr,
            edit_mode=st.edit_mode,
            roi_mgr=self.roi_mgr,
            data_dir=DATA_DIR,
            snapshot_keep=int(self.runtime_cfg.get("snapshot_keep", 10)),
            last_results=st.last_results,
        )
        if consumed:
            if sample_msg:
                st.status = sample_msg
            return

        if key != -1:
            cmd = key_to_cmd(key, UICmd)
            if cmd != UICmd.NONE:
                st.pending_cmd = cmd

        if st.pending_cmd != UICmd.NONE:
            cmd_to_run = st.pending_cmd
            st.pending_cmd = UICmd.NONE
            if (
                self.soak_test.active
                and cmd_to_run not in (
                    UICmd.QUIT,
                    UICmd.TOGGLE_MODE,
                    UICmd.TOGGLE_SERVICE_PANEL,
                )
            ):
                st.status = "SOAK TEST ACTIVE: MANUAL COMMAND BLOCKED"
                return
            execute_command(self, cmd_to_run, frame_gray8, vis_bgr)

    def _add_baseline_from_last_results(self):
        st = self.state

        trk_score = getattr(self.state, "trk_score", 1.0)
        if trk_score < 0.90:
            self.state.status = f"Baseline skipped: low track {trk_score:.3f}"
            print(f"[AUTO BASELINE] {self.state.status}")
            return False

        if not st.last_results:
            st.status = "No inspect result to learn"
            print(f"[AUTO BASELINE] results type={type(st.last_results)}")
            return False

        for roi_id, res in st.last_results.items():
            metrics = getattr(res, "metrics", None) or {}
            roi_name = f"ROI{roi_id}"

            if roi_name in ("ROI2", "ROI3", "ROI4", "ROI5"):
                v = metrics.get("dark_ratio", None)
                if v is not None:
                    self.baseline.add_sample(roi_name, "dark_ratio", v)

            elif roi_name == "ROI6":
                v = metrics.get("blob_count", None)
                if v is None:
                    v = metrics.get("blob", None)
                if v is not None:
                    self.baseline.add_sample(roi_name, "blob_count", v)

        st.baseline_count += 1

        if st.baseline_count >= st.baseline_target_count:
            self.baseline.save()
            st.status = f"Baseline saved ({st.baseline_count}/{st.baseline_target_count})"
            print(f"[AUTO BASELINE] {st.status} -> {self.baseline_path}")
            st.baseline_learning = False
            st.baseline_count = 0
            self.baseline = AutoBaseline(self.baseline_path)
        else:
            st.status = f"Baseline sample added ({st.baseline_count}/{st.baseline_target_count})"

        return True

    def _handle_baseline_key(self, frame_gray8=None):
        st = self.state

        if st.edit_mode:
            st.status = "Baseline learn only in RUN mode"
            return

        if frame_gray8 is None:
            st.status = "No frame for baseline"
            return

        try:
            auto_mode_backup = getattr(self.inspector, "auto_mode", True)
            self.inspector.auto_mode = False

            try:
                st.last_overall_ok, st.last_results = self.inspector.inspect(
                    frame_gray8,
                    auto_mode=False
                )
            finally:
                self.inspector.auto_mode = auto_mode_backup
        except Exception as e:
            st.status = f"Baseline inspect fail: {e}"
            return

        if not st.baseline_learning:
            st.baseline_learning = True
            st.baseline_target_count = int(self.runtime_cfg.get("baseline_ok_count", 10))
            st.baseline_count = 0
            self.baseline = AutoBaseline(self.baseline_path)

        self._add_baseline_from_last_results()

    def _reset_dataset(self):
        import shutil

        targets = [
            os.path.join(DATA_DIR, "dataset", "OK"),
            os.path.join(DATA_DIR, "dataset", "NG"),
            os.path.join(DATA_DIR, "templates"),
        ]
        print("[DBG RESET] dataset targets =", targets)

        for p in targets:
            os.makedirs(p, exist_ok=True)
            for name in os.listdir(p):
                fp = os.path.join(p, name)
                print("[DBG RESET] remove:", fp)
                try:
                    if os.path.isfile(fp) or os.path.islink(fp):
                        os.remove(fp)
                    elif os.path.isdir(fp):
                        shutil.rmtree(fp)
                except Exception as e:
                    print("[DBG RESET] fail:", fp, e)

        self.state.status = "DATASET RESET DONE"
        print("[DBG RESET] DATASET RESET DONE")

    def _reset_baseline(self):
        try:
            print("[DBG RESET] baseline path =", self.baseline_path)
            if os.path.exists(self.baseline_path):
                os.remove(self.baseline_path)
                print("[DBG RESET] baseline removed")
            else:
                print("[DBG RESET] baseline file not found")

            self.baseline = AutoBaseline(self.baseline_path)
            self.state.baseline_learning = False
            self.state.baseline_count = 0
            self.state.status = "BASELINE RESET DONE"
            print("[DBG RESET] BASELINE RESET DONE")
        except Exception as e:
            print("[DBG RESET] BASELINE RESET FAIL:", e)
            self.state.status = f"BASELINE RESET FAIL: {e}"

    def _draw_ui(self, vis):
        st = self.state
        st.active_roi_id = self.roi_mgr.selected_id

        overlay.draw_status_bar(vis, st.status)

        if (not st.edit_mode) and (st.last_overall_ok is not None):
            overlay.draw_overall_banner(vis,st.last_overall_ok,info=getattr(st, "last_overall_info", None),)

        st.last_buttons = render_control_bar(
            vis,
            st.edit_mode,
            show_service=(
                bool(self.service_panel.enabled)
                and not st.edit_mode
            ),
        )

        if bool(self.runtime_cfg.get("enable_pose_guide", True)):
            vis = draw_pose_message(
                vis,
                st.pose_bad_cnt,
                int(self.runtime_cfg.get("pose_bad_n", 5)),
            )

        draw_mode_indicator(vis, st.edit_mode)
        draw_dev_hud(vis, st, self.product_profile)

        if not st.edit_mode:
            # Expensive service diagnostics are sampled only while the panel is
            # actually visible. Password input needs no PLC/ROI data.
            if self.service_panel.visible:
                now = time.time()
                refresh_interval = 1.0 / max(1.0, float(self.service_panel.refresh_hz))
                if (now - self._service_cache_epoch) >= refresh_interval:
                    try:
                        self._service_plc_snapshot = self.plc.get_state_snapshot()
                    except Exception as e:
                        self._service_plc_snapshot = {
                            "serial_open": False,
                            "comm_fault_active": True,
                            "error_code": 71,
                            "error_detail": str(e),
                        }
                    try:
                        self._service_recent_events = self.plc.get_recent_events(24)
                    except Exception:
                        self._service_recent_events = []
                    self._service_cache_epoch = now

                roi_interval = 1.0 / max(0.2, float(self.service_panel.roi_debug_hz))
                if (now - self._service_roi_cache_epoch) >= roi_interval:
                    try:
                        self._service_roi_debug = self.inspector.get_debug_grid()
                    except Exception:
                        self._service_roi_debug = None
                    self._service_roi_cache_epoch = now

            vis = self.service_panel.draw(
                vis,
                plc_snapshot=self._service_plc_snapshot,
                recent_events=self._service_recent_events,
                app_state=st,
                roi_debug=self._service_roi_debug,
                latest_log_path=st.test_error_log_path,
                error_log_path=st.latest_error_log_path,
                test_summary=self._get_service_test_summary(),
                soak_summary=self.soak_test.summary(),
            )
        else:
            self.service_panel.close()
        return vis
    
    def _prepare_frame(self, frame):
        st = self.state

        if frame is None:
            return None, None

        if frame.ndim == 3:
            # B0429 grayscale path usually arrives as 3-channel gray/BGR.
            # B0251/IMX477 nvargus path arrives as real BGR color, so convert to proper gray.
            if frame.shape[2] >= 3:
                frame_gray8 = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
            else:
                frame_gray8 = frame[:, :, 0]
        else:
            frame_gray8 = frame

        if (not st.edit_mode) and frame_gray8 is not None:
            norm_enabled = bool(self.runtime_cfg.get("normalize_enabled", False))
            norm_target = float(self.runtime_cfg.get("normalize_target_mean", 120.0))
            if norm_enabled:
                frame_gray8, _ = normalize_frame(
                    frame_gray8,
                    target_mean=norm_target,
                    do_clahe=True,
                )

        vis = cv2.cvtColor(frame_gray8, cv2.COLOR_GRAY2BGR)
        return frame_gray8, vis
    
    def _read_frame(self):
        started = time.perf_counter()
        try:
            frame = self.cam.read()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._last_camera_read_ms = elapsed_ms
            self._max_camera_read_ms = max(self._max_camera_read_ms, elapsed_ms)
            if elapsed_ms >= 100.0:
                self._slow_camera_read_count += 1
            return frame

        except Exception as e:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._last_camera_read_ms = elapsed_ms
            self._max_camera_read_ms = max(self._max_camera_read_ms, elapsed_ms)
            st = self.state
            st.camera_read_blocked = True

            if not st.camera_error_latched:
                st.camera_error_latched = True

                self._save_plc_error_event(
                    event_type="VISION_RUNTIME_ERROR",
                    error_code=11,
                    message="Camera frame read raised an exception",
                    exception=e,
                )

                self.plc.set_error(
                    code=11,
                    detail=str(e),
                )

            st.status = "CAMERA READ ERROR: RESET REQUIRED"
            return None
    # -------------------------
    # Main loop
    # -------------------------
    def run(self):
        st = self.state
        self.plc.start()

        plc_required = bool(
            self.plc_cfg.get("enabled", False)
        )

        plc_start_error = (
            plc_required
            and not self.plc.is_connected()
        )

        startup_error = plc_start_error

        if plc_required and not plc_start_error:
            self.plc.set_busy()

        # 카메라 초기화
        try:
            self.cam.open()
            st.camera_error_latched = False
            st.camera_read_blocked = False
            recovery_cfg = self._get_plc_recovery_cfg()
            st.camera_error_grace_until = (
                time.time()
                + recovery_cfg["camera_grace_sec"]
            )

        except Exception as e:
            startup_error = True
            st.camera_error_latched = True
            st.camera_read_blocked = True
            st.status = f"CAMERA INIT ERROR: {e}"

            self._save_plc_error_event(
                event_type="VISION_STARTUP_ERROR",
                error_code=10,
                message="Camera initialization failed",
                exception=e,
            )

            self.plc.set_error(
                code=10,
                detail=str(e),
            )

        if plc_start_error:
            self._poll_plc_comm_events()

        # 조명 초기화
        try:
            self.light.start()
            self.runtime_cfg["_light_state"] = self.light.get_state()

            light_failures = self._get_light_failures()

            if light_failures:
                raise RuntimeError(
                    f"light startup failed: {light_failures}"
                )

            st.light_error_latched = False

        except Exception as e:
            startup_error = True
            st.light_error_latched = True
            st.status = f"LIGHT START ERROR: {e}"

            self._save_plc_error_event(
                event_type="VISION_STARTUP_ERROR",
                error_code=20,
                message="Light initialization failed",
                exception=e,
            )

            self.plc.set_error(
                code=20,
                detail=str(e),
            )

        # 시작 과정 중 발생한 PLC 통신 이벤트를 반영한 뒤 Ready를 결정한다.
        self._poll_plc_comm_events()

        if plc_required and not self.plc.is_connected():
            startup_error = True

        try:
            startup_plc_snapshot = self.plc.get_state_snapshot()
            plc_status = int(startup_plc_snapshot.get("status", 0))
            plc_link_verified = bool(
                startup_plc_snapshot.get("link_verified", False)
            )
            if plc_required and (
                plc_status == 3 or not plc_link_verified
            ):
                startup_error = True
        except Exception:
            if plc_required:
                startup_error = True

        if not startup_error:
            self.plc.reset_to_ready(reset_heartbeat=False)
            st.status = "PLC READY" if plc_required else "READY"

        test_cfg = self._get_plc_error_test_cfg()
        if test_cfg["enabled"]:
            print(
                "[PLC TEST] forced-error test mode enabled "
                f"request_path={test_cfg['request_path']}"
            )
            try:
                self.plc.trace_application_event(
                    event="ERROR_TEST_MODE_ENABLED",
                    summary=(
                        "forced-error test mode enabled "
                        f"request_path={test_cfg['request_path']}"
                    ),
                    extra={
                        "test_mode_enabled": True,
                        "test_request_path": test_cfg["request_path"],
                    },
                )
            except Exception:
                pass

        recovery_cfg = self._get_plc_recovery_cfg()
        frame_timeout_sec = recovery_cfg["frame_timeout_sec"]
        last_ok_frame_time = time.time()
        last_camera_vis = None
        last_camera_vis_update_epoch = 0.0

        while True:
            self._poll_plc_comm_events()
            self._poll_plc_error_test_request()
            self._poll_camera_process_request()
            camera_process_status = self._sync_camera_process_state()
            self._complete_auto_plc_reconnect_if_ready()

            frame = None if st.camera_read_blocked else self._read_frame()

            if frame is None:
                # read() may have caused the monitor to enter recovery. Refresh
                # once before deciding whether this is a terminal camera error.
                camera_process_status = self._sync_camera_process_state()
                camera_process_state = str(camera_process_status.get("state", ""))
                camera_recovering = camera_process_state == "RECOVERING"
                isolated_camera_wait = bool(
                    self._camera_recovery_capable()
                    and camera_process_state in ("STARTING", "RUNNING", "RECOVERING")
                )
                now = time.time()
                no_frame_sec = now - last_ok_frame_time

                # A real camera failure must be latched before the soak/PLC
                # callbacks run. Otherwise the next loop can enter another native
                # cap.read() block before the service test notices the fault.
                if (
                    no_frame_sec > frame_timeout_sec
                    and now >= st.camera_error_grace_until
                    and not st.camera_error_latched
                    and not camera_recovering
                ):
                    error = RuntimeError(
                        f"camera frame unavailable for {no_frame_sec:.2f} sec"
                    )

                    st.camera_error_latched = True
                    st.camera_read_blocked = True

                    self._save_plc_error_event(
                        event_type="VISION_RUNTIME_ERROR",
                        error_code=11,
                        message="Camera frame reception stopped",
                        exception=error,
                    )

                    self.plc.set_error(
                        code=11,
                        detail=str(error),
                    )

                    st.status = "CAMERA FRAME ERROR: RESET REQUIRED"

                elif camera_recovering:
                    elapsed = float(
                        camera_process_status.get("recovery_elapsed_sec", 0.0) or 0.0
                    )
                    st.status = f"CAMERA AUTO RECOVERY: {elapsed:.1f}s"
                elif (
                    no_frame_sec > frame_timeout_sec
                    and not st.camera_error_latched
                ):
                    st.status = "WAITING FOR CAMERA FRAME"

                # During isolated capture recovery, do not consume D200=1/2.
                # They remain pending in the Modbus register and are processed
                # after the first healthy frame. Reset/shutdown/emergency remain live.
                if isolated_camera_wait:
                    self._handle_camera_recovery_plc_tick()
                else:
                    self._run_plc_inspect_tick(None, None)
                self._spot_timeout_tick()
                self._run_service_soak_tick(None, None)

                if last_camera_vis is not None:
                    no_frame_vis = last_camera_vis.copy()
                else:
                    no_frame_vis = np.zeros(
                        (self.frame_height, self.frame_width, 3),
                        dtype=np.uint8,
                    )

                warning_layer = no_frame_vis.copy()
                cv2.rectangle(
                    warning_layer,
                    (0, 0),
                    (self.frame_width, 72),
                    (0, 150, 210) if camera_recovering else (0, 0, 180),
                    -1,
                )
                cv2.addWeighted(
                    warning_layer,
                    0.72,
                    no_frame_vis,
                    0.28,
                    0,
                    no_frame_vis,
                )
                recovery_elapsed = float(
                    camera_process_status.get("recovery_elapsed_sec", 0.0) or 0.0
                )
                warning_text = (
                    f"CAMERA AUTO RECOVERY {recovery_elapsed:.1f}s - PLC/UI ACTIVE"
                    if camera_recovering
                    else "CAMERA FRAME UNAVAILABLE - PLC RESET / SERVICE CHECK"
                )
                cv2.putText(
                    no_frame_vis,
                    warning_text,
                    (24, 47),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                no_frame_vis = self._draw_ui(no_frame_vis)
                cv2.imshow(self.win, no_frame_vis)

                try:
                    key = cv2.waitKeyEx(1)
                except Exception:
                    key = cv2.waitKey(1)

                # 카메라가 없어도 SERVICE 패널/종료 키는 계속 동작한다.
                self._handle_key_input(key, None, no_frame_vis)

                if st.quit_requested:
                    break

                continue

            last_ok_frame_time = time.time()
            self._last_good_camera_frame_epoch = last_ok_frame_time
            self._camera_frame_seq += 1
            self._loop_fps_frames += 1
            fps_elapsed = last_ok_frame_time - self._loop_fps_last_epoch
            if fps_elapsed >= 1.0:
                instant_fps = float(self._loop_fps_frames) / max(0.001, fps_elapsed)
                self._loop_fps = (
                    instant_fps
                    if self._loop_fps <= 0
                    else (self._loop_fps * 0.75 + instant_fps * 0.25)
                )
                self._loop_fps_frames = 0
                self._loop_fps_last_epoch = last_ok_frame_time

            frame_gray8, vis = self._prepare_frame(frame)
            if frame_gray8 is None:
                cv2.waitKey(1)
                if st.quit_requested:
                    break
                continue

            try:
                plc_snapshot = self.plc.get_state_snapshot()
                plc_healthy = bool(
                    not plc_required
                    or (
                        bool(plc_snapshot.get("serial_open", False))
                        and bool(plc_snapshot.get("link_verified", False))
                        and not bool(
                            plc_snapshot.get("comm_fault_active", False)
                        )
                        and int(plc_snapshot.get("status", 0)) not in (3, 8)
                        and int(plc_snapshot.get("error_code", 0)) == 0
                        and int(plc_snapshot.get("command", 0)) not in (8, 9)
                    )
                )
            except Exception:
                plc_healthy = not plc_required

            vision_ready = bool(
                not st.edit_mode
                and not st.camera_error_latched
                and not st.light_error_latched
                and not st.plc_shutdown_started
                and not st.plc_ready_consumed
                and not self._camera_is_recovering()
                and plc_healthy
            )

            self._set_plc_vision_ready(
                vision_ready,
                "RUN_HEALTHY_FRAME" if vision_ready else "NOT_READY",
            )

            # edit / run draw
            if st.edit_mode:
                self.editor.update(vis)
            else:
                self._render_run_frame(vis, frame_gray8)

            # Keep only a low-rate fallback frame for camera-error display.
            # Copying full HD BGR/GRAY frames every loop created unnecessary
            # memory-bandwidth pressure while the service panel was open.
            if (
                last_camera_vis is None
                or (last_ok_frame_time - last_camera_vis_update_epoch) >= 1.0
            ):
                last_camera_vis = vis.copy()
                last_camera_vis_update_epoch = last_ok_frame_time

            pending_camera_cmd = self._pending_plc_camera_command
            if pending_camera_cmd:
                try:
                    snapshot = self.plc.get_state_snapshot()
                    command_value = int(snapshot.get("command", 0))
                except Exception:
                    command_value = 0
                expected_value = 1 if pending_camera_cmd == "prepare" else 2
                if command_value == expected_value:
                    self._run_plc_inspect_tick(
                        frame_gray8,
                        vis,
                        forced_cmd=pending_camera_cmd,
                    )
                else:
                    self._pending_plc_camera_command = None
                    self._run_plc_inspect_tick(frame_gray8, vis)
            else:
                self._run_plc_inspect_tick(frame_gray8, vis)
            self._spot_timeout_tick()
            self._run_service_soak_tick(frame_gray8, vis)

            if (
                not st.edit_mode
                and not st.camera_error_latched
                and not st.light_error_latched
                and not st.test_error_active
                and not self.soak_test.active
            ):
                self._run_auto_inspect_tick(frame_gray8, vis)
        
            # Status + banner + control Bar(button) + HUD  Draw
            vis = self._draw_ui(vis)

            # show
            cv2.imshow(self.win, vis)

            # key
            try:
                key = cv2.waitKeyEx(1)
            except Exception:
                key = cv2.waitKey(1)

            self._handle_key_input(key, frame_gray8, vis)

            if st.quit_requested:
                break

        if self.soak_test.active:
            self._stop_service_soak_test("APP_EXIT")

        try:
            cv2.destroyWindow(self.win)
            cv2.waitKey(1)
        except Exception:
            pass

        try:
            self._set_plc_vision_ready(False, "APP_EXIT")
            self.plc.stop()
        except Exception as e:
            print("[PLC] stop failed:", e)

        try:
            self.light.stop()
        except Exception as e:
            print("[LIGHT] stop failed:", e)

        # B0251 / nvargus는 release()에서 Argus cleanup 대기 때문에
        # 종료가 4~5초 지연될 수 있음.
        fast_exit = False
        try:
            pipeline_type = str(self.camera_info.get("pipeline_type", "") or "")
            fast_exit = pipeline_type == "nvargus_bgr"
        except Exception:
            fast_exit = False

        if fast_exit:
            # Give Argus a bounded cleanup opportunity. If release blocks, the
            # daemon thread is abandoned and the process still exits quickly.
            release_done = threading.Event()

            def _release_camera():
                try:
                    self.cam.release()
                except Exception as e:
                    print("[CAM] release failed:", e)
                finally:
                    release_done.set()

            release_thread = threading.Thread(
                target=_release_camera,
                name="nvargus-release",
                daemon=True,
            )
            release_thread.start()
            release_thread.join(timeout=2.5)
            if release_done.is_set():
                print("[CAM] nvargus release completed")
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
                return

            print("[CAM] nvargus release timeout: force process exit")
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

        try:
            self.cam.release()
        except Exception as e:
            print("[CAM] release failed:", e)

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Daol vision application",
    )
    parser.add_argument(
        "--startup-mode",
        choices=("auto", "edit", "run"),
        default="auto",
        help=(
            "auto: terminal launch=EDIT, desktop/system autostart=RUN; "
            "edit/run: explicit override"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    resolved_startup_mode = _resolve_startup_mode(args.startup_mode)
    app = None

    try:
        _acquire_single_instance_lock()
        fault_log_path = _enable_runtime_fault_log()
        if fault_log_path:
            print(f"[RUNTIME LOG] {fault_log_path}")
        app = VisionApp(startup_mode=resolved_startup_mode)
        app.run()
        _write_runtime_marker("NORMAL APP RETURN")

    except Exception as e:
        print(f"[FATAL] vision application terminated: {e}")
        traceback.print_exc()
        _write_runtime_marker(
            f"UNHANDLED EXCEPTION {type(e).__name__}: {e}"
        )

        if app is not None:
            try:
                app._save_plc_error_event(
                    event_type="VISION_FATAL_ERROR",
                    error_code=41,
                    message="Unhandled vision application exception",
                    exception=e,
                )
            except Exception:
                pass

            try:
                app.plc.set_error(code=41, detail=str(e))
            except Exception:
                pass

            try:
                app.plc.stop()
            except Exception:
                pass

            try:
                app.light.stop()
            except Exception:
                pass

            try:
                app.cam.release()
            except Exception:
                pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        raise


if __name__ == "__main__":
    main()
