#!/usr/bin/env python3
"""Isolated camera capture process for Argus/OpenCV stability.

The UI/PLC process never owns an Argus VideoCapture handle.  If the capture
worker stalls, only the worker is killed and recreated while the main process
continues servicing the UI and PLC.
"""

from __future__ import annotations

import ctypes
import json
import multiprocessing as mp
import os
import shlex
import subprocess
import threading
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np

from capture.camera_gst import CameraGST


STATE_CLOSED = "CLOSED"
STATE_STARTING = "STARTING"
STATE_RUNNING = "RUNNING"
STATE_RECOVERING = "RECOVERING"
STATE_FAILED = "FAILED"


def _safe_send(conn, payload) -> None:
    try:
        conn.send(payload)
    except Exception:
        pass


def _camera_worker(
    gst_pipeline: str,
    profile_name: str,
    frame_shape: Tuple[int, int, int],
    shared_buffer,
    active_index,
    frame_seq,
    last_frame_mono,
    stop_event,
    simulate_hang_event,
    event_conn,
) -> None:
    """Capture frames continuously and publish the newest frame in two buffers."""
    cam = None
    try:
        height, width, channels = frame_shape
        frame_bytes = int(height * width * channels)
        buffers = np.frombuffer(shared_buffer, dtype=np.uint8).reshape(
            (2, height, width, channels)
        )

        cam = CameraGST(gst_pipeline)
        if profile_name:
            cam.set_profile(profile_name)
        cam.open()
        _safe_send(event_conn, ("OPENED", os.getpid(), time.time(), ""))

        local_seq = 0
        while not stop_event.is_set():
            if simulate_hang_event.is_set():
                _safe_send(
                    event_conn,
                    ("SIMULATED_HANG", os.getpid(), time.time(), "capture paused"),
                )
                while simulate_hang_event.is_set() and not stop_event.is_set():
                    time.sleep(0.1)
                continue

            frame = cam.read()
            if frame is None:
                raise RuntimeError("camera worker received no frame")

            if frame.ndim == 2:
                frame = np.repeat(frame[:, :, None], channels, axis=2)
            elif frame.ndim == 3 and frame.shape[2] != channels:
                if frame.shape[2] > channels:
                    frame = frame[:, :, :channels]
                else:
                    frame = np.repeat(frame[:, :, :1], channels, axis=2)

            if frame.shape != frame_shape:
                raise RuntimeError(
                    f"camera frame shape mismatch: got={frame.shape}, expected={frame_shape}"
                )
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            next_index = 1 - int(active_index.value)
            np.copyto(buffers[next_index], frame, casting="no")
            local_seq += 1
            active_index.value = next_index
            frame_seq.value = local_seq
            last_frame_mono.value = time.monotonic()

            if local_seq == 1:
                _safe_send(event_conn, ("FIRST_FRAME", os.getpid(), time.time(), ""))

    except BaseException as exc:
        _safe_send(
            event_conn,
            (
                "ERROR",
                os.getpid(),
                time.time(),
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}",
            ),
        )
    finally:
        if cam is not None:
            try:
                cam.release()
            except Exception:
                pass
        _safe_send(event_conn, ("EXIT", os.getpid(), time.time(), ""))
        try:
            event_conn.close()
        except Exception:
            pass


class CameraProcessProxy:
    """CameraGST-compatible proxy backed by an isolated capture process."""

    supports_auto_recovery = True

    def __init__(
        self,
        gst_pipeline: str,
        *,
        width: int,
        height: int,
        channels: int = 3,
        process_cfg: Optional[Dict[str, Any]] = None,
        project_root: str = "",
    ):
        self.gst = str(gst_pipeline)
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.frame_shape = (self.height, self.width, self.channels)
        self.frame_bytes = int(self.height * self.width * self.channels)

        cfg = process_cfg or {}
        self.read_wait_timeout_sec = max(
            0.01, float(cfg.get("read_wait_timeout_sec", 0.08))
        )
        self.stall_timeout_sec = max(
            0.25, float(cfg.get("stall_timeout_sec", 0.75))
        )
        self.open_timeout_sec = max(
            2.0, float(cfg.get("open_timeout_sec", 10.0))
        )
        self.healthy_frames = max(1, int(cfg.get("healthy_frames", 3)))
        self.max_recovery_attempts = max(
            1, int(cfg.get("max_recovery_attempts", 2))
        )
        self.startup_auto_recovery = bool(
            cfg.get("startup_auto_recovery", True)
        )
        default_startup_recovery_timeout = (
            self.max_recovery_attempts
            * (
                self.open_timeout_sec
                + max(2.0, float(cfg.get("daemon_restart_timeout_sec", 10.0)))
                + max(0.0, float(cfg.get("daemon_settle_sec", 1.2)))
                + 1.0
            )
            + 5.0
        )
        self.startup_recovery_timeout_sec = max(
            self.open_timeout_sec,
            float(
                cfg.get(
                    "startup_recovery_timeout_sec",
                    default_startup_recovery_timeout,
                )
            ),
        )
        self.worker_stop_timeout_sec = max(
            0.1, float(cfg.get("worker_stop_timeout_sec", 0.6))
        )
        self.monitor_interval_sec = max(
            0.05, float(cfg.get("monitor_interval_sec", 0.1))
        )

        self.restart_nvargus_daemon = bool(
            cfg.get("restart_nvargus_daemon", True)
        )
        self.daemon_restart_command = str(
            cfg.get(
                "daemon_restart_command",
                "sudo -n /usr/bin/systemctl restart nvargus-daemon",
            )
        ).strip()
        self.daemon_restart_timeout_sec = max(
            2.0, float(cfg.get("daemon_restart_timeout_sec", 10.0))
        )
        self.daemon_settle_sec = max(
            0.0, float(cfg.get("daemon_settle_sec", 1.2))
        )
        self.daemon_restart_required = bool(
            cfg.get("daemon_restart_required", True)
        )

        root = os.path.abspath(project_root or os.getcwd())
        status_path = str(
            cfg.get("status_path", "data/runtime/camera_process_status.json")
        )
        request_path = str(
            cfg.get("request_path", "data/runtime/camera_process_request.json")
        )
        self.status_path = (
            status_path if os.path.isabs(status_path) else os.path.join(root, status_path)
        )
        self.request_path = (
            request_path if os.path.isabs(request_path) else os.path.join(root, request_path)
        )

        self._ctx = mp.get_context("spawn")
        self._shared_buffer = self._ctx.RawArray(
            ctypes.c_ubyte, self.frame_bytes * 2
        )
        self._active_index = self._ctx.Value("i", 0)
        self._frame_seq = self._ctx.Value("Q", 0)
        self._last_frame_mono = self._ctx.Value("d", 0.0)

        self._process = None
        self._stop_event = None
        self._simulate_hang_event = None
        self._event_conn = None
        self._profile_name = "default"
        self._last_read_seq = 0

        self._state_lock = threading.RLock()
        self._worker_lock = threading.RLock()
        self._recovery_lock = threading.Lock()
        self._state = STATE_CLOSED
        self._state_detail = ""
        self._last_worker_error = ""
        self._generation = 0
        self._recovery_count = 0
        self._recovery_success_count = 0
        self._recovery_failure_count = 0
        self._recovery_started_mono = 0.0
        self._last_recovery_elapsed_sec = 0.0
        self._last_recovery_reason = ""
        self._last_recovery_result = ""
        self._last_recovery_detail = ""
        self._last_state_change_epoch = time.time()

        self._shutdown_event = threading.Event()
        self._monitor_thread = None
        self._recovery_thread = None

        # Compatibility with code that checks cam.cap.isOpened().
        self.cap = self
        self._write_status()

    def set_profile(self, name: str) -> None:
        self._profile_name = str(name or "default")

    def _set_state(self, state: str, detail: str = "") -> None:
        with self._state_lock:
            self._state = str(state)
            self._state_detail = str(detail or "")
            self._last_state_change_epoch = time.time()
        self._write_status()

    def _drain_worker_events(self) -> None:
        conn = self._event_conn
        if conn is None:
            return
        try:
            while conn.poll():
                event, pid, epoch, detail = conn.recv()
                if event == "ERROR":
                    self._last_worker_error = str(detail or "")
                elif event == "SIMULATED_HANG":
                    self._state_detail = "simulated capture hang"
        except (EOFError, OSError):
            pass
        except Exception:
            pass

    def _spawn_worker(self) -> None:
        with self._worker_lock:
            self._frame_seq.value = 0
            self._last_frame_mono.value = 0.0
            self._active_index.value = 0
            self._last_read_seq = 0

            self._stop_event = self._ctx.Event()
            self._simulate_hang_event = self._ctx.Event()
            parent_conn, child_conn = self._ctx.Pipe(duplex=False)
            self._event_conn = parent_conn

            process = self._ctx.Process(
                target=_camera_worker,
                args=(
                    self.gst,
                    self._profile_name,
                    self.frame_shape,
                    self._shared_buffer,
                    self._active_index,
                    self._frame_seq,
                    self._last_frame_mono,
                    self._stop_event,
                    self._simulate_hang_event,
                    child_conn,
                ),
                name="vision-camera-capture",
                daemon=True,
            )
            process.start()
            child_conn.close()
            self._process = process
            self._generation += 1

    def _wait_for_healthy_frames(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        first_seq = 0
        while time.monotonic() < deadline:
            self._drain_worker_events()
            process = self._process
            if process is None or not process.is_alive():
                return False
            seq = int(self._frame_seq.value)
            if seq > 0 and first_seq == 0:
                first_seq = seq
            if first_seq > 0 and seq >= first_seq + self.healthy_frames - 1:
                return True
            time.sleep(0.03)
        return False

    def _stop_worker(self, *, force: bool) -> None:
        with self._worker_lock:
            process = self._process
            if process is None:
                return

            if not force and self._stop_event is not None:
                self._stop_event.set()
                process.join(timeout=self.worker_stop_timeout_sec)

            if process.is_alive():
                process.terminate()
                process.join(timeout=self.worker_stop_timeout_sec)

            if process.is_alive():
                try:
                    process.kill()
                except AttributeError:
                    os.kill(process.pid, 9)
                process.join(timeout=1.0)

            try:
                process.close()
            except Exception:
                pass
            try:
                if self._event_conn is not None:
                    self._event_conn.close()
            except Exception:
                pass

            self._process = None
            self._event_conn = None
            self._stop_event = None
            self._simulate_hang_event = None

    def _restart_daemon(self) -> Tuple[bool, str]:
        if not self.restart_nvargus_daemon:
            return True, "daemon restart disabled"
        if not self.daemon_restart_command:
            return not self.daemon_restart_required, "daemon restart command is empty"

        command = shlex.split(self.daemon_restart_command)
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.daemon_restart_timeout_sec,
                check=False,
            )
        except Exception as exc:
            return False, f"daemon restart exception: {type(exc).__name__}: {exc}"

        output = " ".join(
            part.strip()
            for part in (completed.stdout or "", completed.stderr or "")
            if part.strip()
        )
        if completed.returncode != 0:
            return False, (
                f"daemon restart rc={completed.returncode}"
                + (f" output={output}" if output else "")
            )

        if self.daemon_settle_sec > 0:
            time.sleep(self.daemon_settle_sec)
        return True, output or "nvargus-daemon restarted"

    def _ensure_monitor_thread(self) -> None:
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._shutdown_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="camera-process-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def open(self) -> None:
        with self._state_lock:
            if self._state in (STATE_RUNNING, STATE_STARTING, STATE_RECOVERING):
                return
            self._state = STATE_STARTING
            self._state_detail = "starting capture worker"
            self._last_state_change_epoch = time.time()
        self._write_status()

        self._spawn_worker()
        if not self._wait_for_healthy_frames(self.open_timeout_sec):
            self._drain_worker_events()
            initial_error = (
                self._last_worker_error
                or "camera worker did not deliver healthy frames"
            )
            self._stop_worker(force=True)
            self._set_state(STATE_FAILED, initial_error)

            if self.startup_auto_recovery:
                try:
                    self.recover_blocking(
                        reason=(
                            "initial camera startup failed: "
                            f"{initial_error}"
                        ),
                        timeout_sec=self.startup_recovery_timeout_sec,
                    )
                    self._ensure_monitor_thread()
                    return
                except Exception as recovery_error:
                    detail = (
                        f"initial camera open failed: {initial_error}; "
                        f"automatic startup recovery failed: {recovery_error}"
                    )
                    self._set_state(STATE_FAILED, detail)
                    raise RuntimeError(detail)

            raise RuntimeError(initial_error)

        self._set_state(STATE_RUNNING, "capture worker ready")
        self._ensure_monitor_thread()

    def _monitor_loop(self) -> None:
        while not self._shutdown_event.wait(self.monitor_interval_sec):
            self._drain_worker_events()
            status = self.get_status(write_file=False)
            state = status["state"]
            if state != STATE_RUNNING:
                continue

            process = self._process
            if process is None or not process.is_alive():
                self.request_recovery("camera worker exited")
                continue

            frame_age = status.get("frame_age_sec")
            if frame_age is not None and frame_age > self.stall_timeout_sec:
                self.request_recovery(
                    f"camera frame stale for {frame_age:.3f}s"
                )

    def request_recovery(self, reason: str = "camera fault") -> bool:
        if self._shutdown_event.is_set():
            return False
        with self._state_lock:
            if self._state == STATE_RECOVERING:
                return False
            self._state = STATE_RECOVERING
            self._state_detail = str(reason)
            self._recovery_started_mono = time.monotonic()
            self._recovery_count += 1
            self._last_recovery_reason = str(reason)
            self._last_recovery_result = ""
            self._last_recovery_detail = ""
            self._last_state_change_epoch = time.time()
        self._write_status()

        self._recovery_thread = threading.Thread(
            target=self._recover_worker,
            name="camera-process-recovery",
            daemon=True,
        )
        self._recovery_thread.start()
        return True

    def _recover_worker(self) -> None:
        if not self._recovery_lock.acquire(blocking=False):
            return
        started = time.monotonic()
        details = []
        try:
            self._stop_worker(force=True)
            if self._shutdown_event.is_set():
                self._set_state(STATE_CLOSED, "recovery cancelled by shutdown")
                return

            for attempt in range(1, self.max_recovery_attempts + 1):
                if self._shutdown_event.is_set():
                    self._set_state(STATE_CLOSED, "recovery cancelled by shutdown")
                    return
                daemon_ok, daemon_detail = self._restart_daemon()
                details.append(f"attempt={attempt} daemon={daemon_detail}")
                if not daemon_ok and self.daemon_restart_required:
                    if attempt < self.max_recovery_attempts:
                        time.sleep(0.5)
                        continue
                    raise RuntimeError(daemon_detail)

                if self._shutdown_event.is_set():
                    self._set_state(STATE_CLOSED, "recovery cancelled by shutdown")
                    return
                self._spawn_worker()
                if self._wait_for_healthy_frames(self.open_timeout_sec):
                    elapsed = time.monotonic() - started
                    with self._state_lock:
                        self._state = STATE_RUNNING
                        self._state_detail = "automatic recovery completed"
                        self._recovery_success_count += 1
                        self._last_recovery_elapsed_sec = elapsed
                        self._last_recovery_result = "OK"
                        self._last_recovery_detail = "; ".join(details)
                        self._last_state_change_epoch = time.time()
                    self._write_status()
                    return

                self._drain_worker_events()
                worker_error = self._last_worker_error or "no healthy camera frames"
                details.append(f"attempt={attempt} worker={worker_error}")
                self._stop_worker(force=True)
                if attempt < self.max_recovery_attempts:
                    time.sleep(0.5)

            raise RuntimeError("; ".join(details))

        except BaseException as exc:
            elapsed = time.monotonic() - started
            with self._state_lock:
                self._state = STATE_FAILED
                self._state_detail = str(exc)
                self._recovery_failure_count += 1
                self._last_recovery_elapsed_sec = elapsed
                self._last_recovery_result = "FAIL"
                self._last_recovery_detail = str(exc)
                self._last_state_change_epoch = time.time()
            self._write_status()
        finally:
            self._recovery_lock.release()

    def recover_blocking(self, reason: str = "manual recovery", timeout_sec: float = 30.0):
        self.request_recovery(reason)
        deadline = time.monotonic() + max(1.0, float(timeout_sec))
        while time.monotonic() < deadline:
            status = self.get_status()
            if status["state"] == STATE_RUNNING:
                frame = self.read()
                if frame is not None:
                    return frame
            if status["state"] == STATE_FAILED:
                raise RuntimeError(status.get("detail") or "camera recovery failed")
            time.sleep(0.05)
        raise TimeoutError("camera recovery did not finish before timeout")

    def simulate_hang(self, reason: str = "service camera recovery demo") -> bool:
        with self._state_lock:
            if self._state != STATE_RUNNING:
                return False
        event = self._simulate_hang_event
        if event is None:
            return False
        self._last_recovery_reason = str(reason)
        event.set()
        self._write_status()
        return True

    def read(self):
        self._drain_worker_events()
        with self._state_lock:
            state = self._state
        if state != STATE_RUNNING:
            return None

        deadline = time.monotonic() + self.read_wait_timeout_sec
        seq = int(self._frame_seq.value)
        while seq == self._last_read_seq and time.monotonic() < deadline:
            process = self._process
            if process is None or not process.is_alive():
                self.request_recovery("camera worker exited during read")
                return None
            time.sleep(0.003)
            seq = int(self._frame_seq.value)

        if seq <= 0 or seq == self._last_read_seq:
            return None

        buffers = np.frombuffer(self._shared_buffer, dtype=np.uint8).reshape(
            (2, self.height, self.width, self.channels)
        )

        for _ in range(2):
            seq_before = int(self._frame_seq.value)
            index = int(self._active_index.value)
            frame = buffers[index].copy()
            seq_after = int(self._frame_seq.value)
            if seq_before == seq_after and index == int(self._active_index.value):
                self._last_read_seq = seq_after
                return frame

        self._last_read_seq = int(self._frame_seq.value)
        return buffers[int(self._active_index.value)].copy()

    def isOpened(self) -> bool:
        with self._state_lock:
            running = self._state == STATE_RUNNING
        process = self._process
        return bool(running and process is not None and process.is_alive())

    def get_status(self, *, write_file: bool = False) -> Dict[str, Any]:
        self._drain_worker_events()
        now_mono = time.monotonic()
        last_frame_mono = float(self._last_frame_mono.value)
        frame_age = (
            max(0.0, now_mono - last_frame_mono)
            if last_frame_mono > 0
            else None
        )
        with self._state_lock:
            recovery_elapsed = (
                max(0.0, now_mono - self._recovery_started_mono)
                if self._state == STATE_RECOVERING and self._recovery_started_mono > 0
                else self._last_recovery_elapsed_sec
            )
            status = {
                "state": self._state,
                "detail": self._state_detail,
                "is_opened": self.isOpened(),
                "worker_pid": int(self._process.pid) if self._process is not None else 0,
                "worker_alive": bool(
                    self._process is not None and self._process.is_alive()
                ),
                "generation": int(self._generation),
                "frame_seq": int(self._frame_seq.value),
                "frame_age_sec": round(frame_age, 4) if frame_age is not None else None,
                "recovery_count": int(self._recovery_count),
                "recovery_success_count": int(self._recovery_success_count),
                "recovery_failure_count": int(self._recovery_failure_count),
                "recovery_elapsed_sec": round(float(recovery_elapsed), 3),
                "last_recovery_reason": self._last_recovery_reason,
                "last_recovery_result": self._last_recovery_result,
                "last_recovery_detail": self._last_recovery_detail,
                "last_worker_error": self._last_worker_error,
                "last_state_change_epoch": float(self._last_state_change_epoch),
                "status_path": self.status_path,
                "request_path": self.request_path,
            }
        if write_file:
            self._write_status(status)
        return status

    def _write_status(self, status: Optional[Dict[str, Any]] = None) -> None:
        try:
            if status is None:
                # Avoid recursion through get_status(write_file=True).
                now_mono = time.monotonic()
                last_frame_mono = float(self._last_frame_mono.value)
                frame_age = (
                    max(0.0, now_mono - last_frame_mono)
                    if last_frame_mono > 0
                    else None
                )
                with self._state_lock:
                    elapsed = (
                        max(0.0, now_mono - self._recovery_started_mono)
                        if self._state == STATE_RECOVERING
                        and self._recovery_started_mono > 0
                        else self._last_recovery_elapsed_sec
                    )
                    status = {
                        "state": self._state,
                        "detail": self._state_detail,
                        "is_opened": self.isOpened(),
                        "worker_pid": int(self._process.pid) if self._process is not None else 0,
                        "worker_alive": bool(
                            self._process is not None and self._process.is_alive()
                        ),
                        "generation": int(self._generation),
                        "frame_seq": int(self._frame_seq.value),
                        "frame_age_sec": round(frame_age, 4) if frame_age is not None else None,
                        "recovery_count": int(self._recovery_count),
                        "recovery_success_count": int(self._recovery_success_count),
                        "recovery_failure_count": int(self._recovery_failure_count),
                        "recovery_elapsed_sec": round(float(elapsed), 3),
                        "last_recovery_reason": self._last_recovery_reason,
                        "last_recovery_result": self._last_recovery_result,
                        "last_recovery_detail": self._last_recovery_detail,
                        "last_worker_error": self._last_worker_error,
                        "last_state_change_epoch": float(self._last_state_change_epoch),
                        "status_path": self.status_path,
                        "request_path": self.request_path,
                    }
            os.makedirs(os.path.dirname(self.status_path), exist_ok=True)
            temp_path = self.status_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as file:
                json.dump(status, file, ensure_ascii=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, self.status_path)
        except Exception:
            pass

    def release(self) -> None:
        self._shutdown_event.set()
        self._set_state(STATE_CLOSED, "camera proxy closing")
        self._stop_worker(force=False)
        monitor = self._monitor_thread
        if monitor is not None and monitor.is_alive():
            monitor.join(timeout=0.5)
        self._monitor_thread = None
        recovery = self._recovery_thread
        if recovery is not None and recovery.is_alive():
            recovery.join(timeout=1.0)
        self._recovery_thread = None
        self._set_state(STATE_CLOSED, "camera proxy closed")
