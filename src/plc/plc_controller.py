import json
import os
import threading
import time
from typing import Any, Dict, Optional
from datetime import datetime

try:
    import serial
except Exception:
    serial = None


def _crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def _append_crc(data: bytes) -> bytes:
    crc = _crc16_modbus(data)
    return data + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def _check_crc(frame: bytes) -> bool:
    if len(frame) < 4:
        return False
    recv = frame[-2] | (frame[-1] << 8)
    calc = _crc16_modbus(frame[:-2])
    return recv == calc


class DisabledPlcController:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.enabled = False

    def start(self):
        print("[PLC] disabled")

    def stop(self):
        pass

    def tick(self):
        pass

    def poll_command(self) -> Optional[str]:
        return None

    def set_idle(self):
        pass

    def reset_to_ready(self, reset_heartbeat: bool = False):
        pass

    def set_busy(self):
        pass

    def set_done(self, ok: bool, elapsed_ms: int = 0):
        pass

    def set_error(self, code: int = 99, detail: str = ""):
        pass

    def set_shutdown(self):
        pass

    def get_state_snapshot(self) -> Dict[str, Any]:
        return {
            "command": 0,
            "status": 0,
            "result": 0,
            "heartbeat": 0,
            "vision_ready": 0,
            "error_code": 0,
            "error_detail": "",
            "last_inspect_at": "",
            "last_inspect_elapsed_ms": 0,
            "serial_open": False,
            "comm_fault_active": False,
            "comm_reset_required": False,
            "ever_connected": False,
            "link_verified": False,
            "ever_link_verified": False,
            "reconnect_attempt_count": 0,
            "last_connect_epoch": None,
            "last_disconnect_epoch": None,
            "last_valid_rx_epoch": None,
            "slave_id": 0,
            "port": "",
            "rx_count": 0,
            "tx_count": 0,
            "last_rx_epoch": None,
            "last_tx_epoch": None,
            "last_rx_summary": "",
            "last_tx_summary": "",
            "last_rx_hex": "",
            "last_tx_hex": "",
            "heartbeat_last_change_epoch": None,
        }

    def is_connected(self) -> bool:
        return False
    
    def poll_comm_event(self) -> Optional[Dict[str, Any]]:
        return None

    def prepare_reset(self, reset_heartbeat: bool = False):
        pass

    def set_recovering(self, code: int = 0, detail: str = ""):
        pass

    def complete_comm_recovery(self) -> bool:
        return False

    def trace_application_event(
        self,
        event: str,
        summary: str,
        extra: Optional[Dict[str, Any]] = None,
    ):
        pass

    def get_recent_events(self, limit: int = 20):
        return []
    
    def set_vision_ready(self, ready: bool):
        pass

    def set_service_vision_ready(self):
        pass


class ModbusRtuSlaveController:
    CMD_NONE = 0
    CMD_PREPARE = 1
    CMD_INSPECT = 2
    CMD_RESET = 3
    CMD_SHUTDOWN = 8
    CMD_EMERGENCY = 9

    STATUS_READY = 0
    STATUS_BUSY = 1
    STATUS_DONE = 2
    STATUS_ERROR = 3
    STATUS_SHUTDOWN = 8

    RESULT_NONE = 0
    RESULT_OK = 1
    RESULT_NG = 2

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", False))

        serial_cfg = cfg.get("serial", {}) or {}
        modbus_cfg = cfg.get("modbus", {}) or {}
        regs_cfg = cfg.get("registers", {}) or {}
        heartbeat_cfg = cfg.get("heartbeat", {}) or {}
        vision_ready_cfg = cfg.get("vision_ready", {}) or {}

        self.port = str(serial_cfg.get("port", "/dev/ttyUSB0"))
        self.baudrate = int(serial_cfg.get("baudrate", 9600))
        self.bytesize = int(serial_cfg.get("bytesize", 8))
        self.parity = str(serial_cfg.get("parity", "N"))
        self.stopbits = int(serial_cfg.get("stopbits", 1))
        self.timeout = max(0.01, float(serial_cfg.get("timeout", 0.05)))

        self.reconnect_interval_sec = max(
            0.1,
            float(serial_cfg.get("reconnect_interval_sec", 1.0)),
        )
        self.require_reset_after_runtime_disconnect = bool(
            serial_cfg.get("require_reset_after_runtime_disconnect", True)
        )
        self.require_initial_rx = bool(
            serial_cfg.get("require_initial_rx", True)
        )
        self.rx_watchdog_sec = max(
            0.0,
            float(serial_cfg.get("rx_watchdog_sec", 5.0)),
        )
        self.rx_watchdog_startup_grace_sec = max(
            self.rx_watchdog_sec,
            float(serial_cfg.get("rx_watchdog_startup_grace_sec", 15.0)),
        )

        self.slave_id = int(modbus_cfg.get("slave_id", 1))

        self.reg_command = int(regs_cfg.get("command", 200))
        self.reg_status = int(regs_cfg.get("status", 201))
        self.reg_result = int(regs_cfg.get("result", 202))
        self.reg_heartbeat = int(regs_cfg.get("heartbeat", 203))

        self.reg_vision_ready = int(regs_cfg.get("vision_ready", 204))

        self.vision_ready_value = int(
            vision_ready_cfg.get("ready_value", 100)
        )
        self.vision_not_ready_value = int(
            vision_ready_cfg.get("not_ready_value", 0)
        )

        if not 1 <= self.slave_id <= 247:
            raise ValueError(
                f"modbus slave_id must be 1..247: {self.slave_id}"
            )

        register_addresses = [
            self.reg_command,
            self.reg_status,
            self.reg_result,
            self.reg_heartbeat,
            self.reg_vision_ready,
        ]

        if any(reg < 0 or reg > 0xFFFF for reg in register_addresses):
            raise ValueError(
                f"modbus register address out of range: {register_addresses}"
            )

        if len(set(register_addresses)) != len(register_addresses):
            raise ValueError(
                f"modbus register addresses must be unique: {register_addresses}"
            )

        max_reg = max(
            self.reg_command,
            self.reg_status,
            self.reg_result,
            self.reg_heartbeat,
            self.reg_vision_ready,
        )

        self.register_count = max_reg + 1
        self.regs = [0] * self.register_count

        self.last_error_code = 0
        self.last_error_detail = ""
        self.last_inspect_at = ""
        self.last_inspect_elapsed_ms = 0

        self.heartbeat_interval_sec = max(
            0.1,
            float(heartbeat_cfg.get("interval_sec", 0.5)),
        )
        self.heartbeat_max = max(1, int(heartbeat_cfg.get("max_value", 9999)))
        self._last_heartbeat_ts = 0.0

        self._last_cmd_seen = 0

        self._lock = threading.Lock()
        self._ser = None
        self._thread = None
        self._heartbeat_thread = None
        self._stop = threading.Event()

        self._comm_event_lock = threading.Lock()
        self._comm_events = []
        self._comm_fault_active = False
        self._comm_reset_required = False
        self._ever_connected = False
        self._link_verified = False
        self._ever_link_verified = False
        self._last_reconnect_ts = 0.0
        self._reconnect_attempt_count = 0
        self._serial_open_epoch = None
        self._last_connect_epoch = None
        self._last_disconnect_epoch = None
        self._last_valid_rx_epoch = None

        trace_cfg = cfg.get("live_trace", {}) or {}
        self.trace_enabled = bool(trace_cfg.get("enabled", False))
        self.trace_include_state = bool(
            trace_cfg.get("include_state_changes", True)
        )
        self.trace_include_heartbeat = bool(
            trace_cfg.get("include_heartbeat", True)
        )
        self.trace_max_bytes = max(
            1024 * 1024,
            int(trace_cfg.get("max_bytes", 5 * 1024 * 1024)),
        )
        self.trace_keep_files = max(
            1,
            int(trace_cfg.get("keep_files", 2)),
        )

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        trace_path = str(
            trace_cfg.get(
                "path",
                os.path.join("data", "logs", "plc_live.jsonl"),
            )
        )
        if not os.path.isabs(trace_path):
            trace_path = os.path.join(project_root, trace_path)

        self.trace_path = os.path.abspath(trace_path)
        self._trace_lock = threading.Lock()
        self._trace_fp = None
        self._trace_seq = 0
        self._trace_failed = False

        # 메인 비전 UI에서 직접 표시하는 인메모리 Live 상태.
        # trace 파일 저장 여부와 무관하게 유지한다.
        self._live_event_lock = threading.Lock()
        self._live_events = []
        self._rx_count = 0
        self._tx_count = 0
        self._last_rx_epoch = None
        self._last_tx_epoch = None
        self._last_rx_summary = ""
        self._last_tx_summary = ""
        self._last_rx_hex = ""
        self._last_tx_hex = ""

    def _command_name(self, value: int) -> str:
        names = {
            self.CMD_NONE: "NONE",
            self.CMD_PREPARE: "PREPARE",
            self.CMD_INSPECT: "INSPECT",
            self.CMD_RESET: "RESET",
            self.CMD_SHUTDOWN: "SHUTDOWN",
            self.CMD_EMERGENCY: "EMERGENCY",
        }
        return names.get(int(value), f"UNKNOWN({int(value)})")

    def _frame_summary(self, frame: bytes, direction: str) -> str:
        if not frame:
            return "EMPTY"

        if len(frame) < 2:
            return f"SHORT len={len(frame)}"

        addr = frame[0]
        func = frame[1]

        if func & 0x80:
            code = frame[2] if len(frame) > 2 else -1
            return (
                f"slave={addr} EXCEPTION "
                f"fc=0x{func & 0x7F:02X} code={code}"
            )

        if direction == "RX":
            if func in (3, 4) and len(frame) >= 6:
                start = (frame[2] << 8) | frame[3]
                qty = (frame[4] << 8) | frame[5]
                return f"slave={addr} FC{func:02d} READ D{start} qty={qty}"

            if func == 6 and len(frame) >= 6:
                reg = (frame[2] << 8) | frame[3]
                value = (frame[4] << 8) | frame[5]
                suffix = ""
                if reg == self.reg_command:
                    suffix = f" {self._command_name(value)}"
                return (
                    f"slave={addr} FC06 WRITE D{reg}={value}{suffix}"
                )

            if func == 16 and len(frame) >= 7:
                start = (frame[2] << 8) | frame[3]
                qty = (frame[4] << 8) | frame[5]
                return (
                    f"slave={addr} FC16 WRITE_MULTI "
                    f"D{start} qty={qty}"
                )

        if direction == "TX":
            if func in (3, 4) and len(frame) >= 3:
                byte_count = frame[2]
                vals = []
                end = min(3 + byte_count, max(3, len(frame) - 2))
                for pos in range(3, end, 2):
                    if pos + 1 < end:
                        vals.append((frame[pos] << 8) | frame[pos + 1])
                return (
                    f"slave={addr} FC{func:02d} READ_RESPONSE "
                    f"values={vals}"
                )

            if func == 6 and len(frame) >= 6:
                reg = (frame[2] << 8) | frame[3]
                value = (frame[4] << 8) | frame[5]
                return f"slave={addr} FC06 WRITE_ACK D{reg}={value}"

            if func == 16 and len(frame) >= 6:
                start = (frame[2] << 8) | frame[3]
                qty = (frame[4] << 8) | frame[5]
                return (
                    f"slave={addr} FC16 WRITE_MULTI_ACK "
                    f"D{start} qty={qty}"
                )

        return f"slave={addr} fc=0x{func:02X} len={len(frame)}"

    def _open_trace_locked(self, rotate_existing: bool = False):
        if not self.trace_enabled:
            return

        os.makedirs(os.path.dirname(self.trace_path), exist_ok=True)

        if rotate_existing and os.path.exists(self.trace_path):
            for index in range(self.trace_keep_files, 0, -1):
                src = (
                    self.trace_path
                    if index == 1
                    else f"{self.trace_path}.{index - 1}"
                )
                dst = f"{self.trace_path}.{index}"

                if os.path.exists(src):
                    try:
                        if os.path.exists(dst):
                            os.remove(dst)
                        os.replace(src, dst)
                    except Exception:
                        pass

        self._trace_fp = open(
            self.trace_path,
            "a",
            encoding="utf-8",
            buffering=1,
        )

    def _rotate_trace_locked(self):
        if self._trace_fp is not None:
            try:
                self._trace_fp.close()
            except Exception:
                pass
            self._trace_fp = None

        self._open_trace_locked(rotate_existing=True)

    def _start_trace(self):
        if not self.trace_enabled:
            return

        try:
            with self._trace_lock:
                self._open_trace_locked(rotate_existing=True)
        except Exception as e:
            self._trace_failed = True
            print(f"[PLC TRACE] start failed: {e}")
            return

        self._trace_event(
            direction="SYSTEM",
            event="TRACE_START",
            summary=f"trace started path={self.trace_path}",
        )

    def _stop_trace(self):
        if not self.trace_enabled:
            return


        self._trace_event(
            direction="SYSTEM",
            event="TRACE_STOP",
            summary="trace stopped",
        )

        with self._trace_lock:
            if self._trace_fp is not None:
                try:
                    self._trace_fp.close()
                except Exception:
                    pass
                self._trace_fp = None

    def _trace_event(
        self,
        direction: str,
        event: str,
        summary: str,
        frame: Optional[bytes] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        with self._lock:
            registers = {
                "D200": int(self.regs[self.reg_command]),
                "D201": int(self.regs[self.reg_status]),
                "D202": int(self.regs[self.reg_result]),
                "D203": int(self.regs[self.reg_heartbeat]),
                "D204": int(self.regs[self.reg_vision_ready]),
            }

        raw = bytes(frame or b"")
        self._trace_seq += 1

        record = {
            "seq": self._trace_seq,
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3],
            "epoch": time.time(),
            "direction": str(direction),
            "event": str(event),
            "summary": str(summary),
            "hex": raw.hex(" "),
            "length": len(raw),
            "crc_ok": _check_crc(raw) if len(raw) >= 4 else None,
            "registers": registers,
            "serial_open": self.is_connected(),
            "comm_fault_active": bool(self._comm_fault_active),
            "error_code": int(self.last_error_code),
            "error_detail": str(self.last_error_detail),
            "last_inspect_at": str(self.last_inspect_at),
            "last_inspect_elapsed_ms": int(self.last_inspect_elapsed_ms),
        }

        if extra:
            record.update(extra)

        direction_text = str(direction)
        with self._live_event_lock:
            self._live_events.append(dict(record))
            if len(self._live_events) > 120:
                del self._live_events[:-120]

            if direction_text == "RX":
                self._rx_count += 1
                self._last_rx_epoch = record["epoch"]
                self._last_rx_summary = str(summary)
                self._last_rx_hex = str(record.get("hex", ""))
            elif direction_text == "TX":
                self._tx_count += 1
                self._last_tx_epoch = record["epoch"]
                self._last_tx_summary = str(summary)
                self._last_tx_hex = str(record.get("hex", ""))

        if not self.trace_enabled or self._trace_failed:
            return

        try:
            line = json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
            ) + "\n"

            with self._trace_lock:
                if self._trace_fp is None:
                    self._open_trace_locked(rotate_existing=False)

                if self._trace_fp is None:
                    return

                try:
                    current_size = self._trace_fp.tell()
                except Exception:
                    current_size = 0

                if current_size >= self.trace_max_bytes:
                    self._rotate_trace_locked()

                self._trace_fp.write(line)
                self._trace_fp.flush()

        except Exception as e:
            self._trace_failed = True
            print(f"[PLC TRACE] write failed: {e}")

    def trace_application_event(
        self,
        event: str,
        summary: str,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self._trace_event(
            direction="APP",
            event=str(event),
            summary=str(summary),
            extra=extra,
        )

    def get_recent_events(self, limit: int = 20):
        limit = max(1, min(120, int(limit)))
        with self._live_event_lock:
            return [dict(item) for item in self._live_events[-limit:]]

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            print("[PLC] already started")
            return

        self._trace_failed = False
        self._start_trace()
        self.reset_to_ready(reset_heartbeat=True)

        self.set_vision_ready(False)

        self._stop.clear()
        self._last_reconnect_ts = 0.0

        with self._comm_event_lock:
            self._comm_events.clear()

        connected = self._open_serial(initial=True)

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )
        self._thread.start()

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
        )
        self._heartbeat_thread.start()

        if connected:
            print(
                f"[PLC] modbus rtu slave started "
                f"port={self.port} "
                f"baudrate={self.baudrate} "
                f"slave_id={self.slave_id}"
            )
        else:
            print(
                f"[PLC] serial unavailable, reconnecting every "
                f"{self.reconnect_interval_sec:.1f} sec"
            )

    def _push_comm_event(
        self,
        error_code: int,
        message: str,
        *,
        event_type: str = "fault",
        reset_required: Optional[bool] = None,
    ):
        event = {
            "event_type": str(event_type),
            "error_code": int(error_code),
            "message": str(message),
            "reset_required": bool(
                self._comm_reset_required
                if reset_required is None
                else reset_required
            ),
            "serial_open": bool(self.is_connected()),
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3],
        }

        with self._comm_event_lock:
            self._comm_events.append(event)

            if len(self._comm_events) > 20:
                self._comm_events.pop(0)

    def poll_comm_event(self) -> Optional[Dict[str, Any]]:
        with self._comm_event_lock:
            if not self._comm_events:
                return None

            return self._comm_events.pop(0)

    def _close_serial(self):
        ser = self._ser
        self._ser = None

        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def _mark_comm_error(
        self,
        error_code: int,
        message: str,
    ):
        self._trace_event(
            direction="SYSTEM",
            event="SERIAL_ERROR",
            summary=str(message),
            extra={"error_code": int(error_code)},
        )
        self._close_serial()

        # 정상 종료 중 발생한 close/read 예외는 통신 장애로 기록하지 않는다.
        if self._stop.is_set():
            return

        if not self._comm_fault_active:
            self._comm_fault_active = True
            self._comm_reset_required = bool(
                self._ever_link_verified
                and self.require_reset_after_runtime_disconnect
            )
            self._last_disconnect_epoch = time.time()

            self.set_recovering(
                code=error_code,
                detail=message,
            )
            self._push_comm_event(
                error_code=error_code,
                message=message,
                event_type="fault",
                reset_required=self._comm_reset_required,
            )

        print(
            f"[PLC] communication error: {message} "
            f"reset_required={self._comm_reset_required}"
        )

    def _open_serial(self, initial: bool = False) -> bool:
        if self._stop.is_set():
            return False

        if serial is None:
            self._mark_comm_error(
                error_code=70,
                message="pyserial is not installed",
            )
            return False

        try:
            self._reconnect_attempt_count += 1
            opened_serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=self.bytesize,
                parity=self.parity,
                stopbits=self.stopbits,
                timeout=self.timeout,
            )

            if hasattr(opened_serial, "is_open") and not opened_serial.is_open:
                raise RuntimeError("serial port object is not open")

            try:
                opened_serial.reset_input_buffer()
                opened_serial.reset_output_buffer()
            except Exception:
                pass

            self._ser = opened_serial
            now = time.time()
            self._serial_open_epoch = now
            self._last_connect_epoch = now
            self._last_valid_rx_epoch = None
            self._link_verified = False
            self._reconnect_attempt_count = 0

            was_fault = self._comm_fault_active
            first_connection = not self._ever_connected
            self._ever_connected = True
            self._last_cmd_seen = self.CMD_NONE

            if was_fault and self._comm_reset_required:
                self.set_recovering(
                    code=self.last_error_code or 71,
                    detail=(
                        self.last_error_detail
                        or "PLC communication restored; D200=3 reset required"
                    ),
                )
                print(
                    f"[PLC] serial reconnected: {self.port} "
                    f"- D201 remains BUSY until D200=3"
                )
                self._push_comm_event(
                    error_code=self.last_error_code or 71,
                    message="PLC serial communication restored; D200=3 reset required",
                    event_type="reconnected_reset_required",
                    reset_required=True,
                )
            else:
                self._comm_fault_active = False
                self._comm_reset_required = False
                existing_non_comm_error = bool(
                    self.last_error_code not in (0, 70, 71)
                )
                if self.last_error_code in (70, 71):
                    self.last_error_code = 0
                    self.last_error_detail = ""
                if was_fault:
                    if not existing_non_comm_error:
                        self.set_busy()
                    print(
                        f"[PLC] serial reconnected automatically: {self.port}"
                    )
                    self._push_comm_event(
                        error_code=(
                            int(self.last_error_code)
                            if existing_non_comm_error
                            else 0
                        ),
                        message=(
                            "PLC serial restored; existing vision error remains"
                            if existing_non_comm_error
                            else "PLC serial communication restored automatically"
                        ),
                        event_type=(
                            "reconnected_existing_error"
                            if existing_non_comm_error
                            else "reconnected"
                        ),
                        reset_required=False,
                    )

            self._trace_event(
                direction="SYSTEM",
                event="SERIAL_RECONNECTED" if was_fault else "SERIAL_OPEN",
                summary=(
                    f"port={self.port} baud={self.baudrate} "
                    f"slave_id={self.slave_id} first_connection={first_connection} "
                    f"reset_required={self._comm_reset_required}"
                ),
            )

            return True

        except Exception as e:
            error_code = 70 if initial else 71

            self._mark_comm_error(
                error_code=error_code,
                message=(
                    f"serial port open failed: "
                    f"port={self.port}, error={e}"
                ),
            )
            return False
        
    def stop(self):
        self.set_vision_ready(False)
        self._stop.set()

        # read() 대기를 즉시 해제한 뒤 스레드를 종료한다.
        self._close_serial()

        join_timeout = max(1.0, self.timeout + 0.5)

        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            self._thread = None

        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=join_timeout)
            self._heartbeat_thread = None

        self._stop_trace()
        print("[PLC] stopped")

    def tick(self):
        now = time.time()
        if (now - self._last_heartbeat_ts) < self.heartbeat_interval_sec:
            return

        self._last_heartbeat_ts = now

        current = self._get_reg(self.reg_heartbeat)
        next_val = current + 1

        if next_val > self.heartbeat_max:
            next_val = 0

        self._set_reg(self.reg_heartbeat, next_val)

    def _heartbeat_loop(self):
        while not self._stop.wait(
            self.heartbeat_interval_sec
        ):
            self.tick()

    def poll_command(self) -> Optional[str]:
        cmd = self._get_reg(self.reg_command)

        if cmd == self.CMD_NONE:
            self._last_cmd_seen = self.CMD_NONE
            return None

        if cmd == self._last_cmd_seen:
            return None

        self._last_cmd_seen = cmd

        if cmd == self.CMD_PREPARE:
            print("[PLC] command received: PREPARE_REQUEST")
            return "prepare"

        if cmd == self.CMD_INSPECT:
            print("[PLC] command received: INSPECT_REQUEST")
            return "inspect"
        
        if cmd == self.CMD_RESET:
            print("[PLC] command received: RESET_REQUEST")
            return "reset"

        if cmd == self.CMD_SHUTDOWN:
            print("[PLC] command received: SHUTDOWN_REQUEST")
            return "shutdown"

        if cmd == self.CMD_EMERGENCY:
            print("[PLC] command received: EMERGENCY_STOP")
            return "emergency"

        print(f"[PLC] unknown command: {cmd}")
        return "command_error"

    def set_idle(self):
        # D202 결과는 D200=3이 들어오기 전까지 유지
        self._set_reg(self.reg_status, self.STATUS_READY)

    def prepare_reset(
        self,
        reset_heartbeat: bool = False,
    ):
        # D200은 PLC가 직접 0으로 복귀
        # D202는 Reset 명령 즉시 초기화
        self._set_reg(
            self.reg_result,
            self.RESULT_NONE,
        )

        if reset_heartbeat:
            self._set_reg(self.reg_heartbeat, 0)
            self._last_heartbeat_ts = time.time()

        print(
            f"[PLC] reset preparation "
            f"result_cleared=True "
            f"heartbeat_reset={bool(reset_heartbeat)}"
        )

    def reset_to_ready(self, reset_heartbeat: bool = False):
        previous_error_code = int(self.last_error_code)
        previous_error_detail = str(self.last_error_detail)

        self.last_error_code = 0
        self.last_error_detail = ""
        self.last_inspect_at = ""
        self.last_inspect_elapsed_ms = 0

        self._set_reg(self.reg_status, self.STATUS_READY)
        self._set_reg(self.reg_result, self.RESULT_NONE)

        if reset_heartbeat:
            self._set_reg(self.reg_heartbeat, 0)
            self._last_heartbeat_ts = time.time()

        if previous_error_code != 0 or previous_error_detail:
            self._trace_event(
                direction="ERROR",
                event="ERROR_CLEARED",
                summary=(
                    f"error cleared code={previous_error_code} "
                    f"detail={previous_error_detail}"
                ),
                extra={
                    "cleared_error_code": previous_error_code,
                    "cleared_error_detail": previous_error_detail,
                },
            )

        print(
            f"[PLC] reset to ready "
            f"heartbeat_reset={bool(reset_heartbeat)}"
        )

    def set_busy(self):
        # 이전 D202 결과는 Reset 명령 전까지 유지
        self._set_reg(self.reg_status, self.STATUS_BUSY)

    def set_done(self, ok: bool, elapsed_ms: int = 0):
        self._set_reg(self.reg_status, self.STATUS_DONE)
        self._set_reg(
            self.reg_result,
            self.RESULT_OK if ok else self.RESULT_NG,
        )

        self.last_error_code = 0
        self.last_error_detail = ""
        self.last_inspect_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_inspect_elapsed_ms = max(0, int(elapsed_ms))

        print(
            f"[PLC] result set: {'OK' if ok else 'NG'} "
            f"inspect_at={self.last_inspect_at} "
            f"elapsed_ms={self.last_inspect_elapsed_ms}"
        )

    def set_recovering(self, code: int = 0, detail: str = ""):
        self.set_vision_ready(False)
        self.last_error_code = int(code)
        self.last_error_detail = str(detail or "")
        self._set_reg(self.reg_status, self.STATUS_BUSY)
        self._trace_event(
            direction="SYSTEM",
            event="RECOVERY_BUSY",
            summary=(
                f"code={self.last_error_code} "
                f"detail={self.last_error_detail}"
            ),
            extra={
                "error_code": int(self.last_error_code),
                "error_detail": str(self.last_error_detail),
                "comm_reset_required": bool(self._comm_reset_required),
            },
        )

    def complete_comm_recovery(self) -> bool:
        if not self.is_connected():
            return False
        if self.require_initial_rx and not self._link_verified:
            return False

        previous_code = int(self.last_error_code)
        previous_detail = str(self.last_error_detail)
        self._comm_fault_active = False
        self._comm_reset_required = False
        if previous_code in (70, 71):
            self.last_error_code = 0
            self.last_error_detail = ""

        self._trace_event(
            direction="SYSTEM",
            event="COMM_RECOVERY_COMPLETED",
            summary=(
                f"serial communication recovery completed "
                f"previous_code={previous_code}"
            ),
            extra={
                "previous_error_code": previous_code,
                "previous_error_detail": previous_detail,
            },
        )
        return True

    def set_error(self, code: int = 99, detail: str = ""):
        self.set_vision_ready(False)
        # D202의 기존 OK/NG 값은 유지하고 D201만 Error로 변경
        self.last_error_code = int(code)
        self.last_error_detail = str(detail or "")

        self._set_reg(self.reg_status, self.STATUS_ERROR)

        self._trace_event(
            direction="ERROR",
            event="ERROR_SET",
            summary=(
                f"code={self.last_error_code} "
                f"detail={self.last_error_detail}"
            ),
            extra={
                "error_code": int(self.last_error_code),
                "error_detail": str(self.last_error_detail),
            },
        )

        print(
            f"[PLC] error status "
            f"code={self.last_error_code} "
            f"detail={self.last_error_detail}"
        )

    def set_shutdown(self):
        self.set_vision_ready(False)
        # 종료 상태는 D201=8 하나만 사용
        self._set_reg(self.reg_status, self.STATUS_SHUTDOWN)
        print("[PLC] shutdown status set: D201=8")

    def set_vision_ready(self, ready: bool):
        value = (
            self.vision_ready_value
            if bool(ready)
            else self.vision_not_ready_value
        )

        current = self._get_reg(self.reg_vision_ready)
        if current == value:
            return

        self._set_reg(self.reg_vision_ready, value)
        print(f"[PLC] vision ready D{self.reg_vision_ready}={value}")

    def set_service_vision_ready(self):
        """Service-only PLC handshake test.

        Clear D201/D202 first, then assert D204=100. This is intentionally
        separate from the production readiness path.
        """
        self._set_reg(self.reg_status, self.STATUS_READY)
        self._set_reg(self.reg_result, self.RESULT_NONE)
        self.set_vision_ready(True)
        print(
            f"[PLC SERVICE] D{self.reg_status}=0 "
            f"D{self.reg_result}=0 "
            f"D{self.reg_vision_ready}={self.vision_ready_value}"
        )

    def get_state_snapshot(self) -> Dict[str, Any]:
        return {
            "command": self._get_reg(self.reg_command),
            "status": self._get_reg(self.reg_status),
            "result": self._get_reg(self.reg_result),
            "heartbeat": self._get_reg(self.reg_heartbeat),
            "error_code": int(self.last_error_code),
            "error_detail": str(self.last_error_detail),
            "last_inspect_at": str(self.last_inspect_at),
            "last_inspect_elapsed_ms": int(self.last_inspect_elapsed_ms),
            "serial_open": self.is_connected(),
            "comm_fault_active": bool(self._comm_fault_active),
            "comm_reset_required": bool(self._comm_reset_required),
            "ever_connected": bool(self._ever_connected),
            "link_verified": bool(self._link_verified),
            "ever_link_verified": bool(self._ever_link_verified),
            "reconnect_attempt_count": int(self._reconnect_attempt_count),
            "last_connect_epoch": self._last_connect_epoch,
            "last_disconnect_epoch": self._last_disconnect_epoch,
            "last_valid_rx_epoch": self._last_valid_rx_epoch,
            "slave_id": int(self.slave_id),
            "port": str(self.port),
            "rx_count": int(self._rx_count),
            "tx_count": int(self._tx_count),
            "last_rx_epoch": self._last_rx_epoch,
            "last_tx_epoch": self._last_tx_epoch,
            "last_rx_summary": str(self._last_rx_summary),
            "last_tx_summary": str(self._last_tx_summary),
            "last_rx_hex": str(self._last_rx_hex),
            "last_tx_hex": str(self._last_tx_hex),
            "vision_ready": self._get_reg(self.reg_vision_ready),
            "heartbeat_last_change_epoch": float(self._last_heartbeat_ts)
            if self._last_heartbeat_ts > 0
            else None,
        }

    def is_connected(self) -> bool:
        ser = self._ser
        if ser is None:
            return False

        try:
            return bool(getattr(ser, "is_open", True))
        except Exception:
            return False

    def _get_reg(self, idx: int) -> int:
        with self._lock:
            if 0 <= idx < len(self.regs):
                return int(self.regs[idx])
            return 0

    def _set_reg(self, idx: int, val: int):
        old_value = None
        new_value = int(val) & 0xFFFF

        with self._lock:
            if 0 <= idx < len(self.regs):
                old_value = int(self.regs[idx])
                self.regs[idx] = new_value

        if old_value is None or old_value == new_value:
            return

        if not self.trace_include_state:
            return

        if idx == self.reg_heartbeat and not self.trace_include_heartbeat:
            return

        self._trace_event(
            direction="STATE",
            event="REGISTER_CHANGE",
            summary=f"D{idx} {old_value}->{new_value}",
            extra={
                "register": int(idx),
                "old_value": int(old_value),
                "new_value": int(new_value),
            },
        )

    def _read_exact(self, n: int) -> Optional[bytes]:
        if n <= 0:
            return b""

        data = bytearray()
        deadline = time.monotonic() + self.timeout

        while len(data) < n and not self._stop.is_set():
            ser = self._ser
            if ser is None:
                return None

            try:
                chunk = ser.read(n - len(data))
            except Exception as e:
                self._mark_comm_error(
                    error_code=71,
                    message=f"serial read failed: {e}",
                )
                return None

            if chunk:
                data.extend(chunk)
                continue

            if time.monotonic() >= deadline:
                break

        if len(data) != n:
            return None

        return bytes(data)

    def _loop(self):
        while not self._stop.is_set():

            if self._ser is None:
                now = time.time()

                if (
                    now - self._last_reconnect_ts
                    >= self.reconnect_interval_sec
                ):
                    self._last_reconnect_ts = now
                    self._open_serial(initial=False)

                time.sleep(0.05)
                continue

            if self.rx_watchdog_sec > 0:
                now = time.time()
                reference = (
                    self._last_valid_rx_epoch
                    if self._last_valid_rx_epoch is not None
                    else self._serial_open_epoch
                )
                watchdog_limit = (
                    self.rx_watchdog_sec
                    if self._last_valid_rx_epoch is not None
                    else self.rx_watchdog_startup_grace_sec
                )
                if (
                    reference is not None
                    and (now - float(reference)) > watchdog_limit
                ):
                    self._mark_comm_error(
                        error_code=71,
                        message=(
                            "PLC RX watchdog timeout: "
                            f"no valid Modbus request for {now - float(reference):.2f} sec"
                        ),
                    )
                    continue

            try:
                b1 = self._read_exact(1)
                if not b1:
                    continue

                b2 = self._read_exact(1)
                if not b2:
                    continue

                addr = b1[0]
                func = b2[0]

                if func in (3, 4, 6):
                    rest = self._read_exact(6)
                    if not rest:
                        continue
                    frame = b1 + b2 + rest

                elif func == 16:
                    head = self._read_exact(5)
                    if not head:
                        continue
                    byte_count = int(head[4])
                    tail = self._read_exact(byte_count + 2)
                    if not tail:
                        continue
                    frame = b1 + b2 + head + tail

                else:
                    self._trace_event(
                        direction="RX",
                        event="UNSUPPORTED_FUNCTION",
                        summary=(
                            f"slave={addr} unsupported fc=0x{func:02X}"
                        ),
                        frame=b1 + b2,
                    )

                    # 지원하지 않는 function code의 나머지 바이트가
                    # 다음 프레임으로 섞이지 않도록 입력 버퍼를 비운다.
                    try:
                        if self._ser is not None:
                            self._ser.reset_input_buffer()
                    except Exception:
                        pass
                    continue

                self._trace_event(
                    direction="RX",
                    event="FRAME",
                    summary=self._frame_summary(frame, "RX"),
                    frame=frame,
                )

                if not _check_crc(frame):
                    self._trace_event(
                        direction="SYSTEM",
                        event="CRC_ERROR",
                        summary="received frame CRC mismatch",
                        frame=frame,
                    )
                    continue

                if addr not in (self.slave_id, 0):
                    self._trace_event(
                        direction="SYSTEM",
                        event="SLAVE_ID_MISMATCH",
                        summary=(
                            f"received slave={addr}, expected={self.slave_id}"
                        ),
                        frame=frame,
                    )
                    continue

                first_verified_rx = not self._link_verified
                self._last_valid_rx_epoch = time.time()
                self._link_verified = True
                self._ever_link_verified = True

                if first_verified_rx:
                    self._trace_event(
                        direction="SYSTEM",
                        event="PLC_LINK_VERIFIED",
                        summary="first valid Modbus request received",
                    )
                    if not self._comm_reset_required:
                        self._push_comm_event(
                            error_code=0,
                            message="PLC link verified by valid Modbus request",
                            event_type="link_verified",
                            reset_required=False,
                        )

                if func in (3, 4):
                    self._handle_read_holding(addr, frame)

                elif func == 6:
                    self._handle_write_single(addr, frame)

                elif func == 16:
                    self._handle_write_multiple(addr, frame)

            except Exception as e:
                print("[PLC] loop error:", e)
                time.sleep(0.1)

    def _write_response(self, addr: int, payload: bytes):
        if addr == 0:
            return

        ser = self._ser
        if ser is None:
            return

        frame = _append_crc(payload)

        try:
            written = ser.write(frame)
            if written is not None and int(written) != len(frame):
                raise IOError(
                    f"short serial write: {written}/{len(frame)} bytes"
                )
            ser.flush()
            self._trace_event(
                direction="TX",
                event="FRAME",
                summary=self._frame_summary(frame, "TX"),
                frame=frame,
            )
        except Exception as e:
            self._trace_event(
                direction="TX",
                event="WRITE_FAILED",
                summary=f"serial write failed: {e}",
                frame=frame,
            )
            self._mark_comm_error(
                error_code=71,
                message=f"serial write failed: {e}",
            )

    def _write_exception(self, addr: int, func: int, code: int):
        if addr == 0:
            return

        self._write_response(addr, bytes([addr, func | 0x80, code]))

    def _handle_read_holding(self, addr: int, frame: bytes):
        func = frame[1]
        start = (frame[2] << 8) | frame[3]
        qty = (frame[4] << 8) | frame[5]

        if qty <= 0 or qty > 32:
            self._write_exception(addr, func, 3)
            return

        allowed_registers = {
            self.reg_command,
            self.reg_status,
            self.reg_result,
            self.reg_heartbeat,
            self.reg_vision_ready,
        }

        requested_registers = range(
            start,
            start + qty,
        )

        if any(
            reg not in allowed_registers
            for reg in requested_registers
        ):
            self._write_exception(addr, func, 2)
            return

        with self._lock:
            vals = self.regs[start:start + qty]

        data = bytearray()
        for v in vals:
            data.append((int(v) >> 8) & 0xFF)
            data.append(int(v) & 0xFF)

        payload = bytes([addr, func, len(data)]) + bytes(data)
        self._write_response(addr, payload)

    def _handle_write_single(self, addr: int, frame: bytes):
        func = frame[1]
        reg = (frame[2] << 8) | frame[3]
        val = (frame[4] << 8) | frame[5]

        if reg < 0 or reg >= len(self.regs):
            self._write_exception(addr, func, 2)
            return

        if reg != self.reg_command:
            self._write_exception(addr, func, 2)
            return

        self._set_reg(reg, val)

        if addr != 0:
            self._write_response(addr, frame[:-2])

    def _handle_write_multiple(self, addr: int, frame: bytes):
        func = frame[1]
        start = (frame[2] << 8) | frame[3]
        qty = (frame[4] << 8) | frame[5]
        byte_count = frame[6]

        if qty <= 0 or qty > 32 or byte_count != qty * 2:
            self._write_exception(addr, func, 3)
            return

        if start < 0 or start + qty > len(self.regs):
            self._write_exception(addr, func, 2)
            return

        if start != self.reg_command or qty != 1:
            self._write_exception(addr, func, 2)
            return
        
        pos = 7
        for i in range(qty):
            val = (frame[pos] << 8) | frame[pos + 1]
            self._set_reg(start + i, val)
            pos += 2

        payload = bytes([
            addr,
            func,
            (start >> 8) & 0xFF,
            start & 0xFF,
            (qty >> 8) & 0xFF,
            qty & 0xFF,
        ])

        self._write_response(addr, payload)


def create_plc_controller(cfg: Dict[str, Any]):
    if not bool(cfg.get("enabled", False)):
        return DisabledPlcController(cfg)

    backend = str(cfg.get("backend", "modbus_rtu_slave")).strip().lower()

    if backend == "modbus_rtu_slave":
        return ModbusRtuSlaveController(cfg)

    raise ValueError(f"unsupported plc backend: {backend}")