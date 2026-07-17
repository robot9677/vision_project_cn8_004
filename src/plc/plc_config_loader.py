import copy
import json
import os
from typing import Any, Dict


DEFAULT_PLC_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "backend": "modbus_rtu_slave",

    "serial": {
        "port": "/dev/ttyUSB0",
        "baudrate": 9600,
        "bytesize": 8,
        "parity": "N",
        "stopbits": 1,
        "timeout": 0.1,
        "reconnect_interval_sec": 1.0,
        "require_reset_after_runtime_disconnect": True,
        "require_initial_rx": True,
        "rx_watchdog_sec": 5.0,
        "rx_watchdog_startup_grace_sec": 15.0,
    },

    "modbus": {
        "slave_id": 1,
    },

    "registers": {
        "command": 200,
        "status": 201,
        "result": 202,
        "heartbeat": 203,
    },

    "heartbeat": {
        "interval_sec": 0.5,
        "max_value": 9999,
    },

    "live_trace": {
        "enabled": False,
        "path": "data/logs/plc_live.jsonl",
        "max_bytes": 5242880,
        "keep_files": 2,
        "include_state_changes": True,
        "include_heartbeat": False,
    },

    "diagnostics": {
        "enabled": True,
        "index_path": "diagnostics/diagnostics_index.jsonl",
        "index_max_bytes": 2097152,
        "index_keep_files": 2,
        "keep_days": 14,
        "error_keep_files": 60,
        "test_keep_files": 120,
        "normal_keep_files": 5,
        "normal_sample_every": 20,
        "prune_interval_sec": 30.0,
    },

    "recovery": {
        "camera_open_timeout_sec": 2.0,
        "camera_retry_interval_sec": 0.1,
        "camera_grace_sec": 2.0,
        "frame_timeout_sec": 1.0,
    },

    "error_test": {
        "enabled": False,
        "request_path": "data/runtime/plc_error_test_request.json",
        "delete_after_read": True,
        "allowed_types": [
            "camera",
            "light",
            "inspection",
            "plc_comm",
        ],
        "allowed_modes": ["logic", "recovery"],
        "camera_health_frames": 3,
        "camera_health_timeout_sec": 1.5,
        "allow_hardware_fault_tests": False,
    },



    "service_panel": {
        "enabled": False,
        "panel_width": 600,
        "opacity": 0.96,
        "password_sha256": "2e29ae4c214d913412f389feea5bcacf1fca48c8a4669dbd5c7d8c2a989d728f",
        "lock_on_close": True,
        "show_roi_debug": True,
        "rx_tx_rows": 3,
        "refresh_hz": 5.0,
        "roi_debug_hz": 1.0,
        "soak_test": {
            "enabled": True,
            "interval_sec": 30.0,
            "reset_delay_sec": 2.0,
            "start_delay_sec": 1.0,
            "log_dir": "soak_tests",
            "keep_days": 14,
            "keep_sessions": 20,
        },
    },

    "shutdown": {
        "command": "sudo /sbin/shutdown -h now",
        "ready_hold_sec": 2.0,
    },
}


def _merge_default(base: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)

    for key, value in user.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key].update(value)
        else:
            out[key] = value

    return out


def load_plc_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"[PLC CONFIG] not found, use default: {path}")
        return copy.deepcopy(DEFAULT_PLC_CONFIG)

    with open(path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"plc_config must be object: {path}")

    return _merge_default(DEFAULT_PLC_CONFIG, cfg)