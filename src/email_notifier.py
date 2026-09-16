#!/usr/bin/env python3
"""Minimal e-mail notifications for the vision application."""

import argparse
import json
import mimetypes
import os
import shutil
import smtplib
import socket
import subprocess
import threading
import time
from datetime import datetime
from email.message import EmailMessage


class EmailNotifier:
    def __init__(self, *, config_path, logs_root, static_context=None):
        self.config_path = os.path.abspath(config_path)
        self.logs_root = os.path.abspath(logs_root)
        self.context = dict(static_context or {})
        self.config = self._load_json(self.config_path)
        self.enabled = bool(self.config.get("enabled", False))
        self.app_started_epoch = time.time()
        self.boot_id = self._read_text("/proc/sys/kernel/random/boot_id") or str(
            int(self._boot_epoch())
        )
        self.state_path = os.path.join(self.logs_root, "email_notifier_state.json")
        self.log_path = os.path.join(self.logs_root, "email_notifier.jsonl")
        self.pending_dir = os.path.join(self.logs_root, "email_pending")
        self._lock = threading.Lock()
        self._startup_queued = False
        self._last_ng = {}
        os.makedirs(self.logs_root, exist_ok=True)
        print("[EMAIL] enabled" if self.enabled else "[EMAIL] disabled")

    def handle_inspection(self, *, overall_ok, results, run_dir="", recipe=None):
        """Called after inspection. It never blocks the inspection loop."""
        if not self.enabled:
            return

        results = results or {}
        recipe = recipe or {}
        inspected_at = time.time()

        if self._should_send_startup():
            self._startup_queued = True
            threading.Thread(
                target=self._send_startup,
                args=(bool(overall_ok), results, inspected_at),
                daemon=True,
                name="email-startup",
            ).start()

        if not bool(overall_ok):
            self._queue_ng(results, recipe, run_dir, inspected_at)

    # ===== START 2026-08-26 : Google Drive 일일 로그 백업 =====
    def send_daily_backup_status(self, txt_path, zip_path, drive_result=None, error=""):
        """Send only a small Drive backup status mail. Never attach the daily ZIP."""
        if not self.enabled:
            return

        drive_result = dict(drive_result or {})
        success = not bool(error)
        day_name = os.path.basename(txt_path).replace(".txt", "") if txt_path else "UNKNOWN"
        subject = self._subject(
            "일일 검사결과 Drive 백업 {} - {}".format("완료" if success else "실패", day_name)
        )
        lines = [
            "장비: {}".format(self._equipment_name()),
            "백업 파일: {}".format(os.path.basename(zip_path) if zip_path else "-"),
            "Google Drive 백업: {}".format("완료" if success else "실패"),
        ]
        if success:
            lines.append("Drive 상태: {}".format(drive_result.get("status", "uploaded")))
            lines.append("Drive 파일 ID: {}".format(drive_result.get("id", "-")))
        else:
            lines.append("오류: {}".format(str(error)))
        body = "\n".join(lines)

        def worker():
            if self._send_mail(subject, body, ""):
                self._log("DAILY_DRIVE_STATUS_SENT", subject)

        threading.Thread(target=worker, daemon=True, name="email-daily-drive-status").start()

    # ===== END 2026-08-26 : Google Drive 일일 로그 백업 =====

    def send_test_mail(self):
        body = "\n".join(
            [
                "비전 장비 이메일 설정 테스트입니다.",
                "",
                "장비: {}".format(self._equipment_name()),
                "호스트명: {}".format(socket.gethostname()),
                "발송 시각: {}".format(self._format_time(time.time())),
            ]
        )
        return self._send_mail(self._subject("EMAIL TEST"), body, "")

    # ------------------------------------------------------------------
    # Startup mail: once per OS boot, after the first completed inspection
    # ------------------------------------------------------------------
    def _should_send_startup(self):
        if not bool((self.config.get("startup_status") or {}).get("enabled", True)):
            return False
        with self._lock:
            if self._startup_queued:
                return False
            state = self._load_json(self.state_path)
            return state.get("startup_mail_boot_id") != self.boot_id

    def _send_startup(self, overall_ok, results, inspected_at):
        subject = self._subject(
            "부팅 후 첫 검사 완료 - {}".format("OK" if overall_ok else "NG")
        )
        body = self._startup_body(overall_ok, results, inspected_at)
        # ===== START 2026-08-27 : 부팅/NG 메일 수신자 분리 =====
        startup_cfg = self.config.get("startup_status") or {}
        recipients = startup_cfg.get("recipients")

        if self._send_mail(subject, body, "", recipients=recipients):
        # ===== END 2026-08-27 : 부팅/NG 메일 수신자 분리 =====
            self._save_json(
                self.state_path,
                {
                    "startup_mail_boot_id": self.boot_id,
                    "startup_mail_sent_at": self._format_time(time.time()),
                },
            )
            self._log("STARTUP_SENT", subject)
        else:
            with self._lock:
                self._startup_queued = False

    def _startup_body(self, overall_ok, results, inspected_at):
        failed = self._failed_ids(results)
        return "\n".join(
            [
                "[장비 부팅 후 첫 검사 완료]",
                "",
                "장비: {}".format(self._equipment_name()),
                "호스트명: {}".format(socket.gethostname()),
                "부팅 시각: {}".format(self._format_time(self._boot_epoch())),
                "프로그램 시작: {}".format(self._format_time(self.app_started_epoch)),
                "첫 검사 완료: {}".format(self._format_time(inspected_at)),
                "첫 검사 결과: {}".format("OK" if overall_ok else "NG"),
                "NG ROI: {}".format(
                    ", ".join("ROI{}".format(x) for x in failed) or "없음"
                ),
                "",
                "프로파일: {}".format(self.context.get("profile_name") or "-"),
                "레시피: {}".format(self.context.get("recipe_name") or "-"),
                "카메라: {} / {} / 첫 검사 완료".format(
                    self.context.get("camera_name") or "-",
                    self.context.get("camera_size") or "-",
                ),
                "PLC: {}".format(self._plc_text()),
                "LED: {}".format(self._light_text()),
                "네트워크: {}".format(self._network_text()),
                "",
                "※ PLC/LED는 485 통신 성공 여부를 검사하지 않고 설정 상태만 표시합니다.",
            ]
        )

    # ------------------------------------------------------------------
    # NG mail
    # ------------------------------------------------------------------
    def _queue_ng(self, results, recipe, run_dir, inspected_at):
        cfg = self.config.get("ng_alert") or {}
        if not bool(cfg.get("enabled", True)):
            return

        key = self._ng_key(results)
        now = time.time()
        dedupe_sec = max(0, int(cfg.get("dedupe_sec", 300)))
        with self._lock:
            if now - self._last_ng.get(key, 0.0) < dedupe_sec:
                self._log("NG_DUPLICATE_SKIPPED", key)
                return
            self._last_ng[key] = now

        image_path = self._stage_ng_image(run_dir, key, inspected_at)
        delay_sec = max(0, int(cfg.get("delay_sec", 300)))
        threading.Thread(
            target=self._send_ng,
            args=(results, recipe, run_dir, image_path, inspected_at, delay_sec),
            daemon=True,
            name="email-ng",
        ).start()
        self._log("NG_QUEUED", "delay_sec={}".format(delay_sec))

    def _send_ng(self, results, recipe, run_dir, image_path, inspected_at, delay_sec):
        try:
            if delay_sec:
                time.sleep(delay_sec)
            # ===== START 2026-08-26 : 검사결과 저장/로그백업 구조 변경 =====
            failed = self._failed_ids(results)
            labels = [self._roi_label(recipe, x) for x in failed]
            roi_text = "_".join("ROI{}".format(x) for x in failed) or "UNKNOWN"
            label_text = ",".join(labels) if labels else "NG"
            subject = "{}_{:s}({}_NG)".format(self._equipment_name(), roi_text, label_text)
            # ===== START 2026-08-27 : NG 메일 전용 검사명 표시 =====
            body_lines = [self._equipment_name() + " 장비 NG 발생 내용", ""]

            for label in labels:
                body_lines.append("{} : NG".format(label))

            body = "\n".join(body_lines)
            # ===== END 2026-08-27 : NG 메일 전용 검사명 표시 =====
            # ===== END 2026-08-26 : 검사결과 저장/로그백업 구조 변경 =====
            # ===== START 2026-08-27 : 부팅/NG 메일 수신자 분리 =====
            ng_cfg = self.config.get("ng_alert") or {}
            recipients = ng_cfg.get("recipients")

            if self._send_mail(subject, body, image_path, recipients=recipients):
            # ===== END 2026-08-27 : 부팅/NG 메일 수신자 분리 =====
                self._log("NG_SENT", subject)
        finally:
            self._remove_staged_image(image_path)

    def _ng_body(self, results, recipe, run_dir, image_path, inspected_at):
        failed = self._failed_ids(results)
        lines = [
            "[NG 검사 알림]",
            "",
            "장비: {}".format(self._equipment_name()),
            "호스트명: {}".format(socket.gethostname()),
            "발생 시각: {}".format(self._format_time(inspected_at)),
            "프로파일: {}".format(self.context.get("profile_name") or "-"),
            "레시피: {}".format(self.context.get("recipe_name") or "-"),
            "NG ROI: {}".format(
                ", ".join("ROI{}".format(x) for x in failed) or "UNKNOWN"
            ),
            "대표 이미지: {}".format(os.path.basename(image_path) if image_path else "없음"),
            "검사 로그: {}".format(run_dir or "없음"),
            "",
            "[NG 상세]",
        ]

        for roi_id in failed:
            result = results.get(str(roi_id), results.get(roi_id))
            reason = self._value(result, "reason", "FAIL")
            metrics = self._simple_metrics(self._value(result, "metrics", {}))
            setting = self._recipe_for_roi(recipe, roi_id)
            lines.extend(
                [
                    "",
                    "ROI{}".format(roi_id),
                    "- 판정 사유: {}".format(reason),
                    "- 측정 정보: {}".format(
                        json.dumps(metrics, ensure_ascii=False) if metrics else "없음"
                    ),
                    "- 검사 설정: {}".format(
                        json.dumps(setting, ensure_ascii=False) if setting else "없음"
                    ),
                ]
            )
        return "\n".join(lines)

    def _stage_ng_image(self, run_dir, key, inspected_at):
        source = self._pick_image(run_dir)
        if not source:
            return ""
        try:
            os.makedirs(self.pending_dir, exist_ok=True)
            ext = os.path.splitext(source)[1] or ".png"
            name = "{}_{}_ng{}".format(
                datetime.fromtimestamp(inspected_at).strftime("%Y%m%d_%H%M%S"),
                abs(hash(key)) % 100000000,
                ext,
            )
            target = os.path.join(self.pending_dir, name)
            shutil.copy2(source, target)
            return target
        except Exception as exc:
            self._log("NG_IMAGE_COPY_FAILED", str(exc))
            return source

    # ===== START 2026-08-26 : 검사결과 저장/로그백업 구조 변경 =====
    def _pick_image(self, run_dir):
        # New logger passes the exact RAW evidence path, not a run directory.
        if run_dir and os.path.isfile(run_dir) and run_dir.lower().endswith(".png"):
            return run_dir
        if run_dir and os.path.isdir(run_dir):
            path = os.path.join(run_dir, "raw.png")
            return path if os.path.isfile(path) else ""
        return ""

    # ===== START 2026-08-27 : NG 메일 전용 검사명 분리 =====
    @staticmethod
    def _roi_label(recipe, roi_id):
        for item in (recipe or {}).get("inspections") or []:
            item_roi_id = str(
                item.get("roi_id", item.get("roi_name", ""))
            ).replace("ROI", "")

            if item_roi_id == str(roi_id):
                return str(
                    item.get("email_name")
                    or item.get("roi_name")
                    or ("ROI" + str(roi_id))
                )

        return "ROI" + str(roi_id)
    # ===== END 2026-08-27 : NG 메일 전용 검사명 분리 =====

    def _remove_staged_image(self, path):
        if not path:
            return
        try:
            if os.path.commonpath([self.pending_dir, os.path.abspath(path)]) == os.path.abspath(
                self.pending_dir
            ):
                os.remove(path)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # SMTP
    # ------------------------------------------------------------------
    def _send_mail(self, subject, body, attachment_path, recipients=None):
        try:
            smtp_cfg = self.config.get("smtp") or {}
            host = str(smtp_cfg.get("host") or "").strip()
            port = int(smtp_cfg.get("port", 465))
            security = str(smtp_cfg.get("security") or "ssl").lower()
            username = str(smtp_cfg.get("username") or "").strip()
            password = self._smtp_password(smtp_cfg)
            sender = str(smtp_cfg.get("sender") or username).strip()
            # ===== START 2026-08-27 : 부팅/NG 메일 수신자 분리 =====
            if recipients is None:
                recipients = smtp_cfg.get("recipients")

            recipients = self._recipients(recipients)
            # ===== END 2026-08-27 : 부팅/NG 메일 수신자 분리 =====

            if not host or not sender or not recipients or (username and not password):
                raise ValueError("email_config.json SMTP 항목이 완성되지 않았습니다")

            message = EmailMessage()
            message["Subject"] = subject
            message["From"] = sender
            message["To"] = ", ".join(recipients)
            message.set_content(body)

            if attachment_path and os.path.isfile(attachment_path):
                mime, _ = mimetypes.guess_type(attachment_path)
                main_type, sub_type = (mime or "application/octet-stream").split("/", 1)
                with open(attachment_path, "rb") as file:
                    message.add_attachment(
                        file.read(),
                        maintype=main_type,
                        subtype=sub_type,
                        filename=os.path.basename(attachment_path),
                    )

            # ===== START 2026-08-26 : 일일 백업 메일 안정화 =====
            # Retry each SMTP transaction with a fresh connection. This keeps
            # inspection independent from temporary SMTP disconnects and also
            # makes the existing retry_count/retry_interval_sec config effective.
            retry_count = max(0, int(self.config.get("retry_count", 2)))
            retry_interval = max(0, int(self.config.get("retry_interval_sec", 30)))
            timeout_sec = max(1, int(smtp_cfg.get("timeout_sec", 20)))
            last_exc = None

            for attempt in range(retry_count + 1):
                try:
                    if security == "ssl":
                        with smtplib.SMTP_SSL(host, port, timeout=timeout_sec) as smtp:
                            if username:
                                smtp.login(username, password)
                            smtp.send_message(message)
                    else:
                        with smtplib.SMTP(host, port, timeout=timeout_sec) as smtp:
                            if security in ("tls", "starttls"):
                                smtp.starttls()
                            if username:
                                smtp.login(username, password)
                            smtp.send_message(message)
                    if attempt:
                        self._log("SEND_RETRY_SUCCESS", "attempt={}".format(attempt + 1))
                    return True
                except Exception as exc:
                    last_exc = exc
                    self._log(
                        "SEND_RETRY_FAILED",
                        "attempt={}/{} {}".format(attempt + 1, retry_count + 1, exc),
                    )
                    if attempt < retry_count and retry_interval:
                        time.sleep(retry_interval)

            raise last_exc if last_exc is not None else RuntimeError("SMTP send failed")
            # ===== END 2026-08-26 : 일일 백업 메일 안정화 =====
        except Exception as exc:
            print("[EMAIL] send failed:", exc)
            self._log("SEND_FAILED", str(exc))
            return False

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _equipment_name(self):
        # ===== START 2026-08-26 : 검사결과 저장/로그백업 구조 변경 =====
        return str(self.context.get("equipment_name") or self.config.get("equipment_name") or socket.gethostname())
        # ===== END 2026-08-26 : 검사결과 저장/로그백업 구조 변경 =====

    def _subject(self, text):
        return "[{}] {}".format(self._equipment_name(), text)

    def _plc_text(self):
        if bool(self.context.get("plc_enabled", False)):
            return "설정 ON ({}, 통신 검증 제외)".format(
                self.context.get("plc_backend") or "-"
            )
        return "설정 OFF (개발보드, 통신 검증 제외)"

    def _light_text(self):
        backend = str(self.context.get("light_backend") or "disabled")
        if backend.lower() in ("mock", "disabled", "none", "null"):
            return "{} (개발보드, 통신 검증 제외)".format(backend.upper())
        return "설정 ON ({}, 통신 검증 제외)".format(backend)

    def _network_text(self):
        try:
            out = subprocess.check_output(["hostname", "-I"], timeout=2).decode().split()
            ipv4 = [x for x in out if ":" not in x]
            lan = [x for x in ipv4 if not x.startswith("100.")]
            tail = [x for x in ipv4 if x.startswith("100.")]
            parts = []
            if lan:
                parts.append("LAN " + ", ".join(lan))
            if tail:
                parts.append("Tailscale " + ", ".join(tail))
            return " / ".join(parts) or "확인 불가"
        except Exception:
            return "확인 불가"

    def _failed_ids(self, results):
        failed = []
        for key, result in (results or {}).items():
            if self._value(result, "ok", None) is False:
                failed.append(str(self._value(result, "roi_id", key)).replace("ROI", ""))
        return sorted(failed, key=lambda x: int(x) if str(x).isdigit() else 999999)

    def _ng_key(self, results):
        return "|".join(
            "{}:{}".format(
                roi_id,
                self._value(results.get(str(roi_id), results.get(roi_id)), "reason", ""),
            )
            for roi_id in self._failed_ids(results)
        )

    @staticmethod
    def _value(result, key, default=None):
        return result.get(key, default) if isinstance(result, dict) else getattr(result, key, default)

    @staticmethod
    def _simple_metrics(metrics):
        out = {}
        for key, value in (metrics or {}).items():
            if str(key).startswith("_") or len(out) >= 20:
                continue
            if value is None or isinstance(value, (str, int, float, bool)):
                out[str(key)] = value
            elif hasattr(value, "item"):
                try:
                    out[str(key)] = value.item()
                except Exception:
                    pass
        return out

    @staticmethod
    def _recipe_for_roi(recipe, roi_id):
        key = "ROI{}".format(roi_id)
        overrides = (recipe or {}).get("overrides") or {}
        if isinstance(overrides.get(key), dict):
            return overrides[key]
        for item in (recipe or {}).get("inspections") or []:
            if str(item.get("roi_id", item.get("roi_name", ""))).replace("ROI", "") == str(
                roi_id
            ):
                return item
        return {}

    def _smtp_password(self, smtp_cfg):
        env_name = str(smtp_cfg.get("password_env") or "").strip()
        env_value = os.environ.get(env_name, "") if env_name else ""
        return str(env_value or smtp_cfg.get("app_password") or "").replace(" ", "")

    @staticmethod
    def _recipients(value):
        if isinstance(value, str):
            value = value.replace(";", ",").split(",")
        return [str(x).strip() for x in (value or []) if str(x).strip()]

    def _boot_epoch(self):
        try:
            return time.time() - float(self._read_text("/proc/uptime").split()[0])
        except Exception:
            return self.app_started_epoch

    @staticmethod
    def _read_text(path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                return file.read().strip()
        except Exception:
            return ""

    @staticmethod
    def _format_time(epoch):
        return datetime.fromtimestamp(float(epoch)).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _load_json(path):
        try:
            with open(path, "r", encoding="utf-8-sig") as file:
                data = json.load(file)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _save_json(path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp = path + ".tmp"
        with open(temp, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        os.replace(temp, path)

    def _log(self, event, detail=""):
        try:
            with open(self.log_path, "a", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {
                            "timestamp": self._format_time(time.time()),
                            "event": event,
                            "detail": detail,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            pass


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="Vision e-mail test")
    parser.add_argument(
        "--config",
        default=os.path.join(project_root, "data", "config", "email_config.json"),
    )
    parser.add_argument(
        "--logs-root",
        default=os.path.join(project_root, "data", "logs"),
    )
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    notifier = EmailNotifier(
        config_path=args.config,
        logs_root=args.logs_root,
        static_context={"profile_name": "TEST", "recipe_name": "TEST"},
    )
    if not args.test:
        parser.print_help()
        return 0
    if not notifier.enabled:
        print("[EMAIL] email_config.json의 enabled를 true로 변경하세요")
        return 2
    return 0 if notifier.send_test_mail() else 1


if __name__ == "__main__":
    raise SystemExit(main())
