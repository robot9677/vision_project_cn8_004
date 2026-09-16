# ===== START 2026-09-16 : 검사결과 저장/로그백업 공통 안정화 =====
"""Inspection evidence storage, daily summary/archive, and retention policy.

This module is intentionally independent from inspection decision logic.
Failures here must never change an inspection result.
"""
import json
import os
import re
import shutil
import time
import zipfile
from datetime import datetime

import cv2
import numpy as np


class InspectionLogArchiveManager:
    def __init__(self, *, logs_root, project_root, equipment_name="VISION", keep_days=20):
        self.logs_root = os.path.abspath(logs_root)
        self.project_root = os.path.abspath(project_root)
        self.equipment_name = str(equipment_name or "VISION")
        self.keep_days = max(1, int(keep_days))
        os.makedirs(self.logs_root, exist_ok=True)

    @staticmethod
    def _value(result, key, default=None):
        return result.get(key, default) if isinstance(result, dict) else getattr(result, key, default)

    def _failed_ids(self, results):
        out = []
        for key, result in (results or {}).items():
            if self._value(result, "ok", None) is False:
                rid = str(self._value(result, "roi_id", key)).replace("ROI", "")
                out.append(rid)
        return sorted(set(out), key=lambda x: int(x) if x.isdigit() else 999999)

    def _day_dir(self, day):
        return os.path.join(self.logs_root, day)

    def archive_paths(self, day):
        """Return the summary, ZIP and upload-marker paths for one day."""
        day_dir = self._day_dir(day)
        stem = f"{self.equipment_name}_{day}"
        txt_path = os.path.join(day_dir, stem + ".txt")
        zip_path = os.path.join(day_dir, stem + ".zip")
        return txt_path, zip_path, zip_path + ".drive_uploaded"

    def is_day_uploaded(self, day):
        """A marker is written only after Drive confirms the ZIP upload."""
        _txt_path, _zip_path, marker_path = self.archive_paths(day)
        return os.path.isfile(marker_path)

    def _next_sequence(self, day_dir):
        # ===== START 2026-08-27 : 구형/신형 로그 파일명 호환 =====
        max_seq = 0

        try:
            for name in os.listdir(day_dir):

                # 구형
                # 0021_raw_171049_422_...
                # 0021_result_171049_422_...
                old_format = re.match(
                    r"^(\d{4,})_(?:raw|result)_\d{6}_\d{3}_",
                    name,
                )

                # 신형
                # 0021_171049_422_raw_...
                # 0021_171049_422_result_...
                new_format = re.match(
                    r"^(\d{4,})_\d{6}_\d{3}_(?:raw|result)_",
                    name,
                )

                match = new_format or old_format

                if match:
                    max_seq = max(max_seq, int(match.group(1)))

        except OSError:
            pass

        return max_seq + 1
        # ===== END 2026-08-27 : 구형/신형 로그 파일명 호환 =====

    def save_run(self, frame_gray8, overall_ok, results, payload):
        now = time.time()
        day = time.strftime("%Y%m%d", time.localtime(now))
        clock = time.strftime("%H%M%S", time.localtime(now))
        mmm = int((now * 1000) % 1000)
        day_dir = self._day_dir(day)
        os.makedirs(day_dir, exist_ok=True)

        seq = self._next_sequence(day_dir)
        failed = self._failed_ids(results)
        verdict = "OK" if bool(overall_ok) else "{}_NG".format("_".join("ROI" + x for x in failed) or "UNKNOWN")
        # ===== START 2026-08-26 : 로그 파일명 정렬 구조 개선 =====
        raw_name = f"{seq:04d}_{clock}_{mmm:03d}_raw_{verdict}.png"
        result_name = f"{seq:04d}_{clock}_{mmm:03d}_result_{verdict}.json"
        # ===== END 2026-08-26 : 로그 파일명 정렬 구조 개선 =====
        raw_path = os.path.join(day_dir, raw_name)
        result_path = os.path.join(day_dir, result_name)

        # cv2.imwrite writes the supplied inspection frame without visual overlays.
        if frame_gray8 is None or not isinstance(frame_gray8, np.ndarray) or frame_gray8.size == 0:
            raise ValueError("empty inspection frame")
        if not cv2.imwrite(raw_path, frame_gray8):
            raise IOError("raw image write failed: " + raw_path)

        payload = dict(payload or {})
        payload["evidence"] = {
            "sequence": seq,
            "raw_file": raw_name,
            "result_file": result_name,
            "failed_roi_ids": ["ROI" + x for x in failed],
        }
        tmp = result_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, result_path)
        # Retention is intentionally not run in the inspection path. It is
        # applied by the backup worker only after a Drive upload succeeds.
        return raw_path, result_path, day_dir

    def save_inspect_summary(self, overall_ok, results):
        now = time.time()
        day = time.strftime("%Y%m%d", time.localtime(now))
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        mmm = int((now * 1000) % 1000)
        out_dir = os.path.join(self._day_dir(day), "inspect_log")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"inspect_{ts}_{mmm:03d}.json")
        payload = {
            "ts": f"{ts}_{mmm:03d}", "epoch": now, "overall_ok": bool(overall_ok),
            "results": {
                str(k): {
                    "roi_id": str(self._value(v, "roi_id", k)),
                    "ok": bool(self._value(v, "ok", False)),
                    "reason": str(self._value(v, "reason", "")),
                } for k, v in (results or {}).items()
            },
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return path

    def _result_files(self, day_dir):
        # ===== START 2026-08-27 : 구형/신형 로그 파일명 호환 =====
        files = []

        for name in os.listdir(day_dir):

            # 구형
            # 0021_result_171049_422_....json
            old_format = re.match(
                r"^\d{4,}_result_\d{6}_\d{3}_.*\.json$",
                name,
            )

            # 신형
            # 0021_171049_422_result_....json
            new_format = re.match(
                r"^\d{4,}_\d{6}_\d{3}_result_.*\.json$",
                name,
            )

            if old_format or new_format:
                files.append(os.path.join(day_dir, name))

        return sorted(files)
        # ===== END 2026-08-27 : 구형/신형 로그 파일명 호환 =====

    def _camera_events(self, day_dir):
        path = os.path.join(day_dir, "system_log", "camera_recovery.jsonl")
        events = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try: events.append(json.loads(line))
                    except Exception: pass
        except OSError:
            pass
        return events

    def finalize_day(self, day):
        if not re.fullmatch(r"\d{8}", str(day or "")):
            raise ValueError("invalid inspection day: " + str(day))

        day_dir = self._day_dir(day)
        if not os.path.isdir(day_dir):
            return "", ""
        result_files = self._result_files(day_dir)
        if not result_files:
            return "", ""

        total = ok = ng = 0
        roi_ng = {}
        ng_details = []
        epochs = []
        for p in result_files:
            try:
                with open(p, encoding="utf-8") as f: data = json.load(f)
            except Exception:
                continue
            total += 1
            overall = bool(data.get("overall_ok", False))
            ok += int(overall); ng += int(not overall)
            if isinstance(data.get("ts"), (int, float)): epochs.append(float(data["ts"]))
            failed = ((data.get("evidence") or {}).get("failed_roi_ids") or [])
            if not overall:
                for rid in failed: roi_ng[rid] = roi_ng.get(rid, 0) + 1
                raw_file = (data.get("evidence") or {}).get("raw_file", "")
                ng_details.append((raw_file, failed))

        txt_path, zip_path, marker_path = self.archive_paths(day)
        camera_events = self._camera_events(day_dir)
        starts = [e for e in camera_events if e.get("event") == "CAMERA_RECOVERY_START"]
        successes = [e for e in camera_events if e.get("event") == "CAMERA_RECOVERY_OK"]
        failures = [e for e in camera_events if e.get("event") == "CAMERA_RECOVERY_FAIL"]

        lines = [
            f"장비명: {self.equipment_name}", f"검사일: {day}",
            f"최초 검사: {datetime.fromtimestamp(min(epochs)).strftime('%H:%M:%S') if epochs else '-'}",
            f"최종 검사: {datetime.fromtimestamp(max(epochs)).strftime('%H:%M:%S') if epochs else '-'}", "",
            f"TOTAL: {total}", f"OK: {ok}", f"NG: {ng}", "", "[ROI별 NG]",
        ]
        for rid in sorted(roi_ng, key=lambda x: int(str(x).replace("ROI", "")) if str(x).replace("ROI", "").isdigit() else 999999):
            lines.append(f"{rid}: {roi_ng[rid]}회")
        if not roi_ng: lines.append("없음")
        lines += ["", "[NG 상세]"]
        for raw_file, failed in ng_details:
            lines.append(f"{raw_file} : {', '.join(failed) if failed else 'UNKNOWN'}")
        if not ng_details: lines.append("없음")
        lines += ["", "[카메라 복구]", f"복구 요청: {len(starts)}회", f"복구 성공: {len(successes)}회", f"복구 실패: {len(failures)}회"]
        for e in successes + failures:
            lines.append(f"{e.get('time','-')}  {e.get('event')}  {e.get('reason','')}")
        with open(txt_path + ".tmp", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(txt_path + ".tmp", txt_path)

        # Archive source evidence only. The completion marker must never be
        # included because it is created after the upload and would otherwise
        # change a ZIP that has already been uploaded.
        with zipfile.ZipFile(zip_path + ".tmp", "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for root, dirs, files in os.walk(day_dir):
                dirs.sort()
                for name in sorted(files):
                    p = os.path.join(root, name)
                    if p in (zip_path, zip_path + ".tmp", marker_path):
                        continue
                    if name.endswith(".tmp"):
                        continue
                    z.write(p, os.path.relpath(p, day_dir))
        os.replace(zip_path + ".tmp", zip_path)
        return txt_path, zip_path

    def latest_previous_day(self):
        """Return the newest completed inspection day before today.

        Folders with only diagnostic/inspect-summary files are ignored. This
        prevents an incomplete folder from blocking the latest valid backup.
        """
        today = time.strftime("%Y%m%d")
        days = []
        for name in os.listdir(self.logs_root):
            day_dir = self._day_dir(name)
            if not (
                re.fullmatch(r"\d{8}", name)
                and name < today
                and os.path.isdir(day_dir)
            ):
                continue
            if self._result_files(day_dir):
                days.append(name)
        return max(days) if days else ""

    def finalize_latest_previous_day(self):
        day = self.latest_previous_day()
        if not day:
            return "", ""
        return self.finalize_day(day)

    def prune_day_dirs(self):
        """Keep the newest N day folders and delete only uploaded old days."""
        days = sorted(d for d in os.listdir(self.logs_root) if re.fullmatch(r"\d{8}", d) and os.path.isdir(self._day_dir(d)))
        for day in days[:-self.keep_days]:
            if self.is_day_uploaded(day):
                shutil.rmtree(self._day_dir(day))
            else:
                print(
                    "[INSPECT ARCHIVE] retention skipped; Drive marker missing:",
                    day,
                )
# ===== END 2026-09-16 : 검사결과 저장/로그백업 공통 안정화 =====
