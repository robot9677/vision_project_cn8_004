import json
import os
import shutil
import numpy as np
from datetime import datetime

import cv2


def prune_snapshots(path, max_keep=200):
    """Legacy file-count pruning retained for compatibility."""
    try:
        files = []
        for fn in os.listdir(path):
            if fn.endswith(".png") or fn.endswith(".jpg"):
                p = os.path.join(path, fn)
                files.append((os.path.getmtime(p), p))
        files.sort(reverse=True)
        for _, p in files[max_keep:]:
            try:
                os.remove(p)
            except Exception:
                pass
    except Exception:
        pass


def prune_manifests(dir_path, keep=200):
    """Legacy manifest-count pruning retained for compatibility."""
    try:
        items = []
        for fn in os.listdir(dir_path):
            if fn.endswith(".json"):
                p = os.path.join(dir_path, fn)
                items.append((os.path.getmtime(p), p))
        items.sort(reverse=True)

        for _, jpath in items[keep:]:
            try:
                with open(jpath, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                for k in ("raw", "overlay", "crop"):
                    p = meta.get(k)
                    if p and os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                os.remove(jpath)
            except Exception:
                try:
                    os.remove(jpath)
                except Exception:
                    pass
    except Exception:
        pass


def _capture_group_key(meta, filename):
    ts = str((meta or {}).get("ts", "") or "").strip()
    if ts:
        return ts

    stem = os.path.splitext(filename)[0]
    if stem.startswith("cap_") and "_ROI" in stem:
        return stem[4:].split("_ROI", 1)[0]
    return stem


def prune_capture_groups(dir_path, keep=10):
    """Keep the newest N capture operations, not merely N individual files."""
    try:
        groups = {}
        for filename in os.listdir(dir_path):
            if not filename.endswith(".json"):
                continue
            manifest_path = os.path.join(dir_path, filename)
            try:
                with open(manifest_path, "r", encoding="utf-8") as file:
                    meta = json.load(file)
            except Exception:
                meta = {}

            group_key = _capture_group_key(meta, filename)
            group = groups.setdefault(
                group_key,
                {"mtime": 0.0, "manifests": [], "files": set()},
            )
            try:
                group["mtime"] = max(
                    float(group["mtime"]),
                    float(os.path.getmtime(manifest_path)),
                )
            except OSError:
                pass
            group["manifests"].append(manifest_path)
            for key in ("raw", "overlay", "crop", "shared_raw", "shared_overlay"):
                path = meta.get(key) if isinstance(meta, dict) else None
                if path:
                    group["files"].add(str(path))

        ordered = sorted(
            groups.values(),
            key=lambda item: float(item.get("mtime", 0.0)),
            reverse=True,
        )

        for group in ordered[max(1, int(keep)):]:
            for path in group.get("files", set()):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass
            for manifest_path in group.get("manifests", []):
                try:
                    os.remove(manifest_path)
                except OSError:
                    pass
    except Exception as error:
        print(f"[DATASET] capture pruning failed: {error}")


def _get_roi_by_id(roi_mgr, roi_id: int):
    try:
        for r in getattr(roi_mgr, "rois", []):
            if int(r.get("id", -1)) == int(roi_id):
                return r
    except Exception:
        pass
    return None


def crop_roi(gray8, roi_mgr, roi_id: int):
    r = _get_roi_by_id(roi_mgr, roi_id)
    if r is None or gray8 is None:
        return None

    x = int(r.get("x", 0))
    y = int(r.get("y", 0))
    w = int(r.get("w", 0))
    h = int(r.get("h", 0))

    if w <= 0 or h <= 0:
        return None

    H, W = gray8.shape[:2]
    x1 = max(0, min(x, W - 1))
    y1 = max(0, min(y, H - 1))
    x2 = max(x1 + 1, min(W, x1 + w))
    y2 = max(y1 + 1, min(H, y1 + h))

    crop = gray8[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None
    return crop


def _link_or_copy(source_path, destination_path):
    try:
        if os.path.exists(destination_path):
            os.remove(destination_path)
        os.link(source_path, destination_path)
        return
    except Exception:
        pass

    try:
        shutil.copy2(source_path, destination_path)
    except Exception:
        pass



def _json_safe(value):
    """Convert inspection metrics to JSON-safe values without storing image arrays."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            return {
                "type": "ndarray",
                "shape": [int(v) for v in value.shape],
                "dtype": str(value.dtype),
            }
        except Exception:
            return "ndarray"
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _result_to_dict(result):
    if result is None:
        return None
    if isinstance(result, dict):
        ok = result.get("ok")
        reason = result.get("reason")
        metrics = result.get("metrics") or {}
    else:
        ok = getattr(result, "ok", None)
        reason = getattr(result, "reason", None)
        metrics = getattr(result, "metrics", None) or {}
    return {
        "ok": None if ok is None else bool(ok),
        "reason": reason,
        "metrics": _json_safe(metrics),
    }


def _prune_inspection_capture_dirs(root_dir, keep=50):
    try:
        items = []
        for name in os.listdir(root_dir):
            path = os.path.join(root_dir, name)
            if os.path.isdir(path):
                items.append((os.path.getmtime(path), path))
        items.sort(reverse=True)
        for _, path in items[max(1, int(keep)):]:
            shutil.rmtree(path, ignore_errors=True)
    except Exception as error:
        print(f"[CAPTURE ALL] pruning failed: {error}")


def save_inspection_capture(
    frame_gray8,
    vis_bgr,
    *,
    roi_mgr,
    data_dir,
    last_results,
    overall_ok,
    trigger="INSPECT",
    snapshot_keep=50,
):
    """Temporarily save every completed inspection for ROI tuning.

    Output:
      data/dataset/ALL/<timestamp>/
        raw.png
        overlay.png
        ROI1_crop.png ...
        result.json
    """
    if frame_gray8 is None or vis_bgr is None:
        return ""

    root_dir = os.path.join(data_dir, "dataset", "ALL")
    os.makedirs(root_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_dir = os.path.join(root_dir, ts)
    os.makedirs(run_dir, exist_ok=False)

    raw_path = os.path.join(run_dir, "raw.png")
    overlay_path = os.path.join(run_dir, "overlay.png")
    cv2.imwrite(raw_path, frame_gray8)
    cv2.imwrite(overlay_path, vis_bgr)

    results = last_results or {}
    roi_payload = {}

    for roi in getattr(roi_mgr, "rois", []):
        roi_id = int(roi.get("id"))
        crop = crop_roi(frame_gray8, roi_mgr, roi_id)
        crop_name = f"ROI{roi_id}_crop.png"
        crop_path = os.path.join(run_dir, crop_name)
        if crop is not None:
            cv2.imwrite(crop_path, crop)

        roi_payload[str(roi_id)] = {
            "roi": {
                "id": roi_id,
                "name": str(roi.get("name", f"ROI{roi_id}")),
                "x": int(roi.get("x", 0)),
                "y": int(roi.get("y", 0)),
                "w": int(roi.get("w", 0)),
                "h": int(roi.get("h", 0)),
                "angle": float(roi.get("angle", 0.0)),
            },
            "crop": crop_name if crop is not None else None,
            "result": _result_to_dict(results.get(str(roi_id))),
        }

    payload = {
        "ts": ts,
        "trigger": str(trigger),
        "overall_ok": None if overall_ok is None else bool(overall_ok),
        "raw": "raw.png",
        "overlay": "overlay.png",
        "results": roi_payload,
    }

    result_path = os.path.join(run_dir, "result.json")
    temp_path = result_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, result_path)

    # _prune_inspection_capture_dirs(root_dir, keep=max(1, int(snapshot_keep)))
    return run_dir

def handle_sample_keys(
    key,
    frame_gray8,
    vis_bgr,
    *,
    edit_mode,
    roi_mgr,
    data_dir,
    snapshot_keep=10,
    last_results=None,
):
    if edit_mode:
        return False, None

    if key not in (ord("t"), ord("T"), ord("k"), ord("K"), ord("n"), ord("N")):
        return False, None

    if frame_gray8 is None or vis_bgr is None:
        return True, "[SAMPLE SAVE FAILED] frame is empty"

    is_t = key in (ord("t"), ord("T"))
    if is_t:
        roi_id = getattr(roi_mgr, "selected_id", None)
        if roi_id is None:
            return True, "[TEMPLATE SAVE FAILED] no selected ROI"
        crop = crop_roi(frame_gray8, roi_mgr, int(roi_id))
        if crop is None:
            return True, "[TEMPLATE SAVE FAILED] crop is empty"
        template_dir = os.path.join(data_dir, "templates")
        os.makedirs(template_dir, exist_ok=True)
        path = os.path.join(template_dir, f"tape_ok_ROI{int(roi_id)}.png")
        cv2.imwrite(path, crop)
        return True, f"[TEMPLATE SAVED] {path}"

    tag = "OK" if key in (ord("k"), ord("K")) else "NG"
    out_dir = os.path.join(data_dir, "dataset", tag)
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    shared_raw_path = os.path.join(out_dir, f"cap_{ts}_raw.png")
    shared_overlay_path = os.path.join(out_dir, f"cap_{ts}_overlay.png")
    cv2.imwrite(shared_raw_path, frame_gray8)
    cv2.imwrite(shared_overlay_path, vis_bgr)

    results = last_results or {}
    saved_roi_count = 0

    for roi in getattr(roi_mgr, "rois", []):
        roi_id = int(roi.get("id"))
        stem = f"cap_{ts}_ROI{roi_id}"

        raw_path = os.path.join(out_dir, f"{stem}_raw.png")
        overlay_path = os.path.join(out_dir, f"{stem}_overlay.png")
        crop_path = os.path.join(out_dir, f"{stem}_crop.png")
        json_path = os.path.join(out_dir, f"{stem}.json")

        # Keep the existing per-ROI filenames while sharing disk blocks when
        # the filesystem supports hard links.
        _link_or_copy(shared_raw_path, raw_path)
        _link_or_copy(shared_overlay_path, overlay_path)

        crop = crop_roi(frame_gray8, roi_mgr, roi_id)
        if crop is not None:
            cv2.imwrite(crop_path, crop)

        roi_res = results.get(str(roi_id))
        meta = {
            "ts": ts,
            "roi_id": roi_id,
            "tag": tag,
            "raw": raw_path,
            "overlay": overlay_path,
            "crop": crop_path if crop is not None else None,
            "shared_raw": shared_raw_path,
            "shared_overlay": shared_overlay_path,
            "result": {
                "ok": getattr(roi_res, "ok", None),
                "reason": getattr(roi_res, "reason", None),
                "metrics": getattr(roi_res, "metrics", None),
            } if roi_res else None,
        }

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(meta, file, ensure_ascii=False, indent=2)
        saved_roi_count += 1

    prune_capture_groups(out_dir, keep=max(1, int(snapshot_keep)))

    return True, f"[SAMPLE SAVED] {tag} ROI={saved_roi_count} KEEP={int(snapshot_keep)}"
