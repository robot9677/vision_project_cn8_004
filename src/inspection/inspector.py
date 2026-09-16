import os
import json
import time

import cv2
import numpy as np

from .analyzers import run_analyzer
from .preprocess import normalize_by_roi
from .temporal import TemporalMeanFilter
from .aligner import MultiAnchorAligner
from app.app_paths import PROJECT_ROOT
from .recipe import load_recipe, get_roi_cfg, save_recipe
from typing import Dict
from inspection.toolchain import run_toolchain
from inspection.tools_enhance import register_enhance_tools
from inspection.tools_measure import register_measure_tools
from inspection.tools_locate import register_locate_tools
from inspection.tools_identify import register_identify_tools
from inspection.registry.job_registry import JOB_REGISTRY
from inspection.engine.inspect_prepare import prepare_inspection_context
from inspection.engine.inspect_roi_loop import process_all_rois
from inspection.engine.decision_engine import decide_overall
from inspection.engine.job_executor import execute_inspection_job
from inspection.engine.result_model import ROIResult
# ===== START 2026-08-26 : 검사결과 저장/로그백업 구조 변경 =====
from inspection.log_archive_manager import InspectionLogArchiveManager
# ===== END 2026-08-26 : 검사결과 저장/로그백업 구조 변경 =====


def _json_safe_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(item)
            for key, item in value.items()
            if not isinstance(item, np.ndarray)
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _run_presence_job(crop, cfg):
    params = cfg if isinstance(cfg, dict) else {}

    blur_ksize = int(params.get("blur_ksize", 3))
    if blur_ksize < 1:
        blur_ksize = 1
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    threshold_mode = str(params.get("threshold_mode", "fixed")).strip().lower()
    threshold = float(params.get("threshold", 128))
    offset = float(params.get("offset", 0.0))
    invert = bool(params.get("invert", False))

    morph_open = int(params.get("morph_open", 0))
    morph_close = int(params.get("morph_close", 0))
    min_area = int(params.get("min_area", 0))

    img = crop
    if blur_ksize > 1:
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    mean_val = float(np.mean(img))

    if threshold_mode == "mean_offset":
        th_value = mean_val + offset
    elif threshold_mode == "otsu":
        th_value = 0.0
    else:
        th_value = threshold

    th_flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    if threshold_mode == "otsu":
        _ret, bw = cv2.threshold(img, 0, 255, th_flag | cv2.THRESH_OTSU)
        th_value = float(_ret)
    else:
        _ret, bw = cv2.threshold(img, float(th_value), 255, th_flag)

    if morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_open, morph_open))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)

    if morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_close, morph_close))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)

    kept_area = 0
    kept_count = 0
    out = np.zeros_like(bw)

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        out[labels == i] = 255
        kept_area += area
        kept_count += 1

    area_ratio = kept_area / float(out.shape[0] * out.shape[1]) if out.size > 0 else 0.0

    metrics = {
        "presence_area": int(kept_area),
        "presence_count": int(kept_count),
        "presence_ratio": float(area_ratio),
        "th_value": float(th_value),
        "_last_image": out,
    }

    ok = True
    reason = "OK"
    return ok, metrics, reason
   
def _run_none_job(crop, cfg):
    return True, {}, "OK"

def _job_eval_toolchain(ok, metrics, reason, cfg, recipe_default, runtime_cfg):
    job_ok = bool(ok)
    job_reason = "OK" if job_ok else (reason or "FAIL")
    return job_ok, job_reason


def _job_eval_mean_threshold(ok, metrics, reason, cfg, recipe_default, runtime_cfg):
    default_min = float(recipe_default.get("min_mean", 0.0))
    default_max = float(recipe_default.get("max_mean", 255.0))

    min_mean = float(cfg.get("min_mean", default_min))
    max_mean = float(cfg.get("max_mean", default_max))

    use_avg5 = bool(runtime_cfg.get("auto_inspect_avg5", False))
    mean_val = float(metrics.get("mean", 0.0)) if use_avg5 else float(metrics.get("mean_raw", 0.0))

    job_ok = (min_mean <= mean_val <= max_mean)
    job_reason = "OK" if job_ok else ("LOW_MEAN" if mean_val < min_mean else "HIGH_MEAN")
    return bool(job_ok), job_reason


def _job_eval_score_threshold(ok, metrics, reason, cfg, recipe_default, runtime_cfg):
    default_score_thresh = float(recipe_default.get("score_threshold", 0.25))
    score_thresh = float(cfg.get("score_threshold", default_score_thresh))
    score_val = float(metrics.get("score", 0.0))

    job_ok = score_val >= score_thresh
    job_reason = "OK" if job_ok else "LOW_SCORE"
    return bool(job_ok), job_reason

def _job_eval_presence(ok, metrics, reason, cfg, recipe_default, runtime_cfg):
    min_ratio = float(cfg.get("min_ratio", 0.0))
    max_ratio = float(cfg.get("max_ratio", 1.0))
    min_count = int(cfg.get("min_count", 0))
    max_count = int(cfg.get("max_count", 999999))

    ratio = float(metrics.get("presence_ratio", 0.0))
    count = int(metrics.get("presence_count", 0))

    ratio_ok = (min_ratio <= ratio <= max_ratio)
    count_ok = (min_count <= count <= max_count)

    job_ok = ratio_ok and count_ok

    if not ratio_ok:
        if ratio < min_ratio:
            return False, "PRESENCE_RATIO_LOW"
        return False, "PRESENCE_RATIO_HIGH"

    if not count_ok:
        if count < min_count:
            return False, "PRESENCE_COUNT_LOW"
        return False, "PRESENCE_COUNT_HIGH"

    return bool(job_ok), "OK"

def _job_eval_washer(ok, metrics, reason, cfg, recipe_default, runtime_cfg):
    edge_count = int(metrics.get("edge_count", 0))
    min_edge = int(cfg.get("min_edge", 165))

    if edge_count >= min_edge:
        return True, "OK"

    return False, "WASHER_MISSING"

def _job_eval_none(ok, metrics, reason, cfg, recipe_default, runtime_cfg):
    return True, "OK"

JOB_EVALUATORS = {
    "toolchain": _job_eval_toolchain,
    "mean_threshold": _job_eval_mean_threshold,
    "score_threshold": _job_eval_score_threshold,
    "presence": _job_eval_presence,
    "washer_presence": _job_eval_washer, 
    "none": _job_eval_none,
}

def _run_toolchain_job(crop, cfg):
    return run_toolchain(crop, cfg)


def _run_analyzer_job(crop, cfg):
    return run_analyzer(crop, cfg)

JOB_RUNNERS = {
    "toolchain": _run_toolchain_job,
    "mean_threshold": _run_analyzer_job,
    "score_threshold": _run_analyzer_job,
    "presence": _run_presence_job,
    "none": _run_none_job,
}

class Inspector:
    def __init__(self, roi_mgr, recipe_path: str, logs_root: str, runtime_cfg=None):
        self.roi_mgr = roi_mgr
        self.recipe_path = recipe_path
        self.logs_root = logs_root
        self.recipe = load_recipe(recipe_path)
        print("[RECIPE]", "STATIC", recipe_path)
        decision = self.recipe.get("decision", {}) if isinstance(self.recipe, dict) else {}
        self.decision_mode = decision.get("mode", "any_fail_is_ng")
        self.decision_max_fail = int(decision.get("max_fail", 0))
        self.mean_filters = {}
        self.runtime_cfg = runtime_cfg or {}
        self.debug_view_enabled = bool(self.runtime_cfg.get("debug_view_enabled", True))
        self.debug_view_roi_id = str(self.runtime_cfg.get("debug_view_roi_id", "1"))
        self.debug_view_scale = float(self.runtime_cfg.get("debug_view_scale", 1))
        self.aligner = MultiAnchorAligner(
            runtime_cfg=self.runtime_cfg,
            product_profile=(self.runtime_cfg.get("_product_profile") or {}),
            project_root=PROJECT_ROOT,
        )
        self.debug_tiles = {}
        self.debug_grid = None
        self.baseline_path = os.path.join(os.path.dirname(recipe_path), "baseline_profile.json")
        if os.path.exists(self.baseline_path):
            with open(self.baseline_path, "r") as f:
                self.baseline = json.load(f)
        else:
            self.baseline = None

        self._roi_debug_window_init = False
        # ===== START 2026-09-16 : 검사결과 저장/로그백업 공통 안정화 =====
        equipment_cfg_path = os.path.join(PROJECT_ROOT, "data", "config", "equipment_config.json")
        try:
            with open(equipment_cfg_path, encoding="utf-8") as f:
                equipment_cfg = json.load(f)
        except Exception:
            equipment_cfg = {}
        self.log_archive = InspectionLogArchiveManager(
            logs_root=self.logs_root,
            project_root=PROJECT_ROOT,
            equipment_name=equipment_cfg.get("equipment_name", "VISION"),
            keep_days=int(equipment_cfg.get("inspection_day_keep", 20)),
        )
        # ZIP 생성과 Drive 통신은 main_vp의 daemon worker에서만 수행한다.
        # Inspector 생성 과정은 카메라/PLC 시작을 지연시키지 않는다.
        self.email_notifier = None
        # ===== END 2026-09-16 : 검사결과 저장/로그백업 공통 안정화 =====
        self._save_run_counter = 0

        register_enhance_tools()
        register_measure_tools()
        register_locate_tools()
        register_identify_tools()

    def _get_mean_filter(self, roi_id):
        key = str(roi_id)
        if key not in self.mean_filters:
            self.mean_filters[key] = TemporalMeanFilter(win=5)
        return self.mean_filters[key]

    def reset_temporal_filters(self):
        for mean_filter in self.mean_filters.values():
            mean_filter.reset()
        self.mean_filters.clear()

    def _run_inspection_job(
        self,
        crop,
        cfg,
        recipe_default,
        runtime_cfg,
        mean_filter,
        norm_gain,
        roi_dx,
        roi_dy,
        roi_dangle,
        pose,
        trk_score,
    ):
        return execute_inspection_job(
            inspector=self,
            crop=crop,
            cfg=cfg,
            recipe_default=recipe_default,
            runtime_cfg=runtime_cfg,
            mean_filter=mean_filter,
            norm_gain=norm_gain,
            roi_dx=roi_dx,
            roi_dy=roi_dy,
            roi_dangle=roi_dangle,
            pose=pose,
            trk_score=trk_score,
            job_registry=JOB_REGISTRY,
            job_runners=JOB_RUNNERS,
            job_evaluators=JOB_EVALUATORS,
            default_runner=_run_analyzer_job,
            default_evaluator=_job_eval_toolchain,
        )

    def reload_recipe(self):
        self.recipe = load_recipe(self.recipe_path)

    def _show_debug_view(self, roi_id, raw_crop=None, last_img=None):
        if not self.debug_view_enabled:
            return

        def _to_bgr(im):
            if im is None or not isinstance(im, np.ndarray) or im.size == 0:
                return None
            out = im.copy()
            if out.ndim == 2:
                out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            return out

        raw_vis = _to_bgr(raw_crop)
        last_vis = _to_bgr(last_img)

        # 메인 Service 패널에 들어가는 compact ROI RAW/LAST tile.
        cell_w = 180
        cell_h = 120

        def _fit_cell(im, title, color):
            canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            if im is not None:
                h, w = im.shape[:2]
                scale = min((cell_w - 8) / max(1, w), (cell_h - 24) / max(1, h))
                nw = max(1, int(w * scale))
                nh = max(1, int(h * scale))
                resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_NEAREST)
                x0 = (cell_w - nw)  // 2
                y0 = 20 + (cell_h - 20 - nh)  // 2
                canvas[y0:y0+nh, x0:x0+nw] = resized
            cv2.putText(
                canvas,
                title,
                (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(canvas, (0, 0), (cell_w - 1, cell_h - 1), (60, 60, 60), 1)
            return canvas

        left = _fit_cell(raw_vis, f"ROI{roi_id} RAW", (0, 255, 255))
        right = _fit_cell(last_vis, f"ROI{roi_id} LAST", (0, 255, 0))
        pair = cv2.hconcat([left, right])

        self.debug_tiles[str(roi_id)] = pair

        keys = sorted(self.debug_tiles.keys(), key=lambda x: int(x))
        pairs = [self.debug_tiles[k] for k in keys]

        per_row = 2
        blank = np.zeros_like(pair)
        rows = []

        for i in range(0, len(pairs), per_row):
            row = pairs[i:i+per_row]
            while len(row) < per_row:
                row.append(blank.copy())
            rows.append(cv2.hconcat(row))

        grid = cv2.vconcat(rows)
        # 별도 OpenCV 창을 띄우지 않는다.
        # 메인 비전 화면의 Service/PLC 패널에서 이 이미지를 사용한다.
        self.debug_grid = grid

    def get_debug_grid(self):
        grid = getattr(self, "debug_grid", None)
        if grid is None or not isinstance(grid, np.ndarray) or grid.size == 0:
            return None
        return grid.copy()

    def _crop_rotated(self, frame_gray8, roi, dx=0, dy=0, dangle=0.0):
        H, W = frame_gray8.shape[:2]

        x = float(roi.get("x", 0)) + float(dx)
        y = float(roi.get("y", 0)) + float(dy)
        w = max(1, int(roi.get("w", 1)))
        h = max(1, int(roi.get("h", 1)))
        angle = float(roi.get("angle", 0.0)) + float(dangle)

        cx = x + w / 2.0
        cy = y + h / 2.0

        rect = ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect).astype(np.float32)

        min_x = max(0, int(np.floor(np.min(box[:, 0]))))
        min_y = max(0, int(np.floor(np.min(box[:, 1]))))
        max_x = min(W, int(np.ceil(np.max(box[:, 0]))))
        max_y = min(H, int(np.ceil(np.max(box[:, 1]))))

        if max_x <= min_x or max_y <= min_y:
            return None

        patch = frame_gray8[min_y:max_y, min_x:max_x]
        if patch is None or patch.size == 0:
            return None

        local_cx = cx - min_x
        local_cy = cy - min_y

        M = cv2.getRotationMatrix2D((local_cx, local_cy), angle, 1.0)
        rotated = cv2.warpAffine(
            patch,
            M,
            (patch.shape[1], patch.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        crop = cv2.getRectSubPix(rotated, (w, h), (local_cx, local_cy))
        return crop

    def inspect(self, frame_gray8: np.ndarray, auto_mode=False):
        if self.debug_view_enabled:
            self.debug_tiles = {}
            self.debug_grid = None

        results: Dict[str, ROIResult] = {}

        prep = prepare_inspection_context(
            inspector=self,
            frame_gray8=frame_gray8,
            auto_mode=auto_mode,
        )

        frame_gray8 = prep["frame_gray8"]
        norm_gain = prep["norm_gain"]
        trk_score = prep["trk_score"]
        align_result = prep["align_result"]

        results = process_all_rois(
            inspector=self,
            frame_gray8=frame_gray8,
            align_result=align_result,
            norm_gain=norm_gain,
            trk_score=trk_score,
            auto_mode=auto_mode,
        )

        overall_ok = decide_overall(
            recipe=self.recipe,
            results=results,
            auto_mode=auto_mode,
        )
        return overall_ok, results

    # ===== START 2026-09-16 : 검사결과 저장/로그백업 공통 안정화 =====
    def save_run(self, frame_gray8: np.ndarray, overlay_bgr: np.ndarray, overall_ok: bool, results: Dict[str, ROIResult]) -> str:
        """Save every OK/NG inspection. overlay_bgr is intentionally not persisted."""
        # run_inspect_once()가 실제 판정에 사용한 평균 프레임을 직접 전달한다.
        # 제거된 4K 전용 속성(last_inspection_frame_gray8,
        # _inspection_scale_x/y)을 참조하면 RAW/결과 저장이 전부 실패한다.
        inspection_frame = frame_gray8

        out = {
            "overall_ok": bool(overall_ok),
            "ts": time.time(),
            "soak_test": bool(self.runtime_cfg.get("_service_soak_active", False)),
            "frame_info": {
                "roi_coordinate_width": int(self.roi_mgr.W),
                "roi_coordinate_height": int(self.roi_mgr.H),
                "inspection_width": int(inspection_frame.shape[1]),
                "inspection_height": int(inspection_frame.shape[0]),
            },
            "results": {
                k: {
                    "roi_id": str(v.roi_id), "ok": bool(v.ok), "reason": v.reason,
                    "metrics": {k2: _json_safe_value(v2) for k2, v2 in v.metrics.items() if not isinstance(v2, np.ndarray)},
                } for k, v in results.items()
            },
        }
        raw_path, _result_path, _day_dir = self.log_archive.save_run(inspection_frame, overall_ok, results, out)
        return raw_path
    # ===== END 2026-09-16 : 검사결과 저장/로그백업 공통 안정화 =====

    # def save_recipe(path: str, recipe: Dict[str, Any]) -> None:
    #     import os, json
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     with open(path, "w", encoding="utf-8") as f:
    #         json.dump(recipe, f, ensure_ascii=False, indent=2)

    def autotune_recipe_from_frame(self, frame_gray8, save_path=None):
        import copy
        target_mean = float(self.runtime_cfg.get("autotune_target_mean", 50.0))
        margin = float(self.runtime_cfg.get("autotune_margin", 10.0))
        save_path = save_path or self.recipe_path
        
        """
        현재 프레임 기준으로 ROI별 mean을 읽고
        recipe_static.json(overrides)에 ROI별 min/max를 자동 생성해서 저장
        """
        # 1) 기준 ROI로 정규화(현재 inspect랑 동일 로직)
        ref = self.roi_mgr.get_selected()
        if ref is not None:
            ref_crop = self.roi_mgr.crop(frame_gray8, ref["id"])
            frame_n, _gain = normalize_by_roi(frame_gray8, ref_crop, target_mean=target_mean)
        else:
            frame_n = frame_gray8

        overrides = {}
        for roi in getattr(self.roi_mgr, "rois", []):
            roi_id = roi.get("id")
            crop = self.roi_mgr.crop(frame_n, roi_id)
            if crop is None or crop.size == 0:
                continue
            m = float(np.mean(crop))
            mn = max(0.0, m - margin)
            mx = min(255.0, m + margin)
            cfg = get_roi_cfg(self.recipe, roi_id)

            if "tools" in cfg:
                roi_cfg = copy.deepcopy(cfg)

                for step in roi_cfg.get("tools", []):
                    tool_name = str(step.get("tool", "")).strip().lower()
                    params = step.get("params") or {}

                    if tool_name == "measure.blob_count":
                        ok_bt, metrics_bt, _reason_bt = run_toolchain(crop, {
                            "tools": roi_cfg.get("tools", []),
                            "tool_decision": roi_cfg.get("tool_decision", "all_ok"),
                        })

                        blob_count = int(metrics_bt.get("blob_count", 0))
                        areas = metrics_bt.get("blob_areas_kept") or []

                        # expected 는 자동 변경하지 않음
                        # 정상 기준 개수는 사용자가 직접 정하거나 기존 값을 유지

                        if areas:
                            params["area_min"] = int(max(1, min(areas) * 0.7))
                            params["area_max"] = int(max(areas) * 1.3)

                        step["params"] = params

                overrides[f"ROI{roi_id}"] = roi_cfg
            else:
                overrides[f"ROI{roi_id}"] = {
                    "type": "mean_threshold",
                    "min_mean": float(mn),
                    "max_mean": float(mx),
                }
        # 기존 recipe를 베이스로 복사해서, overrides만 교체
        base = self.recipe if isinstance(self.recipe, dict) else {}
        recipe = copy.deepcopy(base)

        # default는 없으면 넣고, 있으면 유지(원하면 여기서만 type 보정)
        recipe.setdefault("default", {"type": "mean_threshold", "min_mean": 0.0, "max_mean": 255.0})

        # 핵심: AUTO는 overrides만 갱신
        recipe["overrides"] = overrides

        # decision은 절대 건드리지 않음(없으면 기본값만 세팅)
        recipe.setdefault("decision", {"mode": "any_fail_is_ng"})

        # print("[DBG AUTO] decision(before save) =", recipe.get("decision"))
        save_recipe(save_path, recipe)
        # print("[DBG AUTO] saved to", save_path)
        # print("[DBG AUTO] decision(after save read) =", load_recipe(save_path).get("decision"))
        self.recipe = recipe  # 즉시 반영
        return recipe
    
    def reset_tracker_template(self):
        if getattr(self, "aligner", None) is not None:
            self.aligner.reset_templates()

    # ===== START 2026-08-26 : 검사결과 저장/로그백업 구조 변경 =====
    def log_result(self, overall_ok, results):
        # Detailed inspect logs are separated from raw/result evidence.
        return self.log_archive.save_inspect_summary(overall_ok, results)
    # ===== END 2026-08-26 : 검사결과 저장/로그백업 구조 변경 =====

    def _check_baseline(self, roi_id, metrics):
        if not self.baseline:
            return True, None

        roi_name = f"ROI{roi_id}"
        if roi_name not in self.baseline:
            return True, None

        data = self.baseline[roi_name]

        # ROI2~5
        if "dark_ratio" in data:
            v = metrics.get("dark_ratio")
            if v is None:
                return True, None

            low = data["dark_ratio"]["low"]
            high = data["dark_ratio"]["high"]

            if v < low:
                return False, "BASELINE_LOW"
            if v > high:
                return False, "BASELINE_HIGH"

            return True, None

        # ROI6
        if "blob_count" in data:
            v = metrics.get("blob_count")
            if v is None:
                v = metrics.get("blob")

            if v is None:
                return True, None

            low = data["blob_count"]["low"]
            high = data["blob_count"]["high"]

            if v < low:
                return False, "BASELINE_LOW"
            if v > high:
                return False, "BASELINE_HIGH"

            return True, None

        return True, None