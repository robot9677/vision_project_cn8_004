# src/ui/overlay_clean.py
import cv2
import numpy as np
from ui import ui_config as cfg

def _roi_text_style(img, base_scale=0.30):
    """
    ROI 전용 텍스트 크기 자동 조절.
    - 해상도에 따라 scale만 비례 증가
    - thickness는 항상 1 유지: bold 방지
    """
    h, w = img.shape[:2]

    sx = float(w) / 1280.0
    sy = float(h) / 720.0
    s = min(sx, sy)

    scale = base_scale * s
    scale = max(0.28, min(0.46, scale))

    thickness = 1

    line_gap = int(round(13 * s))
    line_gap = max(12, min(20, line_gap))

    return float(scale), int(thickness), int(line_gap)

# --- basic drawing helpers ---
def draw_text(img, text, pos, color=None, scale=None, thickness=None, align="lt"):
    if color is None:
        color = cfg.COLOR_TEXT
    if scale is None:
        scale = cfg.FONT_SCALE
    if thickness is None:
        thickness = cfg.FONT_THICK

    (tw, th), baseline = cv2.getTextSize(str(text), cfg.FONT, scale, thickness)
    x, y = int(pos[0]), int(pos[1])

    if align == "ct":
        x = int(x - tw / 2)
    elif align == "rt":
        x = int(x - tw)
 
    # align vertical: use baseline approx (putText uses bottom-left)
    y_draw = int(y + th / 2)
    cv2.putText(img, str(text), (x, y_draw), cfg.FONT, scale, color, int(thickness), cfg.LINE_TYPE)

def draw_rect(img, pt1, pt2, color=None, thickness=1, fill=False):
    if color is None:
        color = cfg.COLOR_TEXT
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    if fill:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, int(thickness), lineType=cfg.LINE_TYPE)

def draw_dashed_circle(img, center, radius, color, thickness=1, dash_len=10):
    import math
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)

    for i in range(0, 360, dash_len * 2):
        a1 = math.radians(i)
        a2 = math.radians(i + dash_len)

        x1 = int(cx + r * math.cos(a1))
        y1 = int(cy + r * math.sin(a1))
        x2 = int(cx + r * math.cos(a2))
        y2 = int(cy + r * math.sin(a2))

        cv2.line(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

# --- higher-level UI elements ---
def draw_status_bar(img, text):
    h, w = img.shape[:2]
    bar_h = cfg.STATUS_HEIGHT if hasattr(cfg, "STATUS_HEIGHT") else 40
    # dark background
    # dark = img.copy()
    # draw_rect(dark, (0, 0), (w, bar_h), color=cfg.STATUS_BG if hasattr(cfg, "STATUS_BG") else (0,0,0), fill=True)
    # alpha = 0.6
    # cv2.addWeighted(dark, alpha, img, 1.0 - alpha, 0, img)
    draw_text(img, text, (cfg.MARGIN if hasattr(cfg, "MARGIN") else 8, int(bar_h/2)+6), color=cfg.COLOR_TEXT, scale=cfg.FONT_SCALE, thickness=cfg.FONT_THICK, align="lt")

def _draw_circle_overlay(img, x, y, w, h, metrics):
    if not (metrics and isinstance(metrics, dict)):
        return

    # --- detected circles / ref ---
    if "circles" in metrics:
        ref_idx = metrics.get("calibration_ref_index", None)

        for i, c in enumerate(metrics["circles"]):
            if isinstance(c, dict):
                ccx = int(c.get("x", 0)) + x
                ccy = int(c.get("y", 0)) + y
                rr = int(c.get("r", 0))
            else:
                ccx = int(c[0]) + x
                ccy = int(c[1]) + y
                rr = int(c[2])

            draw_dashed_circle(
                img,
                (ccx, ccy),
                rr,
                (0, 255, 0),
                thickness=1,
                dash_len=8,
            )

            if ref_idx is not None and int(ref_idx) == i:
                cv2.putText(
                    img,
                    "REF",
                    (ccx - 18, ccy - rr - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    img,
                    (ccx, ccy),
                    4,
                    (0, 255, 255),
                    -1,
                    lineType=cv2.LINE_AA,
                )

    # --- mm / px label ---
    mm_list = metrics.get("diameters_mm")
    px_list = metrics.get("diameters_px")

    for i, c in enumerate(metrics.get("circles", [])):
        if isinstance(c, dict):
            cx = int(c["x"]) + x
            cy = int(c["y"]) + y
        else:
            cx = int(c[0]) + x
            cy = int(c[1]) + y

        text = None
        if isinstance(mm_list, list) and i < len(mm_list):
            text = f"{float(mm_list[i]):.1f} mm"
        elif isinstance(px_list, list) and i < len(px_list):
            text = f"{float(px_list[i]):.1f} px"

        if not text:
            continue

        col = (0, 255, 0)
        flags = metrics.get("judge_flags")
        if isinstance(flags, list) and i < len(flags) and not flags[i]:
            col = (0, 0, 255)

        cv2.putText(
            img,
            text,
            (cx - 40, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            col,
            1,
            cv2.LINE_AA,
        )

    # --- circle stats ---
    unit = metrics.get("unit_mode", "px")
    if unit == "mm":
        avg = metrics.get("diameter_mm_avg", None)
        dmin = metrics.get("diameter_mm_min", None)
        dmax = metrics.get("diameter_mm_max", None)
        txt = (
            f"AVG:{float(avg):.2f}  MIN:{float(dmin):.2f}  MAX:{float(dmax):.2f}"
            if avg is not None and dmin is not None and dmax is not None
            else None
        )
    else:
        avg = metrics.get("diameter_px_avg", None)
        dmin = metrics.get("diameter_px_min", None)
        dmax = metrics.get("diameter_px_max", None)
        txt = (
            f"AVG:{float(avg):.1f}px  MIN:{float(dmin):.1f}px  MAX:{float(dmax):.1f}px"
            if avg is not None and dmin is not None and dmax is not None
            else None
        )

    if txt:
        cv2.putText(
            img,
            txt,
            (x + 10, y + h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # --- spec ---
    target = metrics.get("target_mm")
    tol = metrics.get("tol_mm")
    if target is not None and tol is not None:
        try:
            lo = float(target) - float(tol)
            hi = float(target) + float(tol)
            txt = f"SPEC:{lo:.2f}~{hi:.2f} mm"
            cv2.putText(
                img,
                txt,
                (x + 10, y + h + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 0),
                1,
                cv2.LINE_AA,
            )
        except Exception:
            pass

    # --- result ---
    judge = metrics.get("judge_value", None)
    if judge is not None and target is not None and tol is not None:
        try:
            judge = float(judge)
            target = float(target)
            tol = float(tol)
            if (target - tol) <= judge <= (target + tol):
                col = (0, 255, 0)
                txt = "OK"
            else:
                col = (0, 0, 255)
                txt = "NG"

            cv2.putText(
                img,
                txt,
                (x + w - 40, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                col,
                2,
                cv2.LINE_AA,
            )
        except Exception:
            pass

def _roi_local_to_global(x, y, w, h, angle_deg, px, py):
    cx = x + w / 2.0
    cy = y + h / 2.0

    lx = float(px) - (w / 2.0)
    ly = float(py) - (h / 2.0)

    th = np.radians(float(angle_deg))
    c = np.cos(th)
    s = np.sin(th)

    gx = cx + (lx * c - ly * s)
    gy = cy + (lx * s + ly * c)
    return int(round(gx)), int(round(gy))

def _draw_line_overlay(img, x, y, w, h, angle, metrics):
    if not (metrics and isinstance(metrics, dict)):
        return

    p1 = metrics.get("line_p1")
    p2 = metrics.get("line_p2")
    ctr = metrics.get("line_center")

    if p1 is None or p2 is None:
        lines = metrics.get("lines") or []
        if not lines:
            return
        ln = lines[0]
        p1 = [int(ln["x1"]), int(ln["y1"])]
        p2 = [int(ln["x2"]), int(ln["y2"])]
        ctr = [
            float((ln["x1"] + ln["x2"]) / 2.0),
            float((ln["y1"] + ln["y2"]) / 2.0),
        ]

    gp1 = _roi_local_to_global(x, y, w, h, angle, p1[0], p1[1])
    gp2 = _roi_local_to_global(x, y, w, h, angle, p2[0], p2[1])

    cv2.line(img, gp1, gp2, (0, 255, 255), 2, cv2.LINE_AA)

    if ctr is not None:
        gc = _roi_local_to_global(x, y, w, h, angle, ctr[0], ctr[1])
        cv2.circle(img, gc, 4, (0, 255, 255), -1, lineType=cv2.LINE_AA)

        dx = gp2[0] - gp1[0]
        dy = gp2[1] - gp1[1]
        ang = float(np.degrees(np.arctan2(dy, dx)))
        while ang > 90.0:
            ang -= 180.0
        while ang <= -90.0:
            ang += 180.0

        txt = f"{ang:.1f} deg"
        cv2.putText(
            img,
            txt,
            (gc[0] + 6, gc[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,0.45,(0, 255, 255),1,cv2.LINE_AA,
        )

    target = metrics.get("line_angle_target_deg", None)
    tol = metrics.get("line_angle_tol_deg", None)
    judge_ok = metrics.get("line_angle_ok", None)

    if target is not None and tol is not None:
        lo = float(target) - float(tol)
        hi = float(target) + float(tol)

        spec_txt = f"SPEC:{lo:.1f}~{hi:.1f} deg"
        cv2.putText(
            img,
            spec_txt,
            (x + 10, y + h + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 0),
            1,
            cv2.LINE_AA,
        )

        if judge_ok is not None:
            col = (0, 255, 0) if bool(judge_ok) else (0, 0, 255)
            txt2 = "OK" if bool(judge_ok) else "NG"
            cv2.putText(
                img,
                txt2,
                (x + w - 40, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                col,
                2,
                cv2.LINE_AA,
            )

def _draw_distance_overlay(img, x, y, h, metrics):
    if not (metrics and isinstance(metrics, dict)):
        return

    pairs = metrics.get("distance_pairs") or []
    has_single_judge = metrics.get("distance_judge_value", None) is not None
    has_multi_judge = isinstance(metrics.get("distance_judge_flags"), list)

    if pairs and not has_single_judge and not has_multi_judge:
        try:
            rows = []
            for p in pairs:
                pi = int(p.get("i", -1))
                pj = int(p.get("j", -1))

                x1 = p.get("x1", None)
                y1 = p.get("y1", None)
                x2 = p.get("x2", None)
                y2 = p.get("y2", None)

                if None not in (x1, y1, x2, y2):
                    p1 = (int(x + float(x1)), int(y + float(y1)))
                    p2 = (int(x + float(x2)), int(y + float(y2)))
                    cv2.line(img, p1, p2, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(img, p1, 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                    cv2.circle(img, p2, 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)

                if "distance_mm" in p:
                    rows.append(f"D[{pi}-{pj}]: {float(p['distance_mm']):.2f} mm")
                else:
                    rows.append(f"D[{pi}-{pj}]: {float(p.get('distance_px', 0.0)):.1f} px")

            for row_idx, txt in enumerate(rows[:3]):
                cv2.putText(
                    img,
                    txt,
                    (x + 10, y + h + 30 + (row_idx * 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        except Exception:
            pass

    details = metrics.get("distance_judge_details") or []
    if details:
        try:
            for row_idx, d in enumerate(details[:3]):
                col = (0, 255, 0) if bool(d.get("ok", False)) else (0, 0, 255)

                lx1 = d.get("x1", None)
                ly1 = d.get("y1", None)
                lx2 = d.get("x2", None)
                ly2 = d.get("y2", None)

                if None not in (lx1, ly1, lx2, ly2):
                    p1 = (int(x + float(lx1)), int(y + float(ly1)))
                    p2 = (int(x + float(lx2)), int(y + float(ly2)))
                    cv2.line(img, p1, p2, col, 1, cv2.LINE_AA)
                    cv2.circle(img, p1, 4, col, -1, lineType=cv2.LINE_AA)
                    cv2.circle(img, p2, 4, col, -1, lineType=cv2.LINE_AA)

                unit = str(d.get("unit", "")).strip().lower()
                val = float(d.get("value", 0.0))
                target = float(d.get("target", 0.0))
                tol = float(d.get("tol", 0.0))
                di = int(d.get("i", -1))
                dj = int(d.get("j", -1))

                if unit == "mm":
                    txt = f"D[{di}-{dj}]: {val:.2f} mm ({target - tol:.2f}~{target + tol:.2f})"
                else:
                    txt = f"D[{di}-{dj}]: {val:.1f} px ({target - tol:.1f}~{target + tol:.1f})"

                cv2.putText(
                    img,
                    txt,
                    (x + 10, y + h + 30 + (row_idx * 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    col,
                    1,
                    cv2.LINE_AA,
                )
        except Exception:
            pass

    if not details:
        dval = metrics.get("distance_judge_value", None)
        dunit = str(metrics.get("distance_judge_unit", "")).strip().lower()
        di = metrics.get("distance_judge_i", None)
        dj = metrics.get("distance_judge_j", None)

        if dval is not None and di is not None and dj is not None:
            try:
                txt = None
                col = (0, 255, 0)

                if dunit == "mm":
                    target = metrics.get("distance_target_mm", None)
                    tol = metrics.get("distance_tol_mm", None)
                    if target is not None and tol is not None:
                        lo = float(target) - float(tol)
                        hi = float(target) + float(tol)
                        val = float(dval)
                        col = (0, 255, 0) if (lo <= val <= hi) else (0, 0, 255)
                        txt = f"D[{int(di)}-{int(dj)}]: {val:.2f} mm ({lo:.2f}~{hi:.2f})"

                elif dunit == "px":
                    target = metrics.get("distance_target_px", None)
                    tol = metrics.get("distance_tol_px", None)
                    if target is not None and tol is not None:
                        lo = float(target) - float(tol)
                        hi = float(target) + float(tol)
                        val = float(dval)
                        col = (0, 255, 0) if (lo <= val <= hi) else (0, 0, 255)
                        txt = f"D[{int(di)}-{int(dj)}]: {val:.1f} px ({lo:.1f}~{hi:.1f})"

                if txt:
                    lx1 = metrics.get("distance_judge_x1", None)
                    ly1 = metrics.get("distance_judge_y1", None)
                    lx2 = metrics.get("distance_judge_x2", None)
                    ly2 = metrics.get("distance_judge_y2", None)

                    if None not in (lx1, ly1, lx2, ly2):
                        p1 = (int(x + float(lx1)), int(y + float(ly1)))
                        p2 = (int(x + float(lx2)), int(y + float(ly2)))
                        cv2.line(img, p1, p2, col, 1, cv2.LINE_AA)
                        cv2.circle(img, p1, 4, col, -1, lineType=cv2.LINE_AA)
                        cv2.circle(img, p2, 4, col, -1, lineType=cv2.LINE_AA)

                    cv2.putText(
                        img,
                        txt,
                        (x + 10, y + h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        col,
                        1,
                        cv2.LINE_AA,
                    )
            except Exception:
                pass


def _draw_roi_distance_links(img, x, y, h, metrics):
    if not (metrics and isinstance(metrics, dict)):
        return

    roi_links = metrics.get("roi_distance_links") or []
    if not roi_links:
        return

    try:
        for row_idx, link in enumerate(roi_links[:3]):
            link_ok = link.get("ok", None)
            col = (0, 255, 255)
            if link_ok is True:
                col = (0, 255, 0)
            elif link_ok is False:
                col = (0, 0, 255)

            p1 = (int(float(link.get("x1", 0))), int(float(link.get("y1", 0))))
            p2 = (int(float(link.get("x2", 0))), int(float(link.get("y2", 0))))

            cv2.line(img, p1, p2, col, 1, cv2.LINE_AA)
            cv2.circle(img, p1, 4, col, -1, lineType=cv2.LINE_AA)
            cv2.circle(img, p2, 4, col, -1, lineType=cv2.LINE_AA)

            txt = None
            from_roi_id = int(link.get("from_roi_id", -1))
            to_roi_id = int(link.get("to_roi_id", -1))
            judge_unit = str(link.get("judge_unit", "")).strip().lower()

            if judge_unit == "mm" and "distance_mm" in link:
                val = float(link["distance_mm"])
                target = link.get("target_mm", None)
                tol = link.get("tol_mm", None)
                if target is not None and tol is not None:
                    lo = float(target) - float(tol)
                    hi = float(target) + float(tol)
                    txt = f"R{from_roi_id}-{to_roi_id} {val:.2f} ({lo:.2f}~{hi:.2f})"
                else:
                    txt = f"R{from_roi_id}-{to_roi_id} {val:.2f}"

            elif judge_unit == "px":
                val = float(link.get("distance_px", 0.0))
                target = link.get("target_px", None)
                tol = link.get("tol_px", None)
                if target is not None and tol is not None:
                    lo = float(target) - float(tol)
                    hi = float(target) + float(tol)
                    txt = f"R{from_roi_id}-{to_roi_id} {val:.1f} ({lo:.1f}~{hi:.1f})"
                else:
                    txt = f"R{from_roi_id}-{to_roi_id} {val:.1f}"
            elif "distance_mm" in link:
                txt = f"R{from_roi_id}-{to_roi_id} {float(link['distance_mm']):.2f}"
            else:
                txt = f"R{from_roi_id}-{to_roi_id} {float(link.get('distance_px', 0.0)):.1f}"

            if not txt:
                continue

    except Exception:
        pass

def _draw_roi_distance_list(img, roi_results):
    if not isinstance(roi_results, dict):
        return

    rows = []
    seen = set()

    for k in sorted(roi_results.keys(), key=lambda v: int(v) if str(v).isdigit() else 9999):
        rv = roi_results.get(k)
        if isinstance(rv, dict):
            metrics = rv.get("metrics")
        else:
            metrics = getattr(rv, "metrics", None)

        if not isinstance(metrics, dict):
            continue

        for link in (metrics.get("roi_distance_links") or []):
            from_roi_id = int(link.get("from_roi_id", -1))
            to_roi_id = int(link.get("to_roi_id", -1))
            pair = (from_roi_id, to_roi_id)

            if pair in seen:
                continue
            seen.add(pair)

            col = (0, 255, 255)
            if link.get("ok", None) is True:
                col = (0, 255, 0)
            elif link.get("ok", None) is False:
                col = (0, 0, 255)

            txt = None
            judge_unit = str(link.get("judge_unit", "")).strip().lower()

            if judge_unit == "mm" and "distance_mm" in link:
                val = float(link["distance_mm"])
                target = link.get("target_mm", None)
                tol = link.get("tol_mm", None)
                if target is not None and tol is not None:
                    lo = float(target) - float(tol)
                    hi = float(target) + float(tol)
                    txt = f"R{from_roi_id}-{to_roi_id} {val:.2f} ({lo:.2f}~{hi:.2f})"
                else:
                    txt = f"R{from_roi_id}-{to_roi_id} {val:.2f}"
            elif judge_unit == "px":
                val = float(link.get("distance_px", 0.0))
                target = link.get("target_px", None)
                tol = link.get("tol_px", None)
                if target is not None and tol is not None:
                    lo = float(target) - float(tol)
                    hi = float(target) + float(tol)
                    txt = f"R{from_roi_id}-{to_roi_id} {val:.1f} ({lo:.1f}~{hi:.1f})"
                else:
                    txt = f"R{from_roi_id}-{to_roi_id} {val:.1f}"
            elif "distance_mm" in link:
                txt = f"R{from_roi_id}-{to_roi_id} {float(link['distance_mm']):.2f}"
            else:
                txt = f"R{from_roi_id}-{to_roi_id} {float(link.get('distance_px', 0.0)):.1f}"

            if txt:
                rows.append((txt, col))

    if not rows:
        return

    h_img, w_img = img.shape[:2]
    start_x = 18
    start_y = h_img - 110 - ((len(rows) - 1) * 15)

    for i, (txt, col) in enumerate(rows[:6]):
        ty = start_y + (i * 15)

        cv2.putText(
            img,
            txt,
            (start_x, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            txt,
            (start_x, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            col,
            1,
            cv2.LINE_AA,
        )

def _draw_qr_overlay(
    img,
    rois,
    x,
    y,
    h,
    ok,
    reason,
    color,
    metrics,
    roi_text_scale,
    roi_text_thickness,
):
    qr_overlay_text = ""
    qr_overlay_color = color

    if metrics and isinstance(metrics, dict):
        qr_text = str(metrics.get("qr_text", "") or "").strip()
        qr_detected = bool(metrics.get("qr_detected", False))

        if qr_detected or ("QR" in str(reason).upper()):
            if ok:
                if qr_text:
                    qr_overlay_text = f"OK: {qr_text}"
                else:
                    qr_overlay_text = "OK: QR SCAN OK"
                qr_overlay_color = cfg.COLOR_OK
            else:
                if reason:
                    qr_overlay_text = str(reason)
                else:
                    qr_overlay_text = "NG: QR SCAN FAIL"
                qr_overlay_color = cfg.COLOR_NG

    if not qr_overlay_text:
        return

    roi1 = next((rr for rr in rois if int(rr.get("id", 0)) == 1), None)

    qx = int(x + 2)
    qy = int(y + h + 18)

    if roi1:
        if "rect" in roi1 and roi1.get("rect") is not None:
            rx1, ry1, rw1, rh1 = roi1["rect"]
        else:
            rx1 = int(roi1.get("x", 0))
            ry1 = int(roi1.get("y", 0))
            rw1 = int(roi1.get("w", 0))
            rh1 = int(roi1.get("h", 0))

        qx = int(rx1 + 2)
        qy = int(ry1 + rh1 + 18)

    draw_text(
        img,
        qr_overlay_text,
        (qx, qy),
        color=qr_overlay_color,
        scale=roi_text_scale * 2.0,
        thickness=roi_text_thickness,
        align="lt",
    )

def _build_roi_label_lines(
    *,
    roi_id,
    label,
    x,
    y,
    w,
    h,
    angle,
    active_id,
    compact,
    show_metrics,
    roi_results,
    rid_str,
):
    line1 = f"{label}" if roi_id is not None else label
    line2 = "" if compact else f"x:{x} y:{y} w:{w} h:{h} a:{angle:.1f}"
    line3 = "SELECTED" if (roi_id == active_id and not compact) else ""

    if show_metrics and roi_results is not None:
        rv = roi_results.get(rid_str) if isinstance(roi_results, dict) else None
        if rv is None and isinstance(roi_results, dict):
            rv = roi_results.get(roi_id)

        if rv:
            metrics = rv.get("metrics") if isinstance(rv, dict) else getattr(rv, "metrics", None)

            def _fmt(k, v):
                try:
                    fv = float(v)
                    if k in ("score", "trk_score"):
                        return f"s:{fv:.2f}"
                    if k in ("white_ratio",):
                        return f"wr:{fv:.2f}"
                    if k in ("edge_energy", "lap_var", "laplacian_var"):
                        return f"e:{fv:.1f}"
                    if k in ("mean", "mean_gray", "mean_raw"):
                        return f"m:{fv:.1f}"
                    return f"{k}:{fv:.2f}"
                except Exception:
                    sv = str(v)
                    if k in ("qr_data", "barcode", "text"):
                        sv = sv[:12]
                        return f"qr:{sv}"
                    return f"{k}:{sv[:12]}"

            parts = []
            if isinstance(metrics, dict):
                order = ["mean", "score", "trk_score", "white_ratio", "edge_energy", "qr_data"]
                for k in order:
                    if k in metrics and metrics[k] is not None:
                        parts.append(_fmt(k, metrics[k]))
                    if len(parts) >= 3:
                        break

            if parts:
                line2 = " ".join(parts)

    return [line1, line2] + ([line3] if line3 else [])


def _draw_roi_label_block(
    img,
    *,
    x,
    y,
    w,
    h,
    angle,
    lines,
    compact,
    roi_text_color,
    roi_text_scale,
    roi_text_thickness,
    base_font,
    base_font_scale,
    base_thickness,
    line_spacing,
):
    h_img, w_img = img.shape[:2]

    if compact:
        lines2 = [s for s in lines if s]
        if not lines2:
            return

        sizes = [
            cv2.getTextSize(str(s), base_font, roi_text_scale, roi_text_thickness)[0]
            for s in lines2
        ]

        max_text_w = max(sz[0] for sz in sizes) if sizes else 0
        max_text_h = max(sz[1] for sz in sizes) if sizes else 10

        line_gap = max(max_text_h + 3, int(round(roi_text_scale * 38)))
        total_h = max_text_h + max(0, len(lines2) - 1) * line_gap

        margin = max(8, int(round(roi_text_scale * 30)))

        tx = int(x + 2)
        tx = max(2, min(tx, w_img - max_text_w - 2))

        # 기본: ROI 상단 바깥쪽으로 배치
        ty_above = int(y - margin - total_h)

        # 위쪽 공간이 부족하면 ROI 하단 바깥쪽으로 배치
        if ty_above >= 2:
            ty = ty_above
        else:
            ty = int(y + h + margin)
            ty = min(ty, h_img - total_h - 2)

        for i, t in enumerate(lines2):
            draw_text(
                img,
                t,
                (tx, ty + (i * line_gap)),
                color=roi_text_color,
                scale=roi_text_scale,
                thickness=1,
                align="lt",
            )
        return
    
    sizes = [cv2.getTextSize(s, base_font, base_font_scale, base_thickness)[0] for s in lines]
    heights = [cv2.getTextSize(s, base_font, base_font_scale, base_thickness)[0][1] for s in lines]
    max_w = max(sz[0] for sz in sizes) if sizes else 0
    total_h = sum(heights) + max(0, (len(lines) - 1)) * line_spacing

    margin = 10
    cx = x + w / 2
    cy = y + h / 2

    label_local = np.array([0.0, -(h / 2.0 + margin + total_h)], dtype=float)

    th = np.radians(angle)
    c = np.cos(th)
    s = np.sin(th)
    rot = np.array([[c, -s], [s, c]], dtype=float)

    label_pt = np.array([cx, cy], dtype=float) + rot @ label_local

    bg_x1 = int(label_pt[0] - max_w / 2 - 4)
    bg_y1 = int(label_pt[1] - 4)
    bg_x2 = bg_x1 + max_w + 8
    bg_y2 = bg_y1 + total_h + 8

    if bg_x1 < 2:
        bg_x2 += (2 - bg_x1)
        bg_x1 = 2
    if bg_x2 > w_img - 2:
        shift = bg_x2 - (w_img - 2)
        bg_x1 -= shift
        bg_x2 -= shift
    if bg_y1 < 2:
        bg_y2 += (2 - bg_y1)
        bg_y1 = 2
    if bg_y2 > h_img - 2:
        shift = bg_y2 - (h_img - 2)
        bg_y1 -= shift
        bg_y2 -= shift

    top_text_y = bg_y1 + 4 + heights[0]

    temp = img.copy()
    draw_rect(temp, (bg_x1, bg_y1), (bg_x2, bg_y2), color=(0, 0, 0), fill=True)
    draw_rect(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color=(50, 50, 50), thickness=1)
    alpha = 0.4
    cv2.addWeighted(temp, alpha, img, 1 - alpha, 0, img)

    cur_y = top_text_y + heights[0]
    for i, t in enumerate(lines):
        tx = bg_x1 + 4
        draw_text(
            img,
            t,
            (tx, cur_y),
            color=roi_text_color,
            scale=roi_text_scale,
            thickness=roi_text_thickness,
            align="lt",
        )
        if i + 1 < len(lines):
            cur_y += heights[i + 1] + line_spacing

def _normalize_roi_entry(r):
    if isinstance(r, dict):
        roi_id = r.get("id")
        label = r.get("label") or r.get("name") or f"ROI{roi_id}"
        rect = r.get("rect", r.get("bbox", (r.get("x", 0), r.get("y", 0), r.get("w", 0), r.get("h", 0))))
        x = int(rect[0])
        y = int(rect[1])
        w = int(rect[2])
        h = int(rect[3])
        angle = float(r.get("angle", 0.0))
    else:
        roi_id = getattr(r, "id", None)
        label = getattr(r, "name", getattr(r, "label", f"ROI{roi_id}"))
        x = int(getattr(r, "x", 0))
        y = int(getattr(r, "y", 0))
        w = int(getattr(r, "w", 0))
        h = int(getattr(r, "h", 0))
        angle = float(getattr(r, "angle", 0.0))

    return roi_id, label, x, y, w, h, angle


def _resolve_roi_result(roi_results, roi_id):
    rid_str = str(roi_id)
    rv = None
    ok = None
    metrics = None
    reason = ""

    if roi_results is not None and isinstance(roi_results, dict):
        rv = roi_results.get(rid_str) if rid_str in roi_results else roi_results.get(roi_id)

    if rv is not None:
        ok = rv.get("ok") if isinstance(rv, dict) else getattr(rv, "ok", None)
        metrics = rv.get("metrics") if isinstance(rv, dict) else getattr(rv, "metrics", None)
        reason = rv.get("reason", "") if isinstance(rv, dict) else getattr(rv, "reason", "")

    return rid_str, ok, metrics, reason


def _resolve_roi_draw_style(roi_id, active_id, ok, base_thickness):
    color = cfg.COLOR_ROI if hasattr(cfg, "COLOR_ROI") else (0, 200, 200)
    thickness = base_thickness

    if ok is True:
        color = cfg.COLOR_OK
    elif ok is False:
        color = cfg.COLOR_NG

    if roi_id == active_id:
        thickness = thickness + 1
        color = cfg.COLOR_ROI_ACTIVE if hasattr(cfg, "COLOR_ROI_ACTIVE") else color

    return color, thickness

def draw_rois(
    img,
    rois=None,
    active_id=None,
    roi_results=None,
    show_only_selected=False,
    compact=False,
    show_metrics=False,
    **kwargs,
):
    """
    rois: list of dicts {'id':int, 'label':str, 'rect':(x,y,w,h)} OR objects with x,y,w,h,name,id
    active_id: roi id to highlight
    roi_results: optional dict keyed by roi id or str(id)
    show_only_selected: bool
    """
    if rois is None:
        # if older code passed a roi_mgr, try to adapt
        rois = []

    base_font = cfg.FONT

    roi_text_scale, roi_text_thickness, roi_line_gap = _roi_text_style(
        img,
        base_scale=0.30,
    )

    base_font_scale = roi_text_scale
    # ===== START 2026-09-16 : 001~004 RUN 표시 가시성 공통 변경 =====
    # RUN mode ROI border: keep text thin, but make the ROI box clearly visible.
    base_thickness = 8
    # ===== END 2026-09-16 : 001~004 RUN 표시 가시성 공통 변경 =====
    line_spacing = max(4, int(round(roi_line_gap * 0.35)))

    roi_text_color = (0, 255, 0)

    for r in rois:
        # normalize roi dict/object to {id, x,y,w,h, name/label}
        roi_id, label, x, y, w, h, angle = _normalize_roi_entry(r)

        if show_only_selected and active_id is not None and roi_id != active_id:
            continue

        rid_str, ok, metrics, reason = _resolve_roi_result(roi_results, roi_id)
        color, thickness = _resolve_roi_draw_style(roi_id, active_id, ok, base_thickness)

        # rotated ROI box
        cx = x + w / 2
        cy = y + h / 2

        rect = ((cx, cy), (w, h), angle)

        box = cv2.boxPoints(rect)
        box = np.int32(box)

        cv2.polylines(img, [box], True, color, thickness, lineType=cv2.LINE_AA)

        # center point
        cv2.circle(img, (int(cx), int(cy)), 3, color, -1, lineType=cv2.LINE_AA)

        # circle draw
        _draw_circle_overlay(img, x, y, w, h, metrics)

        # line draw
        _draw_line_overlay(img, x, y, w, h, angle, metrics)

        # distance judge
        _draw_distance_overlay(img, x, y, h, metrics)

        # distance links
        _draw_roi_distance_links(img, x, y, h, metrics)

        # QR scan result text (ROI 하단, QR일 때만)
        _draw_qr_overlay(
            img,
            rois,
            x,
            y,
            h,
            ok,
            reason,
            color,
            metrics,
            roi_text_scale,
            roi_text_thickness,
        )

        # prepare label lines
        lines = _build_roi_label_lines(
            roi_id=roi_id,
            label=label,
            x=x,
            y=y,
            w=w,
            h=h,
            angle=angle,
            active_id=active_id,
            compact=compact,
            show_metrics=show_metrics,
            roi_results=roi_results,
            rid_str=rid_str,
        )

        _draw_roi_label_block(
            img,
            x=x,
            y=y,
            w=w,
            h=h,
            angle=angle,
            lines=lines,
            compact=compact,
            roi_text_color=roi_text_color,
            roi_text_scale=roi_text_scale,
            roi_text_thickness=roi_text_thickness,
            base_font=base_font,
            base_font_scale=base_font_scale,
            base_thickness=base_thickness,
            line_spacing=line_spacing,
        )

        _draw_roi_distance_list(img, roi_results)


def draw_overall_banner(img, overall_ok, info=None):
    h, w = img.shape[:2]

    # ----- 위치 정책 (여기만 수정하면 전체 위치 변경됨) -----
    POS = {
        # 3배 확대된 OVERALL 문구가 화면 위쪽에서 잘리지 않도록 하향 이동.
        "overall": ("ct", (0, 75)),      # center-top
        "debug":   ("rb", (12, 88)),     # right-bottom (버튼바 피해서)
    }
    # -------------------------------------------------------

    ng = int(info.get("ng", 0)) if isinstance(info, dict) else 0
    total = int(info.get("total", 0)) if isinstance(info, dict) else 0

    mode = info.get("mode") if isinstance(info, dict) else None
    allowed = info.get("max_fail") if isinstance(info, dict) else None

    if total > 0:
        if overall_ok:
            if mode == "allow_fail_count" and allowed is not None:
                text = f"OVERALL: OK (NG:{ng}/{total}, allowed:{allowed})"
            else:
                text = f"OVERALL: OK ({total-ng}/{total})"
        else:
            text = f"OVERALL: NG (NG:{ng}/{total})"
    else:
        text = "OVERALL: -"

    color = cfg.COLOR_OK if overall_ok else cfg.COLOR_NG

    # ---- OVERALL (중앙 상단) ----
    align, (mx, my) = POS["overall"]
    x = w // 2 + mx
    y = my
    # ===== START 2026-09-16 : 001~004 RUN 표시 가시성 공통 변경 =====
    draw_text(img, text, (x, y-10), color=color, scale=2.4, thickness=6, align=align)
    # ===== END 2026-09-16 : 001~004 RUN 표시 가시성 공통 변경 =====

    # ---- debug (우측 하단) ----
    if isinstance(info, dict):
        parts = []

        if "norm_gain" in info:
            try:
                parts.append(f"gain={float(info['norm_gain']):.2f}")
            except Exception:
                parts.append(f"gain={info['norm_gain']}")

        if "dx" in info:
            parts.append(f"dx={int(info['dx'])}")

        if "dy" in info:
            parts.append(f"dy={int(info['dy'])}")

        if parts:
            dbg = "  ".join(parts)

            align, (mx, my) = POS["debug"]
            x = w - mx
            y = h - my

            draw_text(img, dbg, (x-230, y+2), color=cfg.COLOR_TEXT, scale=0.6, thickness=1, align=align)


def draw_origin_axes(img, origin=(40, 60), axis_len=80):
    ox, oy = int(origin[0]), int(origin[1])

    # origin point
    cv2.circle(img, (ox, oy), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    # X axis
    cv2.arrowedLine(img, (ox, oy), (ox + axis_len, oy), (0, 255, 255), 1, line_type=cv2.LINE_AA, tipLength=0.12)
    draw_text(img, "X+", (ox + axis_len + 8, oy), color=(0, 255, 255), scale=0.5, thickness=1, align="lt")

    # Y axis
    cv2.arrowedLine(img, (ox, oy), (ox, oy + axis_len), (0, 255, 255), 1, line_type=cv2.LINE_AA, tipLength=0.12)
    draw_text(img, "Y+", (ox - 4, oy + axis_len + 14), color=(0, 255, 255), scale=0.5, thickness=1, align="lt")

    # origin label
    draw_text(img, f"(0,0)", (ox + 6, oy - 10), color=(0, 255, 255), scale=0.5, thickness=1, align="lt")

def draw_selected_roi_info(img, roi, parent_roi=None):
    if not roi:
        return

    x = int(roi.get("x", 0))
    y = int(roi.get("y", 0))
    w = int(roi.get("w", 0))
    h = int(roi.get("h", 0))
    rid = roi.get("id", "")

    angle = float(roi.get("angle", 0.0))
    cx = int(x + w / 2)
    cy = int(y + h / 2)

    text = f"Selected ROI: {rid}  x:{x} y:{y} w:{w} h:{h} a:{angle:.1f}  cx:{cx} cy:{cy}"

    if parent_roi and parent_roi.get("id") != rid:
        px = int(parent_roi.get("x", 0))
        py = int(parent_roi.get("y", 0))
        dx = x - px
        dy = y - py
        text += f"  rel:{dx},{dy}"
        
    draw_text(img, text, (140, 20), color=(0, 255, 255), scale=0.6, thickness=1, align="lt")

def draw_control_bar(img, buttons):
    h, w = img.shape[:2]
    bar_h = 72
    temp = img.copy()
    y0 = h - bar_h
    draw_rect(temp, (0, y0), (w, h), color=(30, 30, 30), fill=True)
    cv2.addWeighted(temp, 0.85, img, 0.15, 0, img)

    n = len(buttons)
    if n == 0:
        return buttons
    min_btn_w = 100
    btn_w = max(min_btn_w, int(w / n))
    btn_h = bar_h - 16
    x = 8
    new_buttons = []
    for i, b in enumerate(buttons):
        if b.get("rect") is None:
            x1 = x
            y1 = y0 + 8
            x2 = min(w - 8, x1 + btn_w - 8)
            y2 = y1 + btn_h
            x = x2 + 8
            b["rect"] = (int(x1), int(y1), int(x2), int(y2))
        else:
            x1,y1,x2,y2 = b["rect"]
            b["rect"] = (int(x1), int(y1), int(x2), int(y2))

        x1,y1,x2,y2 = b["rect"]
        bg_color = b.get("color", (60,60,60))
        draw_rect(img, (x1, y1), (x2, y2), color=bg_color, fill=True)
        draw_rect(img, (x1, y1), (x2, y2), color=(180,180,180), thickness=1)
        label = b.get("label", b.get("id",""))
        (tw, th), baseline = cv2.getTextSize(label, cfg.FONT, 0.7, 2)
        tx = x1 + max(8, (x2-x1 - tw)//2)
        ty = y1 + max(24, (y2-y1 + th)//2)
        draw_text(img, label, (tx, ty), color=cfg.COLOR_TEXT, scale=0.7, thickness=2)
        new_buttons.append(b)
    return new_buttons