def _get_result(results, roi_id):
    return results.get(str(roi_id)) or results.get(roi_id)


def _apply_interlocks(recipe, results):
    decision = recipe.get("decision") or {}
    rules = decision.get("interlocks") or []

    if isinstance(rules, dict):
        rules = [rules]

    for rule in rules:
        source_roi_id = rule.get("source_roi_id")
        source = _get_result(results, source_roi_id)

        # 해당 ROI가 없는 다른 레시피에는 영향 없음
        if source is None or bool(source.ok):
            continue

        reason = str(rule.get("reason", "INTERLOCK_NG"))

        for roi_id in rule.get("force_ng_roi_ids", []):
            target = _get_result(results, roi_id)
            if target is None:
                continue

            if not isinstance(target.metrics, dict):
                target.metrics = {}

            # 원래 검사 결과 보존
            target.metrics.setdefault(
                "interlock_original_ok",
                bool(target.ok),
            )
            target.metrics.setdefault(
                "interlock_original_reason",
                str(target.reason),
            )

            target.metrics["interlock_forced_ng"] = True
            target.metrics["interlock_source_roi"] = str(source_roi_id)

            target.ok = False
            target.reason = reason


def decide_overall(*, recipe, results, auto_mode=False):
    # 모든 ROI 측정이 끝난 후 인터록 적용
    _apply_interlocks(recipe, results)

    decision = recipe.get("decision") or {}
    mode = (decision.get("mode") or "any_fail_is_ng").strip().lower()

    oks = [r.ok for r in results.values()]

    if not oks:
        overall_ok = False
    elif mode == "any_fail_is_ng":
        overall_ok = all(oks)
    elif mode == "majority_ok":
        overall_ok = sum(1 for value in oks if value) >= (len(oks) / 2)
    elif mode == "allow_fail_count":
        max_fail = int(decision.get("max_fail", 0))
        fail_count = sum(1 for value in oks if not value)
        overall_ok = fail_count <= max_fail
    else:
        overall_ok = all(oks)

    if not auto_mode and recipe.get("debug", False):
        print(f"[DBG] overall decision by recipe : {mode}")

    return overall_ok