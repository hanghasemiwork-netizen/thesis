"""
rule_checker.py
Checks evacuation distances against SCIE rules from rules/scie_rules.json.

Simplified model — no local risk category:
  - Dead-end space (one exit route)  → "dead_end_max_m"
  - Two-exit space                   → "two_exits_max_m"

The limits live in scie_rules.json, not hardcoded in this file.
"""

import json

def load_rules(rules_path: str) -> dict:
    with open(rules_path, "r") as f:
        return json.load(f)
    
def get_applicable_rule(rules: dict, building_ut: str) -> dict:
     if building_ut in rules:
        return rules[building_ut]
     return rules.get("UT_I")


def check_space(path_result: dict, rules: dict,building_ut: str) -> list:
    """
    Check a single space result against SCIE rules.
    Returns a list of violation dicts (empty if compliant).
    """
    violations = []

    if not path_result["reachable"]:
        violations.append({
            "type": "UNREACHABLE",
            "space_id": path_result["space_id"],
            "space_name": path_result["space_name"],
            "rule_id": "SCIE-UNREACHABLE",
            "article": "General",
            "description": "Space has no path to any fire exit door",
            "measured_m": None,
            "limit_m": None,
            "excess_m": None,
            "two_exits": False
        })
        return violations

    two_exits = path_result.get("has_two_exits", False)
    rule = get_applicable_rule(rules, building_ut)
    if rule:
        limit = rule["two_exits_max_m"] if two_exits else rule["dead_end_max_m"]

        # Total distance = inside space + route through corridors to exit
        total_distance = round(
            path_result["art57_distance_m"] + path_result["art61_distance_m"], 3
        )

        if total_distance > limit:
            violations.append({
                "type":        "ART57_VIOLATION",
                "space_id":    path_result["space_id"],
                "space_name":  path_result["space_name"],
                "rule_id":     rule["id"],
                "article":     rule["article"],
                "description": rule["description"],
                "measured_m":  total_distance,
                "limit_m":     limit,
                "excess_m":    round(total_distance - limit, 3),
                "two_exits":   two_exits,
                "path_type":   "two exits" if two_exits else "dead-end",
                "detail":      (f"inside space: {path_result['art57_distance_m']}m + "
                                f"route to exit: {path_result['art61_distance_m']}m = "
                                f"total: {total_distance}m")
            })

    return violations


def check_all(path_results: list, rules_path: str, building_ut) -> dict:
    """
    Run SCIE rule checks on all path results.

    Returns:
        {
            "total_spaces": int,
            "compliant": int,
            "violations": int,
            "violation_list": [...],
            "summary_by_rule": {...}
        }
    """
    rules = load_rules(rules_path)

    all_violations = []
    compliant_count = 0

    for result in path_results:
        violations = check_space(result, rules, building_ut)
        if violations:
            all_violations.extend(violations)
        else:
            compliant_count += 1

    summary = {}
    for v in all_violations:
        rid = v["rule_id"]
        if rid not in summary:
            summary[rid] = {"count": 0, "article": v["article"]}
        summary[rid]["count"] += 1

    print(f"\n[Rules] Checked {len(path_results)} spaces")
    print(f"[Rules] Compliant: {compliant_count}")
    print(f"[Rules] Violations: {len(all_violations)}")

    for v in all_violations:
        print(f"  ❌ {v['space_name']} — "
              f"{v['measured_m']}m > {v['limit_m']}m "
              f"(+{v['excess_m']}m)")

    return {
        "total_spaces": len(path_results),
        "compliant": compliant_count,
        "violations": len(all_violations),
        "violation_list": all_violations,
        "summary_by_rule": summary
    }
