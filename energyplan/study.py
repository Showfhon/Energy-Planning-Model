"""Scenario studies: parameter overrides, sensitivity sweeps and comparisons.

A planning exercise is rarely one optimisation.  It is a base case plus a dozen
variants -- "what if offshore wind costs 30% more", "what if demand grows
faster", "what if the carbon cap tightens".  These helpers make that loop cheap
by editing the raw scenario document before it is validated, so every variant
goes through exactly the same checks as the base case.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .data import load_scenario

__all__ = ["apply_override", "apply_overrides", "parse_override", "run_sensitivity", "compare"]

_LIST_KEYS = {
    "technologies": "name",
    "storage": "name",
    "regions": "name",
    "lines": "name",
}


def parse_override(text: str) -> Tuple[str, Any]:
    """Parse ``path=value``.  Values are read as JSON, falling back to text."""
    if "=" not in text:
        raise ValueError(f"override must look like path=value, got {text!r}")
    path, _, raw = text.partition("=")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return path.strip(), value


def apply_override(spec: dict, path: str, value: Any) -> dict:
    """Return a copy of ``spec`` with the dotted ``path`` set to ``value``.

    Lists of named entries are addressed by name, so
    ``technologies.solar.capex`` and ``policy.voll`` both work.  A trailing
    ``*`` multiplies the existing value instead of replacing it, which is what
    you usually want for cost sensitivities::

        apply_override(spec, "technologies.wind_offshore.capex*", 1.3)
    """
    spec = copy.deepcopy(spec)
    multiply = path.endswith("*")
    if multiply:
        path = path[:-1]
    parts = [p for p in path.split(".") if p]
    if not parts:
        raise ValueError("empty override path")

    node: Any = spec
    for depth, part in enumerate(parts[:-1]):
        if isinstance(node, dict) and part in _LIST_KEYS and isinstance(node.get(part), list):
            key_field = _LIST_KEYS[part]
            name = parts[depth + 1]
            match = next(
                (item for item in node[part] if str(item.get(key_field)) == name), None
            )
            if match is None:
                raise KeyError(f"{path}: no {part[:-1]} named {name!r}")
            node = match
            parts = parts[:depth + 1] + parts[depth + 2:]
            # ``parts`` shrank by one; continue from the same logical position.
            return _finish(spec, node, parts[depth + 1:], value, multiply, path)
        if not isinstance(node, dict):
            raise KeyError(f"{path}: cannot descend into {type(node).__name__}")
        if part not in node or node[part] is None:
            node[part] = {}
        node = node[part]
    return _finish(spec, node, parts[-1:], value, multiply, path)


def _finish(spec: dict, node: Any, remaining: Sequence[str], value: Any,
            multiply: bool, path: str) -> dict:
    for part in remaining[:-1]:
        if part not in node or node[part] is None:
            node[part] = {}
        node = node[part]
    leaf = remaining[-1]
    if multiply:
        current = node.get(leaf)
        if current is None:
            raise KeyError(f"{path}: nothing to multiply")
        if isinstance(current, dict):
            node[leaf] = {k: v * value for k, v in current.items()}
        else:
            node[leaf] = current * value
    else:
        node[leaf] = value
    return spec


def apply_overrides(spec: dict, overrides: Iterable[str]) -> dict:
    for item in overrides:
        path, value = parse_override(item)
        spec = apply_override(spec, path, value)
    return spec


def run_sensitivity(
    spec: dict,
    path: str,
    values: Sequence[Any],
    solver: str = "auto",
    progress: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """Solve the scenario once per value of ``path`` and summarise each run."""
    from .model import CapacityExpansionModel

    rows: List[Dict[str, Any]] = []
    for value in values:
        variant = apply_override(spec, path, value)
        scenario = load_scenario(variant)
        if progress:
            progress(f"solving {path}={value} ...")
        model = CapacityExpansionModel(scenario)
        result = model.solve(solver=solver)
        rows.append(summarise(result, label=f"{path}={value}", value=value))
    return rows


def summarise(result, label: str = "", value: Any = None) -> Dict[str, Any]:
    """One row of headline numbers for a solved plan."""
    if not result.optimal:
        return {"label": label, "value": value, "status": result.status}
    final = result.years[-1]
    row = {
        "label": label,
        "value": value,
        "status": result.status,
        "npv_bn_usd": result.objective / 1e9,
        "lcoe_usd_mwh": result.lcoe(),
        "cumulative_mtco2": result.total_emissions_t() / 1e6,
        "final_renewable_share": final.renewable_share,
        "final_price_usd_mwh": final.marginal_price,
        "unserved_mwh": sum(
            s.unserved_mwh * result.scenario.period_weight(s.year) for s in result.years
        ),
    }
    for name, mw in sorted(final.capacity_mw.items()):
        row[f"cap_{name}_mw"] = mw
    for name, mw in sorted(final.storage_power_mw.items()):
        row[f"cap_{name}_mw"] = mw
    return row


def compare(rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> str:
    """Render sensitivity or comparison rows as a fixed-width table."""
    if not rows:
        return "(no runs)"
    if columns is None:
        columns = ["label", "npv_bn_usd", "lcoe_usd_mwh", "cumulative_mtco2",
                   "final_renewable_share", "final_price_usd_mwh", "unserved_mwh"]
    columns = [c for c in columns if any(c in row for row in rows)]

    def cell(row: Dict[str, Any], column: str) -> str:
        value = row.get(column)
        if value is None:
            return "-"
        if isinstance(value, float):
            if column.endswith("share"):
                return f"{value:.1%}"
            return f"{value:,.2f}" if abs(value) < 1000 else f"{value:,.0f}"
        return str(value)

    header = [c.replace("_", " ") for c in columns]
    table = [[cell(row, c) for c in columns] for row in rows]
    widths = [max(len(header[i]), max(len(r[i]) for r in table)) for i in range(len(columns))]
    lines = ["  ".join(header[i].ljust(widths[i]) for i in range(len(columns)))]
    lines.append("  ".join("-" * w for w in widths))
    for row in table:
        lines.append("  ".join(
            row[i].ljust(widths[i]) if i == 0 else row[i].rjust(widths[i])
            for i in range(len(columns))
        ))
    return "\n".join(lines)
