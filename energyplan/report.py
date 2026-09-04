"""Reporting: terminal tables, CSV export and a self-contained HTML report.

The HTML report embeds its own SVG charts, so it opens in any browser with no
network access and no JavaScript library.  Every chart is paired with the table
it was drawn from, which is both an accessibility requirement and the fastest
way for a planner to check a number.
"""

from __future__ import annotations

import csv
import html
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["text_report", "write_csv", "html_report", "write_json"]

# Categorical palette, validated for colour-vision deficiency on both the light
# and dark chart surfaces.  Hues are assigned to technologies in a fixed order
# so that adding or removing a technology never repaints the others.
PALETTE_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PALETTE_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                "#d55181", "#008300", "#9085e9", "#e66767"]


def _fmt(value: float, digits: int = 0) -> str:
    return f"{value:,.{digits}f}"


def _ge_prefix(summary) -> str:
    """A leading ≥ marks a shadow price the solver could not pin down."""
    return "≥ " if summary.carbon_price_is_degenerate else ""


def _series_names(result) -> List[str]:
    """A stable ordering of every technology and storage series in the plan."""
    names = [t.name for t in result.scenario.technologies]
    names += [s.name for s in result.scenario.storage]
    return names


def _colour_index(result) -> Dict[str, int]:
    """Map each series to a slot in the palette, in a fixed order.

    The palette has eight validated hues.  A ninth series does not get a new
    hue; it reuses hue 1 with a diagonal hatch, so identity is still carried by
    a second visual channel rather than by a colour a reader cannot separate.
    """
    return {name: i for i, name in enumerate(_series_names(result))}


def _style(slot: int) -> Tuple[int, bool]:
    """Return ``(palette slot, hatched)`` for a series.

    Colours are referenced as CSS custom properties rather than literal hex, so
    the dark-mode steps -- chosen for the dark surface rather than flipped from
    the light ones -- swap in automatically.
    """
    return slot % len(PALETTE_LIGHT), slot >= len(PALETTE_LIGHT)


def _colour(style: Tuple[int, bool]) -> str:
    """The CSS variable carrying this slot's hue; slot -1 is the neutral ink."""
    return "var(--text-secondary)" if style[0] < 0 else f"var(--series-{style[0] + 1})"


def _pattern_defs(styles: Sequence[Tuple[int, bool]]) -> str:
    """SVG hatch patterns for whichever styles need them."""
    slots = sorted({slot for slot, hatched in styles if hatched})
    if not slots:
        return ""
    parts = ["<defs>"]
    for slot in slots:
        colour = _colour((slot, False))
        parts.append(
            f'<pattern id="hatch{slot}" width="6" height="6" '
            f'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<rect width="6" height="6" fill="{colour}" fill-opacity="0.3"/>'
            f'<rect width="3" height="6" fill="{colour}"/>'
            f"</pattern>"
        )
    parts.append("</defs>")
    return "".join(parts)


def _fill(style: Tuple[int, bool]) -> str:
    slot, hatched = style
    return f"url(#hatch{slot})" if hatched else _colour(style)


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    columns = len(headers)
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i in range(columns):
            widths[i] = max(widths[i], len(str(row[i])))
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)), line]
    for row in rows:
        out.append("  ".join(str(row[i]).rjust(widths[i]) if i else str(row[i]).ljust(widths[i])
                             for i in range(columns)))
    return "\n".join(out)


def text_report(result, width: int = 88) -> str:
    """A readable summary of the plan for the terminal."""
    if not result.optimal:
        return (
            f"Plan status: {result.status}\n"
            f"{result.solution.message or 'no feasible plan was found'}\n"
        )

    scenario = result.scenario
    lines: List[str] = []
    rule = "=" * width
    lines.append(rule)
    lines.append(f"OPTIMAL ENERGY INVESTMENT PLAN - {scenario.name}")
    if scenario.description:
        lines.append(scenario.description)
    lines.append(rule)

    years = [s.year for s in result.years]
    lines.append("")
    lines.append(f"Horizon              {years[0]}-{years[-1]}  ({len(years)} milestone years)")
    lines.append(f"Discount rate        {scenario.discount_rate:.1%}")
    lines.append(f"Solver               {result.solution.solver}")
    lines.append(f"NPV of system cost   {_fmt(result.objective / 1e9, 2)} bn USD")
    lines.append(f"System LCOE          {_fmt(result.lcoe(), 2)} USD/MWh")
    lines.append(f"Cumulative CO2       {_fmt(result.total_emissions_t() / 1e6, 1)} MtCO2")

    unserved = sum(s.unserved_mwh * scenario.period_weight(s.year) for s in result.years)
    lines.append(f"Unserved energy      {_fmt(unserved, 0)} MWh over the horizon")

    # ---- installed capacity -------------------------------------------------
    lines.append("")
    lines.append("INSTALLED CAPACITY (MW)")
    names = sorted({n for s in result.years for n in s.capacity_mw})
    storage_names = sorted({n for s in result.years for n in s.storage_power_mw})
    rows = []
    for name in names:
        rows.append([name] + [_fmt(s.capacity_mw.get(name, 0.0)) for s in result.years])
    for name in storage_names:
        rows.append([f"{name} (storage)"]
                    + [_fmt(s.storage_power_mw.get(name, 0.0)) for s in result.years])
    rows.append(["TOTAL"] + [
        _fmt(sum(s.capacity_mw.values()) + sum(s.storage_power_mw.values()))
        for s in result.years
    ])
    lines.append(_table(["technology"] + [str(y) for y in years], rows))

    # ---- new build ----------------------------------------------------------
    schedule = result.build_schedule()
    if schedule:
        lines.append("")
        lines.append("NEW CAPACITY COMMISSIONED (MW)")
        rows = [
            [name] + [_fmt(schedule[name].get(y, 0.0)) for y in years]
            for name in sorted(schedule)
        ]
        lines.append(_table(["technology"] + [str(y) for y in years], rows))

    # ---- generation ---------------------------------------------------------
    lines.append("")
    lines.append("GENERATION (TWh per year)")
    gen_names = sorted({n for s in result.years for n in s.generation_mwh})
    rows = [
        [name] + [_fmt(s.generation_mwh.get(name, 0.0) / 1e6, 1) for s in result.years]
        for name in gen_names
    ]
    rows.append(["demand"] + [_fmt(s.demand_mwh / 1e6, 1) for s in result.years])
    lines.append(_table(["technology"] + [str(y) for y in years], rows))

    # ---- annual indicators ---------------------------------------------------
    lines.append("")
    lines.append("ANNUAL INDICATORS")
    rows = []
    for s in result.years:
        rows.append([
            str(s.year),
            f"{s.renewable_share:.1%}",
            _fmt(s.emissions_t / 1e6, 1),
            _fmt(s.marginal_price, 1),
            _fmt(s.average_cost, 1),
            _ge_prefix(s) + _fmt(s.carbon_shadow_price, 1),
            _fmt(s.unserved_mwh, 0),
        ])
    lines.append(_table(
        ["year", "renewable", "MtCO2", "price $/MWh", "cost $/MWh",
         "CO2 shadow $/t", "unserved MWh"],
        rows,
    ))

    # ---- cost breakdown -------------------------------------------------------
    lines.append("")
    lines.append("ANNUAL COST BY COMPONENT (million USD, undiscounted)")
    components = ["capital", "fixed_om", "variable_om", "fuel", "carbon",
                  "storage", "transmission", "unserved", "penalty"]
    rows = []
    for component in components:
        values = [s.cost.get(component, 0.0) for s in result.years]
        if max(abs(v) for v in values) < 1e3:
            continue
        rows.append([component] + [_fmt(v / 1e6, 0) for v in values])
    rows.append(["TOTAL"] + [_fmt(s.total_cost / 1e6, 0) for s in result.years])
    lines.append(_table(["component"] + [str(y) for y in years], rows))

    # ---- unmet targets --------------------------------------------------------
    warnings: List[str] = []
    for summary in result.years:
        for region, mw in summary.capacity_shortfall_mw.items():
            warnings.append(
                f"{summary.year}: firm capacity in {region} falls "
                f"{mw:,.0f} MW short of the reserve requirement"
            )
        if summary.emissions_overshoot_t > 1.0:
            warnings.append(
                f"{summary.year}: emissions exceed the cap by "
                f"{summary.emissions_overshoot_t / 1e6:,.2f} MtCO2"
            )
        for label, mwh in summary.share_deficit_mwh.items():
            warnings.append(
                f"{summary.year}: the {label.upper()} target is missed by "
                f"{mwh / 1e6:,.2f} TWh"
            )
        if summary.unserved_mwh > 1.0:
            warnings.append(
                f"{summary.year}: {summary.unserved_mwh:,.0f} MWh of demand is unserved"
            )
    if warnings:
        lines.append("")
        lines.append("UNMET TARGETS")
        for message in warnings:
            lines.append(f"  ! {message}")
        lines.append("  These are priced at the scenario's backstop values, not ignored.")

    # ---- consistency ----------------------------------------------------------
    checks = result.audit()
    lines.append("")
    lines.append("CONSISTENCY CHECKS (relative residuals, should be ~0)")
    lines.append(_table(
        ["check", "residual"],
        [[name, f"{value:.2e}"] for name, value in checks.items()],
    ))
    if any(s.carbon_price_is_degenerate for s in result.years):
        lines.append("")
        lines.append(
            "Note: a CO2 shadow price marked '>=' sits on a degenerate vertex "
            "(a cap of exactly zero).\n"
            "      Call PlanResult.empirical_marginal_carbon_cost(year) for the "
            "reliable figure."
        )
    worst = max(checks.values())
    lines.append("")
    lines.append("All checks pass." if worst < 1e-6
                 else f"WARNING: worst residual {worst:.2e} exceeds tolerance.")
    lines.append(rule)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV / JSON export
# ---------------------------------------------------------------------------


def write_csv(result, directory: str) -> List[str]:
    """Write the plan to a directory of CSV files.  Returns the paths written."""
    os.makedirs(directory, exist_ok=True)
    written: List[str] = []

    def dump(name: str, headers: Sequence[str], rows: Sequence[Sequence]) -> None:
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)
        written.append(path)

    rows = []
    for s in result.years:
        for name, mw in sorted(s.capacity_mw.items()):
            rows.append([s.year, name, "generation", round(mw, 3)])
        for name, mw in sorted(s.storage_power_mw.items()):
            rows.append([s.year, name, "storage_power", round(mw, 3)])
            rows.append([s.year, name, "storage_energy",
                         round(s.storage_energy_mwh.get(name, 0.0), 3)])
    dump("capacity.csv", ["year", "technology", "kind", "value_mw"], rows)

    rows = []
    for s in result.years:
        for name, mwh in sorted(s.generation_mwh.items()):
            rows.append([s.year, name, round(mwh, 3),
                         round(s.curtailment_mwh.get(name, 0.0), 3)])
    dump("generation.csv", ["year", "technology", "generation_mwh", "curtailment_mwh"], rows)

    rows = []
    for s in result.years:
        for component, value in sorted(s.cost.items()):
            rows.append([s.year, component, round(value, 2)])
    dump("costs.csv", ["year", "component", "usd_per_year"], rows)

    rows = []
    for s in result.years:
        rows.append([
            s.year, round(s.demand_mwh, 1), round(s.unserved_mwh, 3),
            round(s.emissions_t, 1), round(s.renewable_share, 5),
            round(s.marginal_price, 3), round(s.carbon_shadow_price, 3),
            round(s.average_cost, 3),
        ])
    dump("indicators.csv",
         ["year", "demand_mwh", "unserved_mwh", "emissions_tco2", "renewable_share",
          "marginal_price_usd_mwh", "carbon_shadow_price_usd_t", "average_cost_usd_mwh"],
         rows)

    index = result.index
    rows = []
    for s in result.years:
        for region in index.regions:
            dispatch = result.dispatch(s.year, region)
            prices = result.hourly_prices(s.year, region)
            for day in range(index.n_days):
                for hour in range(index.n_hours):
                    for name, grid in dispatch.items():
                        rows.append([
                            s.year, region, day, hour,
                            round(index.rep.weights[day], 3),
                            name, round(grid[day][hour], 4),
                            round(prices[day][hour], 4),
                        ])
    dump("dispatch.csv",
         ["year", "region", "rep_day", "hour", "day_weight", "series", "mw", "price_usd_mwh"],
         rows)
    return written


def write_json(result, path: str) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2)
    return path


# ---------------------------------------------------------------------------
# SVG charts
# ---------------------------------------------------------------------------


def _stacked_bar_svg(
    categories: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float]]],
    styles: Sequence[Tuple[int, bool]],
    unit: str,
    width: int = 560,
    height: int = 300,
) -> str:
    """A stacked bar chart with a 2px surface gap between segments."""
    left, right, top, bottom = 62, 16, 16, 34
    plot_w = width - left - right
    plot_h = height - top - bottom
    totals = [sum(values[i] for _, values in series) for i in range(len(categories))]
    peak = max(totals) if totals else 0.0
    if peak <= 0:
        peak = 1.0
    step = _nice_step(peak / 4.0)
    axis_max = step * max(1, int(peak / step) + 1)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">',
             _pattern_defs(styles)]
    for t in range(int(axis_max / step) + 1):
        value = t * step
        y = top + plot_h - plot_h * value / axis_max
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">'
            f'{_axis_label(value)}</text>'
        )
    slot = plot_w / max(1, len(categories))
    bar_w = min(46.0, slot * 0.6)
    for i, category in enumerate(categories):
        x = left + slot * (i + 0.5) - bar_w / 2
        cursor = 0.0
        for j, (name, values) in enumerate(series):
            value = values[i]
            if value <= 0:
                continue
            height_px = plot_h * value / axis_max
            y = top + plot_h - plot_h * (cursor + value) / axis_max
            gap = 2.0 if cursor > 0 else 0.0
            drawn = max(0.0, height_px - gap)
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{drawn:.1f}" '
                f'rx="2" fill="{_fill(styles[j])}" class="seg">'
                f'<title>{html.escape(name)} - {html.escape(str(category))}: '
                f'{_axis_label(value)} {html.escape(unit)}</title></rect>'
            )
            cursor += value
        parts.append(
            f'<text x="{left + slot * (i + 0.5):.1f}" y="{height - 12}" '
            f'class="tick" text-anchor="middle">{html.escape(str(category))}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _stacked_area_svg(
    hours: Sequence[int],
    series: Sequence[Tuple[str, Sequence[float]]],
    styles: Sequence[Tuple[int, bool]],
    demand: Sequence[float],
    width: int = 560,
    height: int = 320,
) -> str:
    """Dispatch over one representative day.

    Positive contributions stack upward from zero; storage charging is a
    negative contribution and stacks downward, so the height of the positive
    stack always equals demand plus charging rather than silently exceeding it.
    """
    left, right, top, bottom = 62, 16, 16, 34
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = len(hours)

    upper = max(
        (sum(max(0.0, values[i]) for _, values in series) for i in range(n)), default=0.0
    )
    upper = max(upper, max(demand) if demand else 0.0, 1.0)
    lower = min(
        (sum(min(0.0, values[i]) for _, values in series) for i in range(n)), default=0.0
    )
    step = _nice_step(max(upper, -lower) / 4.0)
    axis_max = step * max(1, int(upper / step) + 1)
    axis_min = -step * (int(-lower / step) + 1) if lower < 0 else 0.0
    span = axis_max - axis_min

    def px(i: int) -> float:
        return left + plot_w * (i / max(1, n - 1))

    def py(value: float) -> float:
        return top + plot_h - plot_h * (value - axis_min) / span

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">',
             _pattern_defs(styles)]
    ticks = int(round(span / step)) + 1
    for t in range(ticks):
        value = axis_min + t * step
        y = py(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'class="{"axis" if value == 0 else "grid"}"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">'
            f'{_axis_label(value)}</text>'
        )

    up = [0.0] * n
    down = [0.0] * n
    for j, (name, values) in enumerate(series):
        top_edge, bottom_edge = [], []
        for i in range(n):
            value = values[i]
            if value >= 0:
                bottom_edge.append(up[i])
                up[i] += value
                top_edge.append(up[i])
            else:
                bottom_edge.append(down[i])
                down[i] += value
                top_edge.append(down[i])
        points = " ".join(f"{px(i):.1f},{py(top_edge[i]):.1f}" for i in range(n))
        points += " " + " ".join(
            f"{px(i):.1f},{py(bottom_edge[i]):.1f}" for i in range(n - 1, -1, -1)
        )
        parts.append(
            f'<polygon points="{points}" fill="{_fill(styles[j])}" fill-opacity="0.92" '
            f'stroke="var(--surface-1)" stroke-width="1">'
            f'<title>{html.escape(name)}</title></polygon>'
        )

    demand_points = " ".join(f"{px(i):.1f},{py(demand[i]):.1f}" for i in range(n))
    parts.append(
        f'<polyline points="{demand_points}" fill="none" stroke="var(--text-primary)" '
        f'stroke-width="2" stroke-dasharray="5 3"><title>demand</title></polyline>'
    )
    for i in range(0, n, max(1, n // 8)):
        parts.append(
            f'<text x="{px(i):.1f}" y="{height - 12}" class="tick" text-anchor="middle">'
            f'{hours[i]:02d}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _nice_step(raw: float) -> float:
    import math

    if raw <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 2.5, 5, 10):
        if raw <= factor * magnitude:
            return factor * magnitude
    return 10 * magnitude


def _axis_label(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _legend(names: Sequence[str], styles: Sequence[Tuple[int, bool]]) -> str:
    items = []
    for i, name in enumerate(names):
        colour = _colour(styles[i])
        if styles[i][1]:
            swatch = (
                f"background:repeating-linear-gradient(45deg,{colour} 0 3px,"
                f"color-mix(in srgb,{colour} 30%,transparent) 3px 6px)"
            )
        else:
            swatch = f"background:{colour}"
        items.append(
            f'<span class="key"><i style="{swatch}"></i>{html.escape(name)}</span>'
        )
    return f'<div class="legend">{"".join(items)}</div>'


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def html_report(result, path: str, title: Optional[str] = None) -> str:
    """Write a self-contained HTML report and return its path."""
    scenario = result.scenario
    title = title or f"Energy investment plan - {scenario.name}"
    if not result.optimal:
        body = f"<h1>{html.escape(title)}</h1><p>No optimal plan: {html.escape(result.status)}.</p>"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(_render_shell(title, body))
        return path

    years = [s.year for s in result.years]
    labels = [str(y) for y in years]
    colour_of = _colour_index(result)

    def series_for(getter, names) -> List[Tuple[str, List[float]]]:
        return [(name, [getter(s).get(name, 0.0) for s in result.years]) for name in names]

    cap_names = sorted({n for s in result.years for n in s.capacity_mw})
    cap_names += sorted({n for s in result.years for n in s.storage_power_mw})
    cap_series = [
        (name, [
            s.capacity_mw.get(name, s.storage_power_mw.get(name, 0.0)) / 1000.0
            for s in result.years
        ])
        for name in cap_names
    ]
    cap_styles = [_style(colour_of.get(n, 0)) for n in cap_names]

    gen_names = sorted({n for s in result.years for n in s.generation_mwh})
    gen_series = [
        (name, [s.generation_mwh.get(name, 0.0) / 1e6 for s in result.years])
        for name in gen_names
    ]
    gen_styles = [_style(colour_of.get(n, 0)) for n in gen_names]

    components = ["capital", "fixed_om", "variable_om", "fuel", "carbon",
                  "storage", "transmission", "unserved", "penalty"]
    cost_series = [
        (c, [s.cost.get(c, 0.0) / 1e9 for s in result.years]) for c in components
        if any(abs(s.cost.get(c, 0.0)) > 1e3 for s in result.years)
    ]
    cost_styles = [_style(i) for i in range(len(cost_series))]

    # Dispatch on the peak representative day of the final year.
    final = result.years[-1].year
    index = result.index
    peak_day = index.rep.peak_day_position
    dispatch = result.dispatch(final)
    demand_curve = dispatch.pop("demand")[peak_day]
    hours = [int(round(h * index.rep.hours_per_step)) for h in range(index.n_hours)]
    disp_series = [(name, grid[peak_day]) for name, grid in dispatch.items()]
    disp_styles = [
        _style(colour_of.get(name.replace(" (net)", ""), i))
        for i, (name, _) in enumerate(disp_series)
    ]

    unserved_total = sum(s.unserved_mwh * scenario.period_weight(s.year) for s in result.years)
    tiles = [
        ("NPV of system cost", f"{result.objective / 1e9:,.1f}", "bn USD"),
        ("System LCOE", f"{result.lcoe():,.1f}", "USD/MWh"),
        ("Cumulative CO2", f"{result.total_emissions_t() / 1e6:,.0f}", "MtCO2"),
        ("Renewable share, final year", f"{result.years[-1].renewable_share:.0%}", str(final)),
        ("Unserved energy", f"{unserved_total:,.0f}", "MWh"),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="tile-label">{html.escape(label)}</div>'
        f'<div class="tile-value">{html.escape(value)}</div>'
        f'<div class="tile-unit">{html.escape(unit)}</div></div>'
        for label, value, unit in tiles
    )

    checks = result.audit()
    check_rows = [[name, f"{value:.2e}",
                   "pass" if value < 1e-6 else "CHECK"] for name, value in checks.items()]

    body = f"""
<h1>{html.escape(title)}</h1>
<p class="lede">{html.escape(scenario.description or 'Least-cost capacity expansion plan.')}
Horizon {years[0]}&ndash;{years[-1]}, discount rate {scenario.discount_rate:.1%},
solved with {html.escape(result.solution.solver)}.</p>
<div class="tiles">{tile_html}</div>

<section>
  <h2>Installed capacity</h2>
  {_legend([n for n, _ in cap_series], cap_styles)}
  {_stacked_bar_svg(labels, cap_series, cap_styles, "GW")}
  <details><summary>Data table &mdash; capacity, GW</summary>
  {_html_table(["technology"] + labels,
               [[n] + [f"{v:,.2f}" for v in vals] for n, vals in cap_series])}
  </details>
</section>

<section>
  <h2>Generation mix</h2>
  {_legend([n for n, _ in gen_series], gen_styles)}
  {_stacked_bar_svg(labels, gen_series, gen_styles, "TWh")}
  <details><summary>Data table &mdash; generation, TWh</summary>
  {_html_table(["technology"] + labels,
               [[n] + [f"{v:,.1f}" for v in vals] for n, vals in gen_series])}
  </details>
</section>

<section>
  <h2>Annual system cost by component</h2>
  {_legend([n for n, _ in cost_series], cost_styles)}
  {_stacked_bar_svg(labels, cost_series, cost_styles, "bn USD")}
  <details><summary>Data table &mdash; cost, bn USD per year</summary>
  {_html_table(["component"] + labels,
               [[n] + [f"{v:,.2f}" for v in vals] for n, vals in cost_series])}
  </details>
</section>

<section>
  <h2>Dispatch on the peak day, {final}</h2>
  <p class="note">Stacked generation across the representative day with the highest
  net load. Storage charging stacks below the zero line, so the height of the
  positive stack is demand plus charging. The dashed line is demand.</p>
  {_legend([n for n, _ in disp_series] + ["demand"], list(disp_styles) + [(-1, False)])}
  {_stacked_area_svg(hours, disp_series, disp_styles, demand_curve)}
</section>

<section>
  <h2>Annual indicators</h2>
  {_html_table(
      ["year", "demand TWh", "renewable", "MtCO2", "price USD/MWh",
       "cost USD/MWh", "CO2 shadow USD/t", "unserved MWh"],
      [[s.year, f"{s.demand_mwh / 1e6:,.1f}", f"{s.renewable_share:.1%}",
        f"{s.emissions_t / 1e6:,.1f}", f"{s.marginal_price:,.1f}",
        f"{s.average_cost:,.1f}",
        _ge_prefix(s) + f"{s.carbon_shadow_price:,.1f}",
        f"{s.unserved_mwh:,.0f}"] for s in result.years])}
</section>

<section>
  <h2>Consistency checks</h2>
  <p class="note">Relative residuals recomputed from the reported results,
  independently of the solver.</p>
  {_html_table(["check", "residual", "verdict"], check_rows)}
</section>
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_render_shell(title, body))
    return path


def _render_shell(title: str, body: str) -> str:
    """Wrap the report body in the page shell.

    Placeholder substitution rather than ``str.format`` -- the shell is mostly
    CSS, and doubling every brace to escape it is a reliable source of bugs.
    """
    return (
        _HTML_SHELL
        .replace("__TITLE__", html.escape(title))
        .replace("__BODY__", body)
    )


_HTML_SHELL = """<!doctype html><!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --surface-0: #f4f4f2; --surface-1: #fcfcfb; --border: #dcdcd6;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #78776f;
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  --series-4: #eda100;
  --series-5: #e87ba4;
  --series-6: #008300;
  --series-7: #4a3aa7;
  --series-8: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-0: #111110; --surface-1: #1a1a19; --border: #33332f;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #94938a;
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
    --series-4: #c98500;
    --series-5: #d55181;
    --series-6: #008300;
    --series-7: #9085e9;
    --series-8: #e66767;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 20px 64px;
  background: var(--surface-0); color: var(--text-primary);
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
}
main { max-width: 940px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 8px; letter-spacing: -0.01em; }
h2 { font-size: 17px; margin: 0 0 12px; }
.lede { color: var(--text-secondary); margin: 0 0 24px; max-width: 68ch; }
.note { color: var(--text-secondary); font-size: 13px; margin: -4px 0 12px; max-width: 68ch; }
.tiles { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 28px; }
.tile {
  flex: 1 1 150px; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 14px;
}
.tile-label { font-size: 12px; color: var(--text-secondary); }
.tile-value { font-size: 24px; font-variant-numeric: tabular-nums; margin-top: 2px; }
.tile-unit { font-size: 12px; color: var(--text-muted); }
section {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 18px 20px; margin-bottom: 20px;
}
.chart { width: 100%; height: auto; display: block; }
.grid { stroke: var(--border); stroke-width: 1; }
.axis { stroke: var(--text-muted); stroke-width: 1; }
.tick { fill: var(--text-muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.seg { transition: opacity .12s; }
.seg:hover { opacity: .78; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-bottom: 10px; }
.key { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-secondary); }
.key i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
details { margin-top: 12px; }
summary { cursor: pointer; font-size: 13px; color: var(--text-secondary); }
table { border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 13px; }
th, td {
  text-align: right; padding: 5px 8px; border-bottom: 1px solid var(--border);
  font-variant-numeric: tabular-nums; white-space: nowrap;
}
th:first-child, td:first-child { text-align: left; }
thead th { color: var(--text-secondary); font-weight: 600; }
.scroll { overflow-x: auto; }
</style>
</head>
<body><main>
__BODY__
</main></body>
</html>
"""
