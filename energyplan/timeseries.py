"""Hourly profiles and representative-day reduction.

Solving a capacity-expansion model on all 8760 hours of every planning year is
usually unnecessary and often intractable.  The standard remedy is to pick a
handful of *representative days*, keep the hours inside each day in
chronological order (so storage can arbitrage across the day), and weight each
day by how many real days it stands for.

This module

* synthesises plausible demand / solar / wind shapes so the planner runs with
  no external data at all,
* reads real profiles from CSV when you have them,
* clusters days with a from-scratch k-medoids, and always keeps the extreme
  net-load day so the reserve-margin and peak-hour behaviour survives the
  reduction.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

HOURS_PER_YEAR = 8760
DAYS_PER_YEAR = 365

__all__ = [
    "synthesise_profiles",
    "load_profiles_csv",
    "RepresentativeDays",
    "cluster_days",
    "aggregate_hours",
]


# ---------------------------------------------------------------------------
# Synthetic profiles
# ---------------------------------------------------------------------------


def _seasonal(day: int, peak_day: int, amplitude: float) -> float:
    """A smooth annual cycle peaking on ``peak_day``."""
    return 1.0 + amplitude * math.cos(2.0 * math.pi * (day - peak_day) / DAYS_PER_YEAR)


def synthesise_profiles(seed: int = 20250101, latitude: float = 23.5) -> Dict[str, List[float]]:
    """Return deterministic 8760-hour shapes for demand, solar and wind.

    The shapes are stylised but structurally realistic: a summer-peaking
    subtropical demand curve with a weekday/weekend split, a solar profile that
    follows daylight hours at the given latitude, and two wind profiles with
    the negative summer correlation typical of monsoon regimes.  Real projects
    should replace these with measured data via :func:`load_profiles_csv`.
    """
    rng = random.Random(seed)

    demand: List[float] = []
    solar: List[float] = []
    wind_on: List[float] = []
    wind_off: List[float] = []

    # Persistent (auto-correlated) noise for wind.
    wind_state = 0.0
    wind_state_off = 0.0

    for hour in range(HOURS_PER_YEAR):
        day = hour // 24
        hod = hour % 24
        weekday = (day % 7) < 5

        # ---- demand: seasonal cooling load + double daily peak -------------
        season = _seasonal(day, peak_day=200, amplitude=0.18)     # July peak
        diurnal = (
            1.0
            + 0.22 * math.sin(2.0 * math.pi * (hod - 9) / 24.0)
            + 0.12 * math.sin(4.0 * math.pi * (hod - 6) / 24.0)
        )
        shape = season * diurnal * (1.0 if weekday else 0.90)
        demand.append(max(0.05, shape * (1.0 + rng.gauss(0.0, 0.012))))

        # ---- solar: daylight window that widens in summer ------------------
        declination = 23.45 * math.sin(2.0 * math.pi * (284 + day) / 365.0)
        phi = math.radians(latitude)
        delta = math.radians(declination)
        cos_omega = -math.tan(phi) * math.tan(delta)
        cos_omega = max(-1.0, min(1.0, cos_omega))
        half_day = math.degrees(math.acos(cos_omega)) / 15.0    # hours of sun / 2
        offset = hod + 0.5 - 12.0
        if abs(offset) < half_day:
            elevation = math.cos(math.pi * offset / (2.0 * half_day))
            clouds = max(0.0, 1.0 - abs(rng.gauss(0.0, 0.22)))
            solar.append(max(0.0, min(1.0, 0.92 * elevation * clouds)))
        else:
            solar.append(0.0)

        # ---- wind: auto-correlated, winter-strong --------------------------
        wind_state = 0.88 * wind_state + rng.gauss(0.0, 1.0)
        wind_state_off = 0.92 * wind_state_off + rng.gauss(0.0, 1.0)
        winter = _seasonal(day, peak_day=15, amplitude=0.35)
        on = 0.24 * winter * math.exp(0.45 * wind_state - 0.10)
        off = 0.46 * winter * math.exp(0.40 * wind_state_off - 0.08)
        wind_on.append(max(0.0, min(1.0, on)))
        wind_off.append(max(0.0, min(1.0, off)))

    # Normalise the demand shape to mean 1.0 so annual energy is exact.
    mean = sum(demand) / len(demand)
    demand = [d / mean for d in demand]

    return {
        "demand": demand,
        "solar": solar,
        "wind_onshore": wind_on,
        "wind_offshore": wind_off,
    }


def load_profiles_csv(path: str) -> Dict[str, List[float]]:
    """Read profiles from a CSV whose header names each column.

    The file must contain 8760 data rows.  A ``demand`` column is rescaled to
    mean 1.0 automatically; capacity-factor columns are used as given.
    """
    columns: Dict[str, List[float]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path}: missing header row")
        for name in reader.fieldnames:
            columns[name.strip()] = []
        for row in reader:
            for name in columns:
                raw = row.get(name, "")
                columns[name].append(float(raw) if raw not in ("", None) else 0.0)

    for name, values in columns.items():
        if len(values) != HOURS_PER_YEAR:
            raise ValueError(
                f"{path}: column {name!r} has {len(values)} rows, expected {HOURS_PER_YEAR}"
            )
    for name, values in columns.items():
        if name.startswith("demand"):
            mean = sum(values) / len(values)
            if mean <= 0:
                raise ValueError(f"{path}: column {name!r} sums to zero")
            columns[name] = [v / mean for v in values]
    return columns


# ---------------------------------------------------------------------------
# Representative days
# ---------------------------------------------------------------------------


@dataclass
class RepresentativeDays:
    """The reduced time domain used by the optimisation."""

    day_indices: List[int]           # which calendar day each cluster stands for
    weights: List[float]             # how many real days each represents
    hours_per_day: int
    hours_per_step: float            # duration of one modelled step, in hours
    peak_day_position: int = 0       # index into ``day_indices`` of the peak day

    @property
    def n_days(self) -> int:
        return len(self.day_indices)

    def steps(self) -> List[Tuple[int, int]]:
        return [(d, h) for d in range(self.n_days) for h in range(self.hours_per_day)]

    def total_hours(self) -> float:
        """Should equal 8760: a check that the weighting is consistent."""
        return sum(self.weights) * self.hours_per_day * self.hours_per_step

    def slice_profile(self, profile: Sequence[float]) -> List[List[float]]:
        """Return ``profile`` reduced to ``[day][hour]`` averages."""
        out = []
        for day in self.day_indices:
            start = day * 24
            block = profile[start:start + 24]
            out.append(aggregate_hours(block, self.hours_per_day))
        return out


def aggregate_hours(day_values: Sequence[float], hours_per_day: int) -> List[float]:
    """Average a 24-hour block down to ``hours_per_day`` chronological steps."""
    if hours_per_day == 24:
        return list(day_values)
    if 24 % hours_per_day:
        raise ValueError("hours_per_day must divide 24")
    width = 24 // hours_per_day
    return [
        sum(day_values[i * width:(i + 1) * width]) / width
        for i in range(hours_per_day)
    ]


def _daily_matrix(profiles: Dict[str, Sequence[float]], keys: Sequence[str]) -> List[List[float]]:
    """Build one feature vector per calendar day, scaled so no series dominates."""
    rows: List[List[float]] = []
    scales = {}
    for key in keys:
        series = profiles[key]
        span = max(series) - min(series)
        scales[key] = span if span > 1e-9 else 1.0
    for day in range(DAYS_PER_YEAR):
        vector: List[float] = []
        for key in keys:
            series = profiles[key]
            scale = scales[key]
            vector.extend(v / scale for v in series[day * 24:(day + 1) * 24])
        rows.append(vector)
    return rows


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        d = x - y
        total += d * d
    return total


def cluster_days(
    profiles: Dict[str, Sequence[float]],
    n_days: int = 8,
    hours_per_day: int = 24,
    seed: int = 20250101,
    feature_keys: Optional[Sequence[str]] = None,
    net_load: Optional[Sequence[float]] = None,
    max_iter: int = 60,
) -> RepresentativeDays:
    """Pick ``n_days`` representative days by k-medoids, keeping the peak day.

    k-medoids (rather than k-means) is used deliberately: the centre of each
    cluster is a *real* day, so the chronology inside it is physically
    consistent and storage cycling behaves sensibly.
    """
    if n_days < 1:
        raise ValueError("n_days must be at least 1")
    keys = list(feature_keys) if feature_keys else sorted(profiles.keys())
    keys = [k for k in keys if len(profiles[k]) == HOURS_PER_YEAR]
    if not keys:
        raise ValueError("no usable 8760-hour profiles to cluster on")

    data = _daily_matrix(profiles, keys)
    n = len(data)
    n_days = min(n_days, n)

    # The day containing the highest net load must survive the reduction,
    # otherwise the model under-builds firm capacity.
    if net_load is None:
        net_load = profiles.get("demand", profiles[keys[0]])
    daily_peak = [max(net_load[d * 24:(d + 1) * 24]) for d in range(n)]
    peak_day = max(range(n), key=lambda d: daily_peak[d])

    rng = random.Random(seed)

    # ---- k-means++ style seeding, with the peak day locked in as medoid 0 --
    medoids = [peak_day]
    while len(medoids) < n_days:
        best = [min(_distance(data[i], data[m]) for m in medoids) for i in range(n)]
        total = sum(best)
        if total <= 0:
            remaining = [i for i in range(n) if i not in medoids]
            if not remaining:
                break
            medoids.append(rng.choice(remaining))
            continue
        target = rng.random() * total
        running = 0.0
        for i, value in enumerate(best):
            running += value
            if running >= target and i not in medoids:
                medoids.append(i)
                break
        else:  # pragma: no cover - numerical fallback
            remaining = [i for i in range(n) if i not in medoids]
            medoids.append(rng.choice(remaining))

    assignment = [0] * n
    for _ in range(max_iter):
        # Assign every day to its nearest medoid.
        for i in range(n):
            best_k, best_d = 0, float("inf")
            for k, m in enumerate(medoids):
                d = _distance(data[i], data[m])
                if d < best_d:
                    best_k, best_d = k, d
            assignment[i] = best_k

        # Move each medoid to the member minimising within-cluster distance.
        moved = False
        for k in range(len(medoids)):
            members = [i for i in range(n) if assignment[i] == k]
            if not members:
                continue
            if k == 0:
                continue  # medoid 0 stays pinned to the peak day
            best_member, best_cost = medoids[k], float("inf")
            for candidate in members:
                cost = 0.0
                for other in members:
                    cost += _distance(data[candidate], data[other])
                    if cost >= best_cost:
                        break
                if cost < best_cost:
                    best_member, best_cost = candidate, cost
            if best_member != medoids[k]:
                medoids[k] = best_member
                moved = True
        if not moved:
            break

    counts = [0] * len(medoids)
    for k in assignment:
        counts[k] += 1
    # A cluster can end up empty; give it a nominal weight of zero rather than
    # dropping it, so indices stay stable.
    weights = [float(c) for c in counts]

    # Rescale so the weights reproduce exactly 365 days.
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w * DAYS_PER_YEAR / total_weight for w in weights]

    return RepresentativeDays(
        day_indices=list(medoids),
        weights=weights,
        hours_per_day=hours_per_day,
        hours_per_step=24.0 / hours_per_day,
        peak_day_position=0,
    )
