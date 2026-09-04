"""Command-line interface: ``python -m energyplan ...``

Subcommands
-----------
``solve``        optimise one scenario and report it
``sensitivity``  sweep one parameter and tabulate the headline results
``compare``      solve several scenario files side by side
``example``      write a ready-to-run example scenario
``solvers``      list the solver backends available here
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Sequence

from . import __version__


def _load_spec(path: str) -> dict:
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError:
            raise SystemExit(
                "PyYAML is needed to read .yaml scenarios. "
                "Install it (pip install pyyaml) or use the JSON form."
            )
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="energyplan",
        description="Least-cost energy investment planning.",
    )
    parser.add_argument("--version", action="version", version=f"energyplan {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(sub):
        sub.add_argument("--solver", default="auto",
                         choices=["auto", "highs", "cbc", "simplex"],
                         help="LP backend (default: fastest available)")
        sub.add_argument("--set", dest="overrides", action="append", default=[],
                         metavar="PATH=VALUE",
                         help="override a scenario field, e.g. "
                              "--set technologies.solar.capex=600 "
                              "(append '*' to the path to multiply instead)")
        sub.add_argument("--days", type=int, default=None,
                         help="number of representative days")
        sub.add_argument("--hours", type=int, default=None,
                         help="modelled hours per representative day (must divide 24)")
        sub.add_argument("-v", "--verbose", action="store_true")

    solve = subparsers.add_parser("solve", help="optimise one scenario")
    solve.add_argument("scenario", help="path to a JSON or YAML scenario")
    solve.add_argument("--html", metavar="FILE", help="write an HTML report")
    solve.add_argument("--csv", metavar="DIR", help="write CSV result tables")
    solve.add_argument("--json", metavar="FILE", help="write a JSON summary")
    solve.add_argument("--quiet", action="store_true", help="suppress the text report")
    common(solve)

    sens = subparsers.add_parser("sensitivity", help="sweep one parameter")
    sens.add_argument("scenario")
    sens.add_argument("--vary", required=True, metavar="PATH",
                      help="scenario field to sweep, e.g. technologies.solar.capex")
    sens.add_argument("--values", required=True,
                      help="comma-separated values, e.g. 500,700,900")
    sens.add_argument("--csv", metavar="FILE", help="write the sweep table as CSV")
    common(sens)

    comp = subparsers.add_parser("compare", help="solve several scenarios side by side")
    comp.add_argument("scenarios", nargs="+")
    comp.add_argument("--csv", metavar="FILE")
    common(comp)

    example = subparsers.add_parser("example", help="write an example scenario")
    example.add_argument("output", nargs="?", default="scenario.yaml")
    example.add_argument("--name", default="island",
                         choices=["island", "minimal"],
                         help="which bundled example to write")

    subparsers.add_parser("solvers", help="list available LP backends")
    return parser


def _prepare(args) -> "tuple":
    from .data import load_scenario
    from .study import apply_overrides

    spec = _load_spec(args.scenario)
    if args.overrides:
        spec = apply_overrides(spec, args.overrides)
    if args.days is not None:
        spec["representative_days"] = args.days
    if args.hours is not None:
        spec["hours_per_day"] = args.hours
    return spec, load_scenario(spec)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "solvers":
        from .solvers import available_solvers

        names = available_solvers()
        print("Available LP backends (fastest first):")
        for name in names:
            note = {
                "highs": "SciPy/HiGHS - recommended for large models",
                "cbc": "PuLP/CBC",
                "simplex": "built-in, pure Python, no dependencies",
            }[name]
            print(f"  {name:<10} {note}")
        return 0

    if args.command == "example":
        from .examples import write_example

        path = write_example(args.name, args.output)
        print(f"Wrote {path}")
        print(f"Run it with:  python -m energyplan solve {path} --html plan.html")
        return 0

    from .data import ScenarioError
    from .simplex import SimplexError
    from .model import CapacityExpansionModel
    from .report import html_report, text_report, write_csv, write_json
    from .study import compare, run_sensitivity, summarise

    try:
        if args.command == "solve":
            spec, scenario = _prepare(args)
            model = CapacityExpansionModel(scenario)
            started = time.perf_counter()
            problem = model.build()
            if args.verbose:
                stats = problem.stats()
                print(f"[model] {stats['variables']:,} variables, "
                      f"{stats['constraints']:,} constraints, "
                      f"{stats['nonzeros']:,} non-zeros "
                      f"(built in {time.perf_counter() - started:.1f}s)",
                      file=sys.stderr)
            result = model.solve(solver=args.solver, verbose=args.verbose)
            if not result.optimal:
                print(f"No optimal plan found: {result.status}", file=sys.stderr)
                if result.solution.message:
                    print(result.solution.message, file=sys.stderr)
                return 2
            if not args.quiet:
                print(text_report(result))
            if args.html:
                print(f"Wrote {html_report(result, args.html)}")
            if args.csv:
                for path in write_csv(result, args.csv):
                    print(f"Wrote {path}")
            if args.json:
                print(f"Wrote {write_json(result, args.json)}")
            worst = max(result.audit().values())
            return 0 if worst < 1e-6 else 3

        if args.command == "sensitivity":
            spec, _ = _prepare(args)
            values: List = []
            for token in args.values.split(","):
                token = token.strip()
                try:
                    values.append(json.loads(token))
                except json.JSONDecodeError:
                    values.append(token)
            rows = run_sensitivity(
                spec, args.vary, values, solver=args.solver,
                progress=(lambda msg: print(msg, file=sys.stderr)) if args.verbose else None,
            )
            print(f"\nSENSITIVITY: {args.vary}")
            print(compare(rows))
            if args.csv:
                _write_rows_csv(rows, args.csv)
                print(f"\nWrote {args.csv}")
            return 0

        if args.command == "compare":
            from .study import apply_overrides
            from .data import load_scenario

            rows = []
            for path in args.scenarios:
                spec = _load_spec(path)
                if args.overrides:
                    spec = apply_overrides(spec, args.overrides)
                if args.days is not None:
                    spec["representative_days"] = args.days
                if args.hours is not None:
                    spec["hours_per_day"] = args.hours
                scenario = load_scenario(spec)
                if args.verbose:
                    print(f"solving {path} ...", file=sys.stderr)
                result = CapacityExpansionModel(scenario).solve(solver=args.solver)
                rows.append(summarise(result, label=os.path.basename(path)))
            print(compare(rows))
            if args.csv:
                _write_rows_csv(rows, args.csv)
                print(f"\nWrote {args.csv}")
            return 0

    except ScenarioError as error:
        print(f"Scenario error: {error}", file=sys.stderr)
        return 1
    except SimplexError as error:
        print(f"Solver error: {error}", file=sys.stderr)
        return 1
    except FileNotFoundError as error:
        print(f"File not found: {error.filename}", file=sys.stderr)
        return 1

    parser.error("unknown command")
    return 1


def _write_rows_csv(rows, path: str) -> None:
    import csv

    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
