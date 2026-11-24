#!/usr/bin/env python3
"""Granularity gate (Step 2.2): fail non-zero on any missing suite/scenario/negative.

Checks, all derived from the plan's use-case table
(``docs/plans/2026-11-23-15-27-airllm-exhaustive-lab-plan.md``):

1. per-feature component suite exists with one scenario per use-case id;
2. per-dependency live suite exists with one scenario per dependency use case;
3. e2e suite holds one scenario per feature use case (runs last);
4. every use case has a negative scenario in each tier that covers it;
5. every unit file holds a negative test and every ``src/`` function is
   exercised by a unit test (static proxy; the real 100% line+branch proof
   is the pytest-cov gate, optionally verified here via ``--coverage-xml``);
6. all five ``nb-*`` use cases map to their paired notebook + papermill
   executed output -- verified as mapping/table entries IN THE PLAN
   (notebooks land in Step 3.1, so notebook files are NOT required here);
7. no banned weight strings (70B/405B/671B/Kimi/Flash-Next/real tokens).

Scenario tags are read with ``ast`` from decorators
``@pytest.mark.scenario("<use-case>")`` / ``dep`` / ``negative`` plus the
module-level ``FEATURE`` assignment. Exit 0 + "zero missing" on green.
"""

from __future__ import annotations

import argparse
import ast
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLAN = Path("/Users/mavasbel/Dev/repos/docs/plans/2026-11-23-15-27-airllm-exhaustive-lab-plan.md")

# (feature, component file, {use-case: [dep use-cases]}) -- plan use-case table.
COMPONENT = {
    "tests/component/test_bench.py": (
        "bench",
        {
            "bench-tiny-cpu": ["hub-fetch-tiny", "shard-split-reuse"],
            "bench-compression-matrix": ["shard-split-reuse"],
        },
    ),
    "tests/component/test_infer.py": (
        "infer",
        {
            "infer-generate-cpu": ["hub-fetch-tiny"],
            "infer-chat-template": ["hub-fetch-tiny"],
            "infer-profiling": ["shard-split-reuse"],
            "infer-negative-mac-error": ["shard-split-reuse"],
        },
    ),
    "tests/component/test_train_surface.py": (
        "train",
        {"train-surface": ["train-cli-help", "train-blocked-mac"]},
    ),
}

# (live file, dependency, {dep use-case}) -- one suite per dependency.
LIVE = {
    "tests/live/test_hub.py": ("huggingface-hub", {"hub-fetch-tiny"}),
    "tests/live/test_shards.py": ("airllm-shards", {"shard-split-reuse"}),
    "tests/live/test_train_cli.py": (
        "airllm-train",
        {"train-cli-help", "train-blocked-mac"},
    ),
}

# e2e: one scenario per feature use case (notebooks excluded by design).
E2E_FILE = "tests/e2e/test_lab.py"
E2E_SCENARIOS = {
    "bench-tiny-cpu",
    "bench-compression-matrix",
    "infer-generate-cpu",
    "infer-chat-template",
    "infer-profiling",
    "infer-negative-mac-error",
    "train-surface",
}

# nb-* use cases -> paired notebook + papermill output (plan mapping only).
NOTEBOOKS = {
    "nb-testing-smoke": "notebooks/testing/01-smoke.ipynb",
    "nb-inference-minimal": "notebooks/inference/02-minimal-generate.ipynb",
    "nb-inference-chat-profiling": "notebooks/inference/03-chat-profiling.ipynb",
    "nb-training-surface": "notebooks/training/04-train-surface.ipynb",
    "nb-benchmarking-matrix": "notebooks/benchmarking/05-tiny-matrix.ipynb",
}
EXECUTED_DIR = "notebooks/executed/"

BANNED = ["70B", "405B", "671B", "33B", "Kimi", "Flash-Next", "use_auth_token", "hf_token="]


def parse_marks(path):
    """Return (feature, {scenario: {'deps': set, 'negative': bool}})."""
    tree = ast.parse(path.read_text(), filename=str(path))
    feature = None
    scenarios = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "FEATURE":
            value = node.value
            if isinstance(value, ast.Constant):
                feature = value.value
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        scen, deps, neg = None, set(), False
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            func = dec.func
            # Unwrap pytest.mark.<kind>: Attribute(mark) over Name(pytest).
            if not (isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Attribute)
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "pytest"
                    and func.value.attr == "mark"):
                continue
            kind = func.attr
            args = [a.value for a in dec.args if isinstance(a, ast.Constant)]
            if kind == "scenario" and args:
                scen = args[0]
            elif kind == "dep":
                deps.update(args)
            elif kind == "negative":
                neg = True
        if scen is not None:
            entry = scenarios.setdefault(scen, {"deps": set(), "negative": False})
            entry["deps"] |= deps
            entry["negative"] = entry["negative"] or neg
    return feature, scenarios


def src_functions():
    """{module: [function names]} for every src/airllm_lab/*.py (minus dunders)."""
    out = {}
    for path in sorted((REPO / "src" / "airllm_lab").glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        names = [
            n.name
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not (n.name.startswith("__") and n.name.endswith("__"))
        ]
        out[path.name] = names
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-xml", default=None,
                        help="optional coverage.xml to verify 100%% line+branch")
    args = parser.parse_args()
    missing = []

    def need(condition, message):
        if not condition:
            missing.append(message)
        return condition

    # 1+2. component + live suites, scenarios, deps, negatives.
    for rel, (feature, expected) in COMPONENT.items():
        path = REPO / rel
        need(path.exists(), "suite file missing: %s" % rel)
        if not path.exists():
            continue
        got_feature, scenarios = parse_marks(path)
        need(got_feature == feature, "%s: FEATURE=%r, want %r" % (rel, got_feature, feature))
        for use_case, deps in expected.items():
            need(use_case in scenarios, "%s: scenario missing: %s" % (rel, use_case))
            if use_case in scenarios:
                for dep in deps:
                    need(dep in scenarios[use_case]["deps"],
                         "%s: %s missing dep tag: %s" % (rel, use_case, dep))
                need(scenarios[use_case]["negative"],
                     "%s: %s missing negative scenario" % (rel, use_case))
    for rel, (dep, expected) in LIVE.items():
        path = REPO / rel
        need(path.exists(), "live suite missing: %s (dependency %s)" % (rel, dep))
        if not path.exists():
            continue
        _, scenarios = parse_marks(path)
        for use_case in expected:
            need(use_case in scenarios, "%s: dep scenario missing: %s" % (rel, use_case))
            if use_case in scenarios:
                need(scenarios[use_case]["negative"],
                     "%s: %s missing negative scenario" % (rel, use_case))

    # 3+4. e2e scenarios + negatives.
    e2e = REPO / E2E_FILE
    need(e2e.exists(), "e2e suite missing: %s" % E2E_FILE)
    if e2e.exists():
        _, scenarios = parse_marks(e2e)
        for use_case in sorted(E2E_SCENARIOS):
            need(use_case in scenarios, "%s: scenario missing: %s" % (E2E_FILE, use_case))
            if use_case in scenarios:
                need(scenarios[use_case]["negative"],
                     "%s: %s missing negative scenario" % (E2E_FILE, use_case))

    # 5. unit negatives + static src-function coverage.
    unit_dir = REPO / "tests" / "unit"
    unit_text = ""
    for path in sorted(unit_dir.glob("test_*.py")):
        _, scenarios = parse_marks(path)
        unit_text += path.read_text()
        has_negative = ("negative" in path.read_text().lower()
                        or any(v["negative"] for v in scenarios.values()))
        need(has_negative, "%s: no negative test" % path.name)
    for module, names in src_functions().items():
        for name in names:
            need(name in unit_text, "unit: src %s function untested: %s" % (module, name))

    # 6. nb-* mapping entries live in the PLAN (notebooks land in Step 3.1).
    plan_text = PLAN.read_text() if PLAN.exists() else ""
    need(PLAN.exists(), "plan file missing: %s" % PLAN)
    for use_case, notebook in NOTEBOOKS.items():
        need(use_case in plan_text, "plan mapping missing use case: %s" % use_case)
        need(notebook in plan_text, "plan mapping missing notebook: %s" % notebook)
    need(EXECUTED_DIR in plan_text, "plan mapping missing papermill output dir: %s" % EXECUTED_DIR)

    # 7. banned strings nowhere in tests.
    for path in sorted((REPO / "tests").rglob("test_*.py")):
        text = path.read_text()
        for banned in BANNED:
            need(banned not in text, "%s: banned string present: %s" % (path.name, banned))

    # Optional: verify a real coverage.xml at 100% line+branch.
    if args.coverage_xml:
        xml_path = Path(args.coverage_xml)
        need(xml_path.exists(), "coverage xml missing: %s" % xml_path)
        if xml_path.exists():
            root = ET.parse(str(xml_path)).getroot()
            for cls in root.iter("class"):
                line_rate = float(cls.get("line-rate", 1))
                branch_rate = float(cls.get("branch-rate", 1))
                need(line_rate == 1.0 and branch_rate == 1.0,
                     "coverage <100%% in %s (line=%s branch=%s)"
                     % (cls.get("filename"), line_rate, branch_rate))

    print("granularity check: %d missing item(s)" % len(missing))
    for item in missing:
        print("MISSING: %s" % item)
    if missing:
        return 1
    print("zero missing: suites, scenarios, negatives, unit refs, nb-plan mapping clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
