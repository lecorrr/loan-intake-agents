"""The run's report: a line per case with what went wrong, a summary, and a JSON file.

The data is synthetic, so the report shows the values the model got wrong.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from loan_agents import __version__
from loan_agents.evaluation.scoring import CaseResult, FieldScore

if TYPE_CHECKING:
    from loan_agents.evaluation.runner import EvalRun


def _seconds(milliseconds: float) -> str:
    return f"{milliseconds / 1000:.1f} s"


def case_report(result: CaseResult) -> str:
    verdict = "pass" if result.passed else "FAIL"
    fields = f"fields {result.fields.right}/{result.fields.expected}"
    if result.fields.extra:
        fields += f" +{result.fields.extra}"
    photo = (
        f"photo {result.reading.right}/{result.reading.expected}"
        if result.reading.expected or result.reading.extra
        else "no photo"
    )
    tokens = result.input_tokens + result.output_tokens
    total_ms = sum(ms for _, ms in result.turns)
    lines = [
        f"{result.id:4} {verdict}  {fields:14} {photo:11} {_seconds(total_ms):>7}  "
        f"{tokens:>6,} tokens  {result.about}"
    ]
    details = (
        ([f"error: {result.error}"] if result.error else [])
        + [f"guardrail broken: {breach}" for breach in result.breaches]
        + result.problems
        + result.fields.notes
        + result.reading.notes
        + result.notes
    )
    lines += [f"       - {detail}" for detail in details]
    return "\n".join(lines)


def _percentile(values: Sequence[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[max(1, math.ceil(share * len(ordered))) - 1]


def summary(results: Sequence[CaseResult]) -> str:
    fields, reading = FieldScore(), FieldScore()
    for result in results:
        fields.add(result.fields)
        reading.add(result.reading)
    passed = sum(result.passed for result in results)
    breaches = sum(len(result.breaches) for result in results)
    lines = [
        f"Cases passed: {passed}/{len(results)}",
        f"Fields extracted right: {fields.right}/{fields.expected} "
        f"({fields.wrong} wrong, {fields.missed} missed; {fields.extra} the client did not give)",
        f"Certificate fields read right: {reading.right}/{reading.expected} "
        f"({reading.wrong} wrong, {reading.missed} missed; {reading.extra} not on the photo)",
        "Guardrails: held in every case"
        if breaches == 0
        else f"Guardrails: broken {breaches} times; see the cases above",
    ]
    for kind in ("message", "photo"):
        times = [ms for result in results for turn, ms in result.turns if turn == kind]
        if times:
            lines.append(
                f"Time per {kind}: {_seconds(_percentile(times, 0.5))} median, "
                f"{_seconds(_percentile(times, 0.95))} p95, over {len(times)} "
                + ("turn" if len(times) == 1 else "turns")
            )
    tokens = sum(result.input_tokens + result.output_tokens for result in results)
    if tokens:
        lines.append(f"Model tokens: {tokens:,} in all, {tokens // len(results):,} per case")
    return "\n".join(lines)


def write_report(run: EvalRun, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"cases-{run.run_id}.json"
    report = {
        "version": __version__,
        "model": run.model,
        "run": run.run_id,
        "today": run.today.isoformat(),
        "summary": summary(run.results).splitlines(),
        "cases": [asdict(result) | {"passed": result.passed} for result in run.results],
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
