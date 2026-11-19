"""`loan-agents eval-cases`: scripted applications through the whole agent team, scored.

cases.py reads evals/cases.yaml, runner.py plays the cases, scoring.py scores them and
report.py prints and saves the results. The pass rules are in the README.
"""

from loan_agents.evaluation.cases import CASES_FILE, load_cases
from loan_agents.evaluation.report import write_report
from loan_agents.evaluation.runner import EvalRun, run_eval

__all__ = ["CASES_FILE", "EvalRun", "load_cases", "run_eval", "write_report"]
