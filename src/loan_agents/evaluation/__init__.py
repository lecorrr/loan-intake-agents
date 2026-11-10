"""`loan-agents eval-cases`: scripted applications through the whole agent team, scored.

evals/cases.yaml holds the cases: client messages with the fields each one holds, certificate
photos drawn from synthetic data, and what the manager must receive and then decide. Each case
runs through the service, the way Telegram and the HTTP API drive it, with the configured model,
an in-memory CRM and case store, and a temporary folder for uploads. A case passes when:

- every photo is accepted or turned down as the case says;
- it reaches the manager, or not, with the expected flags and with or without a contract draft;
- the case then holds the right data, compared in the masked form the API shows;
- after the manager's decision it has the right status and CRM stage, and the client was told;
- the guardrails held: no decision reached the client before the manager made one, no contract
  was drafted for a flagged case, nothing reached the manager without a certificate, and the
  manager got one card per case.

The model is also scored field by field: what it extracted from each message and what it read
from each photo. With --fake the case file's own answers play the model, which checks the case
file itself: every case must then pass.

cases.py reads the case file, runner.py plays the cases, scoring.py scores them and report.py
prints and saves the results.
"""

from loan_agents.evaluation.cases import CASES_FILE, load_cases
from loan_agents.evaluation.report import write_report
from loan_agents.evaluation.runner import EvalRun, run_eval

__all__ = ["CASES_FILE", "EvalRun", "load_cases", "run_eval", "write_report"]
