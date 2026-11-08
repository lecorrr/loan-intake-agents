# Changelog

## 0.2.0 — 2026-11-08

The CRM is now a self-hosted EspoCRM instead of Bitrix24, whose free plan has no REST webhooks.

- `adapters/espocrm.py`: contact and deal through EspoCRM's REST API with an API key. The deal is
  found by its case field, so a retried step updates it instead of making a second one; a
  returning client (same phone and name) keeps one contact. The summary goes into the deal's
  description, the draft contract and the decision into its stream. Built and tested against a
  real EspoCRM 10.0.11: the required fields, the name-only duplicate check, the error format and
  how phone numbers are replaced all come from its answers.
- Retries only what cannot apply twice: reads and updates on a network error or 502/503/504,
  creates only when they never reached EspoCRM.
- `loan-agents crm-setup`: signs in once as the administrator and adds UAH, the case field
  `cLoanCaseId` (shown on the deal card), the role "Loan intake bot" and the API user, then prints
  the key for `.env`, or saves it there with `--write-env`. Running it again repairs what is
  missing and changes nothing else.
- `serve` stops with one line, not a traceback, when `ESPOCRM_URL` is set without a key.
- `doctor` signs in with the key and checks the stages, the case field and the currency.
- The manager's card links to the deal in EspoCRM.
- Docker compose runs EspoCRM 10.0.11 with MariaDB on 127.0.0.1:8090 in place of the mock.
- Removed: the Bitrix24 adapter, the Bitrix24 mock and `loan-agents mock-crm`.
- `.gitignore` and `.dockerignore` also cover copies of `.env` such as `.env.before-espocrm`.
- Tests: an in-memory EspoCRM modelled on the real one's answers, and live tests that run when
  `LOAN_AGENTS_TEST_ESPOCRM_URL` is set.
- Upgrading: remove the `B24_*` and `MOCK_B24_SECRET` lines from `.env` (unknown keys stop the
  app), set `ESPOCRM_URL` and `ESPOCRM_ADMIN_PASSWORD`, start compose with `--remove-orphans` so
  the old mock container frees port 8090, and run `crm-setup --write-env`. Cases that were waiting
  for a manager before the upgrade point at deals in the old mock; start new ones.

## 0.1.2 — 2026-11-08

- Telegram: messages from one chat are handled in the order they arrived. aiogram runs
  synchronous filters in a thread pool, so a burst from one client could be answered out of
  order (reproduced in 14 of 15 runs); the dispatcher now uses aiogram's per-chat event isolation.
- Telegram: fetching a photo waits at most 10 s for Telegram's file lookup and 20 s for the
  download, and tries once more after a network failure. Before, one stalled lookup held the
  client for 60 s and then asked for the photo again.
- Telegram: stopping the app no longer logs a false "Failed to fetch updates" ERROR. Polling now
  stops through aiogram's `stop_polling`; cancelling its task had left an inner polling task
  running against a closed session.
- README: accuracy and timings measured with the real model; notes on Docker restarts and logs.

## 0.1.1 — 2026-11-08

- The default model is now `qwen3-vl:8b-instruct`. The plain `qwen3-vl:8b` tag is the reasoning
  build, and it kept reasoning with thinking switched off: about 450 output tokens and 9–13 s per
  certificate, against 97 tokens and 2–3 s (RTX 4060 Ti, Ollama 0.40.15).
- `doctor` warns when the configured Ollama model is a reasoning build.
- Every model call logs `input_tokens` and `output_tokens`.
- `eval-documents` prints what was read next to what was expected for each wrong field, and takes
  `--runs N`.
- Samples: the ЗРАЗОК stamp moved to an empty corner. The old stamp across the fields lives on in
  `certificate-stamp-over-fields.png`, a hard case where qwen3-vl reads OCTAVIA as OSTAVIA. A test
  keeps the regular stamp clear of every field.
- Upgrading: set `LLM_MODEL=ollama:qwen3-vl:8b-instruct` in `.env` and run
  `ollama pull qwen3-vl:8b-instruct`.

## 0.1.0 — 2026-11-08

First version.

- LangGraph agent team: a code supervisor routes between intake, document reading, valuation,
  CRM, contract drafting and a manager approval step that pauses the case with `interrupt()`.
- Telegram bot for clients (private chats) and manager decision buttons; HTTP API for a web
  channel and decisions.
- Local model through Ollama (`qwen3-vl:8b` by default) via LangChain, with the answer schema
  enforced and every extraction validated before it reaches the case.
- Bitrix24 through an inbound webhook: idempotent contact and deal, stages, comments, retries on
  rate limits. A Bitrix24-compatible mock for demos.
- Postgres for case checkpoints and the audit trail; memory when no database is set.
- `loan-agents` commands: `serve`, `demo`, `doctor`, `eval-documents`, `make-samples`, `mock-crm`.
- Docker compose with Postgres and the mock; the app container on the host network.
