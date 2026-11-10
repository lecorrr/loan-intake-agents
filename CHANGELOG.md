# Changelog

## 0.4.0 — 2026-11-09

An end-to-end evaluation: 20 scripted applications, five of them with bad or missing documents.

- `evals/cases.yaml`: the applications as client conversations, with the fields each message
  holds, certificate photos drawn from synthetic data, and what the manager must receive and then
  decide. The cases vary how clients write, and cover a receipt sent instead of the certificate,
  a car registered to the client's father, an older car than the one named, a photo too blurred
  to read, no photo at all, the three policy flags and a manager who asks for more documents.
- `loan-agents eval-cases`: runs every case through the service with the local model, an
  in-memory CRM and case store and a temporary folder for photos. It checks each photo's
  acceptance, what reaches the manager, the data the case holds, and the status, CRM stage and
  client message after the decision; and in every case the guardrails: no decision before the
  manager's, no contract for a flagged case, nothing to the manager without a certificate, one
  card per case. It also scores the model field by field, times every turn, counts tokens, and
  writes `data/evals/cases-<run>.json`. With Langfuse running, each case is its own session.
  `--fake` lets the case file's answers play the model; the tests run all 20 cases that way.
- The model adapter reports each call's token counts to an optional callback.
- Synthetic certificates can be blurred; a blurred one's answer key holds no fields.

## 0.3.1 — 2026-11-09

- Docker compose: Langfuse's web container never became healthy, so `up --wait` failed
  after six and a half minutes. Its server listened only on the container's own address,
  which Docker names in `HOSTNAME`, while the health check asked 127.0.0.1. Compose now sets
  `HOSTNAME=0.0.0.0` for it.

## 0.3.0 — 2026-11-09

Langfuse tracing.

- `adapters/tracing_langfuse.py`: one trace per turn, named `start`, `message`, `document` or
  `decision`, and one session per case. Each step of the team and each model call is recorded
  with its timing; model calls also with the model's name and token counts. Traces carry the app
  version as their release.
- Masking before anything is sent: `core.masking.mask_personal_data` masks names wherever they
  appear once a name field has held them, phones, VINs and plates by field and by their shape in
  any text, replaces the client's own messages by their length and pictures by a note. The SDK's
  own upload of pictures is turned off, and the text of LangGraph's interrupt, which holds the
  manager card, is replaced.
- `loan-agents tracing-keys`: makes the Langfuse keys and the administrator's password that `.env`
  lacks, prints them or saves them with `--write-env`, and never replaces one.
- `doctor` checks the Langfuse keys, and says how to start Langfuse when it does not answer.
- Docker compose runs Langfuse 4 under `--profile tracing`: the web interface on 127.0.0.1:3000,
  plus its worker, Postgres, ClickHouse, MinIO and Redis, none published. When it starts, it
  creates what is missing of the organization, the project, the keys and the user from `.env`.
- New settings: `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` (both or
  neither), `LANGFUSE_ADMIN_EMAIL`, `LANGFUSE_ADMIN_PASSWORD`.
- New dependency: `langfuse` 5.2 or newer within 5.x; it needs a Langfuse server 4.59 or newer, or
  Langfuse Cloud.
- Upgrading: nothing is required, since without keys nothing is traced. To turn tracing on, run
  `tracing-keys --write-env`, start the profile and restart the app.

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
