# Changelog

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
