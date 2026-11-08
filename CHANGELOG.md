# Changelog

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
