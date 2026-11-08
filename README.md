# Loan intake agents

A LangGraph agent team that takes a car-secured loan application from a Telegram chat to a CRM
deal and a draft contract, then hands the case to a manager, who decides with a button press.
The model runs locally in Ollama. All data is synthetic.

```
client ── Telegram / HTTP ──► service ──► LangGraph team ──────────────► Bitrix24
                                            supervisor (code) routes:       contact, deal,
                                            intake · documents · valuation  stage, comments
                                            · crm · contract
                                                 │
                                   interrupt: masked card + draft contract
                                                 ▼
                                   manager in Telegram: approve / reject / need documents
                                                 │
                                   resume ──► client is told, deal moves to its stage
```

## What it does

- Collects name, phone, amount, term and the car from free text in Ukrainian, asks for one
  missing item at a time, and answers stock questions (rate, documents, timing) from approved
  texts.
- Reads a photo of the vehicle registration certificate with a vision model, checks the VIN and
  plate formats, and cross-checks make, model, year and owner against the application.
- Values the car from a price table and applies the lending policy: loan-to-value, car age,
  amount and term limits, monthly payment.
- Writes the contact and the deal to Bitrix24. Repeated writes update the same deal instead of
  creating new ones; the deal moves through stages and collects comments for the manager.
- Drafts the loan agreement from a template when nothing is flagged.
- Sends the manager a card with personal data masked, plus the draft. The manager's button
  press resumes the case. The agents never approve, reject, or tell the client a decision.

## Try it without a model or any accounts

```
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/loan-agents demo --fake
```

The demo plays a scripted client conversation and prints the bot's replies, the manager card,
the draft contract and the final CRM state. `--fake` replaces the model with a rule-based reader
and the CRM with memory. If `python3 -m venv` complains about `ensurepip`, install the
`python3-venv` package first.

Checks:

```
.venv/bin/pytest
.venv/bin/ruff check src tests
```

The Postgres tests run only when `LOAN_AGENTS_TEST_DATABASE_URL` points at a database (see
`tests/test_postgres.py`).

## With the local model

The default model is `qwen3-vl:8b-instruct`: about a 6 GB download, 5.8 GB in VRAM, so it runs
comfortably on a 16 GB GPU. It needs Ollama 0.12.7 or newer. On smaller cards, try a 4b instruct
build from Ollama's qwen3-vl page.

```
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-vl:8b-instruct
cp .env.example .env
.venv/bin/loan-agents doctor
.venv/bin/loan-agents eval-documents
.venv/bin/loan-agents demo
```

`doctor` prints every setting (secrets hidden) and checks the policy, prices, template, Ollama,
Bitrix24, Postgres and Telegram; it warns when the model is a reasoning build. `eval-documents`
reads the synthetic certificates in `samples/`, scores every field against the ground truth
embedded in each PNG, and prints what was read next to what was expected; `--runs 3` reads each
sample three times. The first model call after a start also loads the model into VRAM, so it is
slower than the rest.

### Why the instruct build

The plain `qwen3-vl:8b` tag is the reasoning build. It reasons before every answer even when asked
not to, and that reasoning is the slow part. Measured on an RTX 4060 Ti 16 GB with Ollama 0.40.15:

| | `qwen3-vl:8b-instruct` | `qwen3-vl:8b` (reasoning) |
|---|---|---|
| Certificate read, model loaded | 2–3 s | 9–13 s |
| Output tokens per certificate | 97 | about 450 |
| One client message | 1.5–1.6 s | not measured |

Every model call logs `input_tokens` and `output_tokens`, so hidden reasoning shows up in the log.

### A known limit

`samples/certificate-stamp-over-fields.png` has a large ЗРАЗОК stamp across the fields. Both builds
read OCTAVIA under it as OSTAVIA, in every run; without the stamp the instruct build reads it right.
The cross-check catches the misread: the model differs from the application and has no price, so
the case reaches the manager flagged and without a draft contract. The regular samples carry a
small stamp in an empty corner instead, and a test keeps it clear of every field.

## Telegram

1. Create a bot with [@BotFather](https://t.me/BotFather) (`/newbot`) and put its token in
   `TELEGRAM_BOT_TOKEN` in `.env`.
2. Find the manager chat id. Send `/start` to your bot, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `message.chat.id`. Do this while the
   app is not running, because a running bot takes the updates itself.
   - For a solo demo, use your own id: you are then client and manager in the same chat.
   - For a managers' group, add the bot to the group, send `/start@<your_bot>` there, and use the
     group's negative id.
3. Put the id in `MANAGER_CHAT_ID`. `loan-agents doctor` confirms that the bot can reach that chat.
4. Run `.venv/bin/loan-agents serve` and write to the bot.

Clients are served in private chats only, and only buttons pressed in the manager chat count.
`/new` starts the conversation over.

## Bitrix24

Bitrix24's free plan has no REST webhooks, so a compatible mock is included:

```
.venv/bin/loan-agents mock-crm
```

Set `B24_WEBHOOK_URL=http://127.0.0.1:8090/rest/1/demo-secret/` and open http://127.0.0.1:8090/
to watch deals, contacts and comments arrive. The mock keeps data in memory only.

For a real portal (a paid plan or a trial), create an inbound webhook with the CRM scope under
Developer resources → Other → Inbound webhook, and put its URL in `B24_WEBHOOK_URL`. If your deal
pipeline uses other stage ids, set `B24_STAGE_*`. `loan-agents doctor` checks the webhook by
calling `crm.deal.fields`.

## Docker compose

```
cp .env.example .env
docker compose up -d --build
docker compose exec app loan-agents doctor
```

This starts Postgres on 127.0.0.1:5433, the Bitrix24 mock on 127.0.0.1:8090, and the app on
127.0.0.1:8000, which also polls Telegram if a token is set. Ollama stays on the host, where the
GPU is. The app container uses the host network, so it reaches Ollama on 127.0.0.1:11434 and uses
the same URLs as a native run; this needs Docker on Linux. Logs: `docker compose logs -f app`.

## HTTP API

Interactive docs are at http://127.0.0.1:8000/docs.

| Method | Path | Body or header | Does |
|---|---|---|---|
| GET | `/health` | | Version and Telegram polling state |
| POST | `/api/cases` | | Opens a web case; returns its id and the greeting |
| POST | `/api/cases/{id}/messages` | `{"text": "..."}` | One client message |
| POST | `/api/cases/{id}/documents` | multipart `file` | The certificate photo |
| GET | `/api/cases/{id}` | `X-Manager-Token` for Telegram cases | The case, personal data masked |
| POST | `/api/cases/{id}/decision` | `X-Manager-Token`, `{"action": "approve"}` | Manager decision: `approve`, `reject` or `need_documents`, optional `comment` |

Web case ids are random, and the id is the client's access to that case.

```
curl -s -X POST http://127.0.0.1:8000/api/cases
curl -s -X POST http://127.0.0.1:8000/api/cases/<id>/messages \
  -H 'Content-Type: application/json' -d '{"text": "Мене звати Тарас Шевченко"}'
curl -s -X POST http://127.0.0.1:8000/api/cases/<id>/documents -F file=@samples/certificate-ok.png
curl -s -X POST http://127.0.0.1:8000/api/cases/<id>/decision -H "X-Manager-Token: <token>" \
  -H 'Content-Type: application/json' -d '{"action": "approve"}'
```

## Limits by design

- Agents may ask questions, read documents, value the car, write to the CRM and draft the
  contract. They may not approve, reject, quote final terms or move money; only the step that runs
  after the manager's decision speaks about decisions.
- Every model output is validated (types, ranges, formats, cross-checks) before it reaches the
  case.
- With the default local model, no client data goes to a model provider.
- Logs and the audit trail carry case ids, field names and numbers, not personal values. The API's
  case view masks names, phones, VINs and plates; the manager's card masks phones, VINs and plates.
  Full values are kept only where the work needs them: the CRM, the uploaded photo, and the draft
  contract (stored under `data/uploads/` and attached to the manager's card).

## Layout

```
src/loan_agents/
  contracts.py        shared types, units and formats
  core/               pure logic: policy, routing, validation, masking, texts, contract
  agents/graph.py     the LangGraph team
  service.py          one case at a time: locks, uploads, pending decisions
  adapters/           LangChain/Ollama, Bitrix24, Telegram, Postgres, files, CSV prices
  fakes/              stand-ins for tests and --fake
  wiring.py           picks the adapters from the settings
  api.py, cli.py, telegram_bot.py   entry points
  mock_bitrix24.py    the Bitrix24-compatible mock
config/               lending policy and price table (illustrative values)
templates/            the contract template
samples/              synthetic certificates with embedded ground truth
tests/
```

[ARCHITECTURE.md](ARCHITECTURE.md) covers the module rules, the data flow of one turn, and how to
add another CRM, model, channel or loan type.

## Roadmap

- Trace every node and model call in Langfuse.
- An evaluation set for intake extraction, next to the one for documents.
- An email channel through n8n, calling the HTTP API.
- Loans secured by real estate: another document schema and policy branch.

## Disclaimer

A portfolio project. All people, cars and documents are synthetic, the policy values and prices
are illustrative, and the contract is a demo draft, not a legal document.
