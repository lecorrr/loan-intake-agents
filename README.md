# Loan intake agents

A LangGraph agent team that takes a car-secured loan application from a Telegram chat to a CRM
deal and a draft contract, then hands the case to a manager, who decides with a button press.
The model runs locally in Ollama. All data is synthetic.

```
client ── Telegram / HTTP ──► service ──► LangGraph team ──────────────► EspoCRM
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
- Writes the contact and the deal to a self-hosted EspoCRM. Repeated writes update the same deal
  instead of creating new ones, and a returning client keeps one contact; the deal moves through
  stages and collects comments, and the manager's card links to it.
- Drafts the loan agreement from a template when nothing is flagged.
- Sends the manager a card with personal data masked, plus the draft. The manager's button
  press resumes the case. The agents never approve, reject, or tell the client a decision.
- Traces every turn in a self-hosted Langfuse: each step of the team and each model call, with
  timings and token counts, and with personal data masked before it leaves the app.

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
`tests/test_postgres.py`), and the live EspoCRM tests only when one is given (see
`tests/test_espocrm_live.py`).

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
EspoCRM, Langfuse, Postgres and Telegram; it warns when the model is a reasoning build.
`eval-documents` reads the synthetic certificates in `samples/`, scores every field against the
ground truth embedded in each PNG, and prints what was read next to what was expected; `--runs 3`
reads each sample three times. The first model call after a start also loads the model into VRAM, so it is
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

### Measured on the synthetic samples

With `qwen3-vl:8b-instruct` on the same machine, `eval-documents --runs 3` read every field right on
the four regular samples in all three runs (84 of 84), and every field but the car model on the hard
sample: 102 of 105 overall. A new certificate takes about 3 s (2 s when the same image is read
again), a client message about 1.5 s. Ollama unloads the model after 5 idle minutes, so the first
call after a pause takes a few seconds longer.

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
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `message.chat.id`. If `result` is
   empty, send the bot any message and reload. Do this while the app is not running, because a
   running bot takes the updates itself.
   - For a solo demo, use your own id: you are then client and manager in the same chat.
   - For a managers' group, add the bot to the group, send `/start@<your_bot>` there, and use the
     group's negative id.
3. Put the id in `MANAGER_CHAT_ID`. `loan-agents doctor` confirms that the bot can reach that chat.
4. Run `.venv/bin/loan-agents serve` and write to the bot.

Clients are served in private chats only, and only buttons pressed in the manager chat count.
`/new` starts the conversation over. Messages from one chat are handled one at a time, in the
order they arrived; on start, the bot also answers what was sent while it was off.

## Docker compose: Postgres, EspoCRM and the app

Compose runs Postgres on 127.0.0.1:5433, EspoCRM on http://127.0.0.1:8090 with its own MariaDB,
and the app on 127.0.0.1:8000, which also polls Telegram if a token is set. Ollama stays on the
host, where the GPU is. The app container uses the host network, so it reaches Ollama on
127.0.0.1:11434 and uses the same URLs as a native run; this needs Docker on Linux.

Copy `.env.example` to `.env` if there is none yet. In it, set
`ESPOCRM_URL=http://127.0.0.1:8090` and choose `ESPOCRM_ADMIN_PASSWORD`. Then start EspoCRM on its
own; the first start installs it and takes a minute or two:

```
docker compose up -d --wait espocrm
```

Prepare it for the app. This signs in as the administrator and creates what the app needs:

```
docker compose run --rm app loan-agents crm-setup
```

It prints a line `ESPOCRM_API_KEY=...`: put it in `.env` in place of the empty one. Run natively,
`.venv/bin/loan-agents crm-setup --write-env` puts it there itself. Then start everything and
check:

```
docker compose up -d --build
docker compose exec app loan-agents doctor
```

Logs: `docker compose logs -f app` (in UTC). Cases live in Postgres, so they survive a restart:
after `docker compose restart app`, the manager's buttons on a waiting case still work.
Langfuse starts only with `--profile tracing`; see [Tracing](#tracing-with-langfuse).

## EspoCRM

[EspoCRM](https://www.espocrm.com/) is an open-source CRM with a REST API; compose runs version
10.0.11 with a Ukrainian interface. Open http://127.0.0.1:8090 and sign in as the administrator
to see deals (Угоди) with their contacts, stages and comments. The card the manager gets in
Telegram links straight to the deal; the link works on the machine that runs EspoCRM.

`crm-setup` can run again at any time; it repairs what is missing and changes nothing else:

- adds UAH to the currencies; on a CRM without deals it also becomes the main currency;
- adds the deal field `cLoanCaseId` (Номер заявки), read-only after create, and shows it on the
  deal card. The app finds a case's deal by it, so a retried step never makes a second deal;
- creates the role "Loan intake bot": create, read and edit contacts and deals and comment on
  them, nothing else, no deleting;
- creates the API user `loan-intake-bot` with that role and prints its key. The running app signs
  in with this key only, never as the administrator.

How the app writes:

| Case step | Deal stage (`ESPOCRM_STAGE_*`) |
|---|---|
| Collecting documents again | Prospecting |
| Complete, waiting for the manager | Qualification |
| Approved by the manager; signing still ahead | Negotiation |
| Rejected | Closed Lost |

- A client is matched on phone and name together. A returning client keeps one contact; two
  people who share only a name, or only a phone, get separate contacts. A manager can merge
  contacts in EspoCRM.
- The summary for the manager goes into the deal's description; the draft contract and the
  decision go into its stream as comments.
- Reads and updates, which change nothing more when sent twice, are retried on a network error
  or a 502/503/504. A create is retried only when it never reached EspoCRM.

The administrator's password is applied only when EspoCRM installs itself on the first start;
changing it in `.env` later does not change it in EspoCRM. To start the CRM over, run
`docker compose down` and `docker volume rm` the four `loan-intake-agents_espocrm-*` volumes.

`tests/test_espocrm_live.py` runs `crm-setup` and a whole deal against a real EspoCRM when
`LOAN_AGENTS_TEST_ESPOCRM_URL` and `LOAN_AGENTS_TEST_ESPOCRM_PASSWORD` are set; the rest of the
suite uses an in-memory EspoCRM built from a real one's answers.

## Tracing with Langfuse

[Langfuse](https://langfuse.com/) shows what the agents did: one trace per turn of a case, named
`start`, `message`, `document` or `decision`, and the turns of one case grouped into a session
named after the case id. A trace holds every step of the team with its input, output and timing,
and every model call with the model's name and token counts. Without keys nothing is traced.

Compose runs Langfuse 4 under the `tracing` profile: the web interface on http://127.0.0.1:3000,
plus its worker, Postgres, ClickHouse, MinIO and Redis, none of them published. Make the keys
and the administrator's password first; this fills in only the values `.env` lacks:

```
.venv/bin/loan-agents tracing-keys --write-env
```

Then start Langfuse. The first start downloads the images and prepares its databases, which
takes a few minutes:

```
docker compose --profile tracing up -d --wait langfuse-web
```

When it starts, Langfuse creates what is missing of these: the organization, the project
`loan-intake-agents` with these keys, and the user `LANGFUSE_ADMIN_EMAIL`
(`admin@loan-intake.local` unless set) with `LANGFUSE_ADMIN_PASSWORD` from `.env`. Restart the
app so it reads the keys (`docker compose up -d app`, or start `serve` again); `doctor` checks
them. In Langfuse, open the project, then Tracing for single turns or Sessions for whole cases.

What leaves the app is masked first (`core/masking.py`):

- names become initials wherever they appear, once a name field has held them: Т. Ш.;
- phones, VINs and plates are masked as on the manager's card, by field and by their shape in
  any text;
- the client's own messages are replaced by their length, `[client's message, 25 characters]`,
  because nothing marks a name in free text until an agent has read it;
- photos are never sent: the picture in a model call shows as `[image/jpeg, 98 KB, not sent]`,
  and the SDK's own upload of pictures is turned off.

The app does not wait for Langfuse. If it is down, the traces of that time are dropped after a
few seconds of retries in the background, and the app answers as usual.

Langfuse never changes what it has: a new password or secret key in `.env` is not applied to
an existing user or key. To start Langfuse over, run `docker compose --profile tracing down`
and `docker volume rm` the five `loan-intake-agents_langfuse-*` volumes. If port 3000 is taken,
change the published port and `NEXTAUTH_URL` in `docker-compose.yml` and `LANGFUSE_BASE_URL` in
`.env`. Langfuse Cloud works too: set `LANGFUSE_BASE_URL` to its address and the project's keys
from its settings; the masked traces then leave the machine.

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
  Traces mask names, phones, VINs and plates, leave out the client's own words and carry no
  photos.
  Full values are kept only where the work needs them: the CRM, the uploaded photo, and the draft
  contract (stored under `data/uploads/` and attached to the manager's card).

## Layout

```
src/loan_agents/
  contracts.py        shared types, units and formats
  core/               pure logic: policy, routing, validation, masking, texts, contract
  agents/graph.py     the LangGraph team
  service.py          one case at a time: locks, uploads, pending decisions
  adapters/           LangChain/Ollama, EspoCRM, Telegram, Postgres, files, CSV prices, Langfuse
  fakes/              stand-ins for tests and --fake
  wiring.py           picks the adapters from the settings
  api.py, cli.py, telegram_bot.py   entry points
config/               lending policy and price table (illustrative values)
templates/            the contract template
samples/              synthetic certificates with embedded ground truth
tests/
```

[ARCHITECTURE.md](ARCHITECTURE.md) covers the module rules, the data flow of one turn, and how to
add another CRM, model, channel or loan type.

## Roadmap

- An evaluation set for intake extraction, next to the one for documents.
- An email channel through n8n, calling the HTTP API.
- Loans secured by real estate: another document schema and policy branch.

## Disclaimer

A portfolio project. All people, cars and documents are synthetic, the policy values and prices
are illustrative, and the contract is a demo draft, not a legal document.
