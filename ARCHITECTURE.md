# Architecture

A client applies for a car-secured loan in Telegram or through the HTTP API. A team of agents
collects the details, reads the vehicle registration certificate, values the car against the
lending policy, writes the case into the CRM, drafts the loan agreement and hands the complete
file to a manager. Only the manager decides.

Size: about 5,900 lines of Python in one package, plus 3,700 lines of tests, an in-memory
EspoCRM among them.

## Modules

| Module | Owns | May import |
|---|---|---|
| `contracts.py` | The shared vocabulary: request, certificate, valuation, decision, turn result. Money in whole UAH, terms in months, years as calendar years, phones as `+380XXXXXXXXX`. | pydantic |
| `core/` | Pure logic: policy and valuation maths, routing, intake validation, document cross-checks, masking, all Ukrainian texts, contract rendering. No I/O, no clock: the current date is passed in. | contracts, jinja2 |
| `ports.py` | Interfaces to the outside world: `Extractor` (the model), `CrmGateway`, `PriceBook`, `DocumentStore`, `AuditLog`, `Messenger`, `Tracer`. | contracts |
| `agents/` | The LangGraph team: the supervisor (a routing function) and its nodes. Calls ports only. | core, ports, contracts, langgraph |
| `service.py` | One case at a time: per-case locks, upload validation, pending manager decisions, notifying client and manager, naming each turn for the tracer. | agents, ports, core, contracts |
| `adapters/` | One file per external thing: LangChain/Ollama, EspoCRM (with its one-time setup), Telegram, Postgres audit log, files, CSV prices, log-only messenger, Langfuse tracing. | ports, contracts, core |
| `fakes/` | Stand-ins for every port, used by tests and `--fake` runs. | ports, contracts, core |
| `wiring.py` | The only place where adapters are chosen and connected. | everything above |
| `api.py`, `cli.py` | Entry points: the HTTP API (which also runs the bot) and the command line. | wiring, service, contracts, telegram_bot |
| `telegram_bot.py` | Telegram handlers: client messages and photos in private chats, the manager's decision buttons; one chat's updates in arrival order; the polling lifecycle. | service, core, contracts, adapters/telegram |
| `samples.py` | Synthetic certificate images with their ground truth embedded. | pillow |
| `evaluation/` | `loan-agents eval-cases`: reads `evals/cases.yaml`, plays each case through the service with the model, an in-memory CRM and case store, scores the outcome, the guardrails and every field, and reports. | service, wiring, core, contracts, fakes, samples, adapters/llm_langchain |

`tests/test_settings_and_boundaries.py` enforces the import rules:

- `core` imports nothing from `agents`, `service`, `adapters`, `fakes` or `wiring`, and no I/O
  library (httpx, aiogram, psycopg, langchain, langgraph, langfuse, opentelemetry, fastapi).
- `agents` never imports `adapters` or `fakes`.
- Adapters never import each other, `agents` or `service`.

## One client turn

```
Telegram / HTTP API
  -> service.handle_text / handle_document     lock the case, validate input, store the upload
       manager decision pending?  -> reply "waiting", stop
  -> graph.ainvoke(input, thread_id=case_id)   one Langfuse trace when tracing is on
       supervisor picks the next step from the case state, nodes run, until:
         END        the agents need something from the client
         interrupt  the case file is complete and waits for a manager
  -> Messenger.send_to_client(new replies)
  -> Messenger.send_approval_request(card)      only on interrupt
```

A manager decision (Telegram button or `POST /api/cases/{id}/decision`) resumes the graph with
`Command(resume=decision)`. `finalize` tells the client and moves the CRM deal to its stage.

## The agent team

| Node | Model? | Does | Never does |
|---|---|---|---|
| supervisor | no | Picks the next step from the case state (`core/routing.py`) | Decide anything about the loan |
| intake | text | Extracts name, phone, amount, term and car from the client's words; answers stock questions from approved texts | Invent values; promise terms |
| ask / ask_document | no | Asks for the first missing item or for the certificate photo | |
| documents | vision | Reads the certificate into fields, validates VIN and plate, cross-checks with the application | Accept an unreadable photo |
| valuation | no | Prices the car, applies the policy, computes the payment, raises flags | Approve or decline |
| crm | no | Find-or-create the deal by its case field and the contact by phone and name, keeps the contact's name and phone current, writes the summary | Delete or touch other deals |
| contract | no | Fills the agreement template, only when no policy or document flags | Send it to the client |
| submit + approval | no | Tells the client the file went to a manager, then waits | |
| finalize | no | Relays the manager's decision, moves the deal stage | Run without a manager decision |

The supervisor is code on purpose. In a lender, routing must be predictable and auditable, so the
models do only what code can't: understand free text and read a photo.

## Autonomy limits

- Agents may ask questions, read documents, write the contact, deal and comments in the CRM, and
  draft the contract.
- Agents may not approve or decline, tell the client a decision, quote final terms or move money.
  Only `finalize` speaks about decisions, and it runs only after a manager's resume.
- Model output never acts directly: every extraction is validated (types, ranges, formats,
  cross-checks) before it reaches the case state.
- Car names are handled differently on the two sides of the cross-check (`core/vehicles.py`). A
  make or model the client gives is matched to a known name, also when the model spelled it the
  way it sounds (KASHKAI for QASHQAI). The certificate is taken as printed: a misread there is
  never guessed into a known name, so it reaches the manager as a mismatch.
- With the default local model, no client data goes to a model provider.
- Logs and the audit trail carry ids, field names and numbers, not personal values. Model errors
  that quote the answer are logged only at DEBUG. The API's case view masks names, phones, VINs and
  plates; the manager's card masks phones, VINs and plates. Full values live only where the work
  needs them: the CRM, the uploaded photo and the contract draft the manager receives.
- Traces pass through `core.masking.mask_personal_data` in the Langfuse SDK's mask hook, so
  every input, output and piece of metadata is masked before it is queued. Names are masked
  wherever they appear once a name field in the same turn has held them, the client's own
  messages are replaced by their length, and pictures by a note. Error texts bypass that hook,
  so a second hook masks them, and replaces the text of LangGraph's interrupt, which holds the
  whole manager card.

## State

- Case state: LangGraph checkpoints, one thread per case. `InMemorySaver` without `DATABASE_URL`,
  `AsyncPostgresSaver` with it.
- Audit trail: Postgres table `audit_events`, or memory.
- Uploaded documents and contract drafts: `data/uploads/<case_id>/`, outside the code tree and never
  in a release.
- Lending policy and price table: `config/`, validated at load. Unknown keys are errors.
- Evaluation cases: `evals/cases.yaml`, validated at load; reports go to `data/evals/`.
- Traces: Langfuse's own Postgres, ClickHouse and MinIO volumes, when the `tracing` profile runs.

## Extension recipes

- Another CRM: add `adapters/<crm>.py` implementing `CrmGateway`, choose it in `wiring.py`.
- Another model or provider: install its LangChain package (e.g. `langchain-openai`) and set
  `LLM_MODEL`, e.g. `openai:gpt-5-mini`. Client data then leaves the machine.
- Another channel (email through n8n): call the HTTP API, or write an adapter that calls `service`.
- Another tracing backend: an adapter whose `run_config` returns that backend's LangChain
  callbacks, chosen in `wiring.py`.
- Another evaluation case: add it to `evals/cases.yaml`; `eval-cases --fake` (and the tests)
  check that the case is consistent before the model ever sees it.
- Real-estate loans: a document schema and reader method, rules in `core/policy.py`, a branch in
  `core/routing.py`.
