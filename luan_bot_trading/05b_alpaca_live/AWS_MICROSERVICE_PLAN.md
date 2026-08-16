# FUTURE: AWS Serverless Auto-Execution Microservice

**Status: PROPOSED — revisit after paper trading validates the strategy.**

Deployment filter / automation design for hands-off T+5 exits, delayed
stop-loss, and entry execution without requiring the user's PC to be on
or a manual daily script run.

## Why

The `02_paper_trade.py` script runs manually on the user's machine. This
requires:
1. The PC to be on
2. The user to launch the script each trading day (~3:45 PM ET)

If the user is away, T+5 exits and stop-losses are delayed until they
return. The manual-close reconcile logic handles state drift, but the
*exit* itself still needs human action. A serverless microservice runs
the same logic on a schedule, independent of the user's machine.

## Design

### Components

```
Region: us-east-1                     Every trading day 3:40 PM ET
  ┌──────────────┐  EventBridge cron   ┌──────────────┐
  │   DynamoDB    │◄──── read/write ───│    Lambda     │
  │ (positions    │                    │  02_worker.py │
  │   state)      │                    │   (executor)  │
  └──────────────┘                    └───────┬───────┘
        ▲                                    │ Alpaca REST
        │ plan.json uploaded here             ▼
   (from PC after 01 run)            ┌─────────────────┐
        │                             │    Alpaca        │
        │                             │   paper API      │
   ┌────┴─────┐                       └─────────────────┘
   │  S3 /    │
   │ DynamoDB │  (plan storage)
   └──────────┘
```

### Key decisions

| Concern | Solution |
|---------|----------|
| **Schedule** | AWS EventBridge/CloudWatch cron — weekdays 3:40 PM ET → 1 Lambda invocation |
| **State** | `positions.json` in DynamoDB (atomic, single-writer) |
| **plan.json** | Uploaded by the user's PC via `boto3`/AWS CLI after each `01` run |
| **Alpaca calls** | Plain `requests` to Alpaca REST (NOT alpaca-py) — keeps Lambda layer lean |
| **Trading calendar** | Alpaca `/v2/calendar` API to skip weekends + market holidays (a weekday cron rule doesn't know holidays) |
| **Logic** | Port `02_paper_trade.py`: reconcile → check exits → check delayed stop → place MOC entries/exits |
| **Secrets** | API keys in Lambda env vars / AWS Secrets Manager |
| **IAM** | Lambda role: read/write DynamoDB, invoke nothing else |

### Worker logic (mirrors `02_paper_trade.py`)

```
1. Read state from DynamoDB (positions.json)
2. RECONCILE: sync local state with Alpaca actual positions (manual closes)
3. CHECK active positions:
   a. T+5 exit reached? → place MOC sell
   b. Delayed stop (-10%, day 1+)? → place market sell
4. COUNT free slots
5. ENTRY (P(PEAD)-priority): read plan.json, reserve top-N, enter those due today
6. Write updated state back to DynamoDB
```

### plan.json upload flow (user machine)

After running `01_fetch_and_predict.py`, the user uploads `plan.json`:

```bash
aws s3 cp plan.json s3://pead-bot/plan.json     # or boto3 → DynamoDB
```

The Lambda reads the fresh plan on its next 3:40 PM wake.

## Why serverless > home-machine Task Scheduler

| | Serverless Lambda | Local Task Scheduler |
|---|---|---|
| Runs when PC off | ✅ | ❌ |
| T+5 exit while away | ✅ | ❌ |
| State survives | ✅ (DynamoDB) | local file |
| Cost | ~$0 (free tier) | free |
| Engineering effort | ~1 day | ~15 min |

## Migration path

1. `02_paper_trade.py` already has the full logic (reconcile, stops, exits,
   P(PEAD)-priority entries) → port largely as-is to `02_worker.py`
2. Swap `positions.json` (local file) for DynamoDB read/write
3. Swap alpaca-py for `requests` (or keep alpaca-py in a Lambda layer)
4. Add EventBridge rule + IAM role + Secrets Manager
5. Test worker against paper trading for a week before touching live

## Requirements before building

- Paper trading demonstrates the strategy is profitable/stable
- User confirms they want hands-off automation
- AWS account + IAM knowledge (or willingness to learn during setup)

---

*Documented 2026-08-05. Do NOT build yet — revisit after paper validation.*
