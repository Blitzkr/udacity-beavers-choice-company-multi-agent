# Reflection Report — Munder Difflin Paper Company Multi-Agent System

## 1. Agent Workflow Diagram — Architecture Decisions

### Agent Roles

The system uses four pydantic-ai agents connected in an orchestrator–worker pattern:

| Agent | Role |
|---|---|
| **Orchestrator** | Entry point. Coordinates the full request lifecycle using four tools. |
| **Inventory Agent** | Checks specific item stock levels and estimates supplier delivery timelines. |
| **Quoting Agent** | Retrieves historical quote context and computes bulk-discounted prices. |
| **Sales Agent** | Validates cash balance, records the sale transaction, and generates a financial report. |

### Workflow (per request)

**Step 0 — `check_availability` (Python tool on Orchestrator)**
Before delegating to any sub-agent, the orchestrator calls a deterministic Python tool that:
1. Queries the live database for all items with positive stock as of the request date (`get_all_inventory`).
2. Scores each in-stock item by how many of its name keywords appear in the customer's message.
3. Extracts the requested quantity from the message text using a regex.
4. Pre-computes a `can_fulfill` boolean (`stock_available >= suggested_quantity`).

**Why this tool exists:** Early versions delegated item selection entirely to gpt-4o-mini. The LLM consistently selected the first item the customer mentioned (often out of stock) rather than available alternatives, and called the quoting sub-agent multiple times in parallel for different items — exhausting the API proxy's connection pool and producing timeouts. Moving item resolution into Python eliminates the fan-out and gives the LLM a single unambiguous `recommended_item` to act on.

**Step 1 (optional) — `delegate_to_inventory`**
If the customer needs a delivery estimate, the orchestrator delegates to the Inventory Agent, which calls `check_delivery_timeline → get_supplier_delivery_date`.

**Step 2 — `delegate_to_quoting`**
The orchestrator calls the Quoting Agent with the exact `recommended_item` and `suggested_quantity`. The agent calls `lookup_quote_history` for historical pricing context, then `compute_quote` to apply the bulk-discount schedule (<100 units = 0%, 100–499 = 5%, 500–999 = 10%, ≥1000 = 15%).

**Step 3 — `delegate_to_sales`**
The orchestrator passes the quoted total to the Sales Agent, which calls `finalize_sale → create_transaction` to record a `sales` entry in the database, then `run_financial_report` to update the company's financial state.

### Key Implementation Decisions

- **Windows asyncio fix:** `asyncio.WindowsSelectorEventLoopPolicy()` is set at startup. The default ProactorEventLoop (IOCP) on Windows crashes when pydantic-ai runs inside a `ThreadPoolExecutor`. Sub-agents each use `asyncio.run()` in a dedicated thread (`_run_in_thread`) to avoid the nested event-loop error.
- **pydantic-ai version compatibility:** A `try/except` import handles both pydantic-ai ≥ 2.x (`OpenAIChatModel` + `OpenAIProvider`) and the legacy 0.x (`OpenAIModel`), since the grading environment may differ from the development environment.
- **Case-insensitive DB lookup:** `compute_quote` uses `LOWER(item_name) = LOWER(:item_name)` so that minor casing differences from the LLM don't produce "item not found" errors.

---

## 2. Evaluation Results — Strengths

The system processed all 20 requests in `test_results.csv`. **13 of 20 orders were fulfilled (65%).**

### Fulfilled Orders — Highlights

**Request 1 (2025-04-01) — Colored paper, 200 sheets**
- `check_availability` matched "Colored paper" (788 units, score = 2 keywords) for a request asking for "A4 glossy paper, heavy cardstock, and colored paper."
- Quoting agent applied the 5% bulk discount (100–499 tier): $0.10 → $0.095/sheet, total $19.00.
- Demonstrates the system's ability to select an available substitute when the customer's primary items are out of stock.

**Request 2 (2025-04-03) — Large poster paper, 500 units**
- Customer request described "colorful poster paper." `check_availability` scored "Large poster paper (24x36 inches)" at score = 2 ("poster" + "paper"), stock = 699 units.
- 10% bulk discount applied (500–999 tier): $1.00 → $0.90/unit, total $450.00.
- Shows the keyword matching bridging the gap between informal customer language and exact catalogue names.

**Request 13 (2025-04-08) — 100 lb cover stock, 500 sheets**
- Request mentioned "A4 printing paper and cardstock." The system matched "100 lb cover stock" (636 in stock), applying a 10% discount for a $225.00 total.
- Demonstrates that specialty paper items can be fulfilled when standard items are exhausted.

**Cash balance movement:**
- Starting cash: $45,059.70 | Final cash: $46,139.20 — a net gain of $1,079.50 across the 13 fulfilled orders, reflecting successful revenue collection after inventory cost.

### Rejected Orders — Appropriate Rejections

**Request 7 (2025-04-07)** — Customer requested 500 sheets of glossy poster boards (24×36 inches). By this date, previous fulfillments had reduced "Large poster paper" stock below 500 units. The system correctly rejected with "insufficient stock to meet your requested quantity."

**Request 10 (2025-04-08)** — Customer requested 500 sheets of high-quality glossy paper. After Request 8 consumed 500 units of Glossy paper, only 87 remained — correctly rejected.

These rejections demonstrate that the database-driven stock tracking accumulates correctly across requests: sales from earlier requests reduce available inventory for later ones.

---

## 3. Suggested Improvements

### Improvement 1 — Partial Fulfillment with Customer Consent

Currently the system rejects any order where `stock_available < suggested_quantity`. For example, Request 10 was rejected because only 87 of the requested 500 glossy paper sheets were available.

**Proposed change:** When stock partially covers the request, the orchestrator could offer the customer a partial shipment:
> "We have 87 sheets of Glossy paper available. Would you like us to fulfil a partial order of 87 sheets now and backorder the remaining 413?"

This would require: (a) a `create_backorder` transaction type, (b) the orchestrator's `check_availability` tool returning a `partial_fulfillment` flag alongside `can_fulfill`, and (c) an updated orchestrator prompt that distinguishes full, partial, and zero fulfillment paths.

### Improvement 2 — Dynamic Synonym Mapping for Better Item Matching

The current keyword-scoring approach uses exact substring matching (e.g., "poster" in "Large poster paper"). This fails for synonyms: a customer asking for "A4 white paper" gets matched to a high-stock paper item with the most overlapping keywords, not necessarily the closest semantic match.

**Proposed change:** Add a pre-computed synonym table (e.g., `"printer paper" → "A4 paper"`, `"construction paper" → "Colored paper"`, `"cover stock" → "Cardstock"`) and apply it before keyword scoring. Alternatively, use a lightweight embedding similarity check (e.g., cosine similarity on sentence-transformer embeddings) to rank candidate items by semantic closeness rather than keyword overlap. This would reduce false matches like Banner paper being recommended when a customer clearly wants printer paper.
