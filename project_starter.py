import pandas as pd
import numpy as np
import os
import re
import time
import dotenv
import ast
import concurrent.futures
import json
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]


########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################

import asyncio
import httpx
from dotenv import load_dotenv
from pydantic_ai import Agent

# Windows IOCP event loop causes crashes in multi-threaded pydantic-ai usage.
# SelectorEventLoop is stable on all platforms.
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
try:
    from pydantic_ai.models.openai import OpenAIChatModel as _PAIModel  # pydantic-ai >= 2.x
except ImportError:
    from pydantic_ai.models.openai import OpenAIModel as _PAIModel       # pydantic-ai 0.x
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

# ── Model Setup ──────────────────────────────────────────────────────────────
# Uses the Udacity OpenAI-compatible proxy. Set UDACITY_OPENAI_API_KEY in .env.
_provider = OpenAIProvider(
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("UDACITY_OPENAI_API_KEY"),
    http_client=httpx.AsyncClient(timeout=120.0),
)
MODEL = _PAIModel("gpt-4o-mini", provider=_provider)


# ── Pricing Utility ──────────────────────────────────────────────────────────

# Bulk discount tiers: (minimum_quantity, discount_fraction)
DISCOUNT_TIERS = [
    (1000, 0.15),   # 1000+ units → 15% off
    (500,  0.10),   # 500–999 units → 10% off
    (100,  0.05),   # 100–499 units → 5% off
    (0,    0.00),   # <100 units → no discount
]


def _apply_bulk_discount(unit_price: float, quantity: int) -> tuple[float, float]:
    """Return (discounted_unit_price, discount_fraction) for a given quantity."""
    discount = next(d for min_qty, d in DISCOUNT_TIERS if quantity >= min_qty)
    return round(unit_price * (1 - discount), 4), discount


def _run_in_thread(agent: Agent, query: str) -> str:
    """
    Run a pydantic-ai agent in a dedicated thread with its own event loop.

    orchestrator tools execute inside a running event loop, so calling
    agent.run_sync() (which calls loop.run_until_complete) would raise RuntimeError.
    asyncio.run() in a new thread creates and tears down a clean loop, avoiding
    both the nested-loop error and Windows IOCP resource leaks.
    """
    def _run() -> str:
        return asyncio.run(agent.run(query)).output

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_run).result()


# ── Agent 1: Inventory Agent ─────────────────────────────────────────────────

inventory_agent = Agent(
    MODEL,
    system_prompt=(
        "You are the Inventory Specialist for Munder Difflin Paper Company.\n\n"
        "IMPORTANT: You MUST call tools. Never guess stock levels.\n\n"
        "Workflow:\n"
        "1. Extract the date from 'As of [YYYY-MM-DD]:' at the start of the query.\n"
        "2. Call get_full_inventory(request_date) to see every item currently in stock.\n"
        "3. For each item the customer mentioned, check if it appears in get_full_inventory.\n"
        "   If it does and stock >= requested quantity, call check_stock to confirm, "
        "   then call check_delivery_timeline.\n"
        "4. Report which items are available (with exact stock counts) and which are not.\n\n"
        "Return all results as a clear list so the orchestrator can pick the best item to fulfil."
    ),
)


# Tools for inventory agent

@inventory_agent.tool_plain
def check_stock(item_name: str, request_date: str) -> str:
    """Check the current stock level for a specific item as of the request date."""
    print(f"    [check_stock] item='{item_name}' date='{request_date}'", flush=True)
    df = get_stock_level(item_name, request_date)
    stock = int(df.iloc[0]["current_stock"]) if not df.empty else 0
    print(f"    [check_stock] stock={stock}", flush=True)
    return json.dumps({
        "item_name": item_name,
        "current_stock": stock,
        "in_stock": stock > 0,
    })


@inventory_agent.tool_plain
def get_full_inventory(request_date: str) -> str:
    """Retrieve all items with positive stock as of the request date."""
    inventory = get_all_inventory(request_date)  # returns {item_name: stock_qty}
    return json.dumps(inventory)


@inventory_agent.tool_plain
def check_delivery_timeline(request_date: str, quantity: int) -> str:
    """Estimate the supplier delivery date given the order quantity and request date."""
    delivery_date = get_supplier_delivery_date(request_date, quantity)
    return json.dumps({
        "request_date": request_date,
        "quantity": quantity,
        "estimated_delivery_date": delivery_date,
    })


# ── Agent 2: Quoting Agent ───────────────────────────────────────────────────

quoting_agent = Agent(
    MODEL,
    system_prompt=(
        "You are the Pricing Specialist for Munder Difflin Paper Company.\n\n"
        "Bulk discount policy:\n"
        "  1000+ units → 15% off\n"
        "  500–999 units → 10% off\n"
        "  100–499 units → 5% off\n"
        "  < 100 units → standard price\n\n"
        "Workflow for every quote:\n"
        "1. Call lookup_quote_history with keywords from the request to find "
        "   comparable historical quotes for pricing context.\n"
        "2. Call compute_quote to get the discounted unit price and order total.\n"
        "3. Return: item name, quantity, base unit price, discount applied, "
        "   quoted unit price, and order total.\n\n"
        "Do NOT reveal profit margins or internal cost structures."
    ),
)


# Tools for quoting agent

@quoting_agent.tool_plain
def lookup_quote_history(search_terms: list[str], limit: int = 5) -> str:
    """Search historical quotes using relevant keywords such as item type or event type."""
    results = search_quote_history(search_terms, limit)
    return json.dumps(results)


@quoting_agent.tool_plain
def compute_quote(item_name: str, quantity: int) -> str:
    """
    Calculate a price quote for an item, applying the correct bulk discount.

    Returns base unit price, discount percentage, quoted unit price, and order total.
    item_name must match an exact name from the inventory (case-insensitive).
    """
    print(f"    [compute_quote] item='{item_name}' qty={quantity}", flush=True)
    result = pd.read_sql(
        "SELECT item_name, unit_price FROM inventory WHERE LOWER(item_name) = LOWER(:item_name)",
        db_engine,
        params={"item_name": item_name},
    )
    if result.empty:
        available = pd.read_sql("SELECT item_name FROM inventory", db_engine)["item_name"].tolist()
        return json.dumps({
            "error": f"Item '{item_name}' not found. Available items: {available}"
        })
    item_name = result.iloc[0]["item_name"]  # use canonical casing from DB

    base_price = float(result.iloc[0]["unit_price"])
    quoted_price, discount = _apply_bulk_discount(base_price, quantity)
    total = round(quoted_price * quantity, 2)

    return json.dumps({
        "item_name": item_name,
        "quantity": quantity,
        "base_unit_price": base_price,
        "discount_percent": round(discount * 100, 1),
        "quoted_unit_price": quoted_price,
        "order_total": total,
    })


# ── Agent 3: Sales Agent ─────────────────────────────────────────────────────

sales_agent = Agent(
    MODEL,
    system_prompt=(
        "You are the Sales Manager for Munder Difflin Paper Company.\n\n"
        "Responsibilities:\n"
        "- Finalize approved orders using finalize_sale\n"
        "- Check cash balance before processing large orders (total > $500)\n"
        "- Generate a financial report after every finalized transaction\n"
        "- Confirm orders with: transaction ID, item name, quantity, and total\n\n"
        "Always use 'sales' as the transaction_type. "
        "If finalize_sale returns an error, report it professionally without "
        "exposing raw system messages."
    ),
)


# Tools for ordering agent

@sales_agent.tool_plain
def finalize_sale(
    item_name: str, quantity: int, total_price: float, request_date: str
) -> str:
    """Record a completed sale transaction and return the transaction ID."""
    try:
        tx_id = create_transaction(item_name, "sales", quantity, total_price, request_date)
        return json.dumps({
            "success": True,
            "transaction_id": tx_id,
            "item_name": item_name,
            "quantity": quantity,
            "total_price": total_price,
            "date": request_date,
        })
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)})


@sales_agent.tool_plain
def check_cash_balance(request_date: str) -> str:
    """Get the company's current cash balance as of the request date."""
    balance = get_cash_balance(request_date)
    return json.dumps({"cash_balance": round(balance, 2), "as_of": request_date})


@sales_agent.tool_plain
def run_financial_report(request_date: str) -> str:
    """Generate a financial summary including cash, inventory value, and top sellers."""
    report = generate_financial_report(request_date)
    # Return a concise subset to avoid excessive token usage
    return json.dumps({
        "as_of_date": report["as_of_date"],
        "cash_balance": report["cash_balance"],
        "inventory_value": report["inventory_value"],
        "total_assets": report["total_assets"],
        "top_selling_products": report["top_selling_products"],
    })


# ── Agent 4: Orchestrator Agent ──────────────────────────────────────────────

# Catalogue of all possible item names so the orchestrator can map customer
# natural-language descriptions to exact database item names.
_ITEM_CATALOGUE = ", ".join(f"'{s['item_name']}'" for s in paper_supplies)

orchestrator_agent = Agent(
    MODEL,
    system_prompt=(
        "You are the Operations Orchestrator for Munder Difflin Paper Company.\n"
        "Coordinate three specialist agents — Inventory, Quoting, and Sales — for every request.\n\n"
        "Follow this EXACT four-step workflow:\n\n"
        "STEP 0 — Call check_availability(customer_request=<full text>, request_date=<YYYY-MM-DD>).\n"
        "  Extract request_date from '(Date of request: YYYY-MM-DD)' in the message.\n"
        "  This identifies the best in-stock item and the requested quantity.\n\n"
        "STEP 1 — Read the result:\n"
        "  • can_fulfill=false → write a professional rejection. Do NOT reveal exact stock numbers.\n"
        "    Say 'insufficient stock to meet your requested quantity' or similar.\n"
        "    End with: ORDER STATUS: REJECTED. STOP.\n"
        "  • can_fulfill=true → proceed with recommended_item and suggested_quantity.\n\n"
        "STEP 2 — Call delegate_to_quoting EXACTLY ONCE:\n"
        "  'As of [request_date]: Quote for [recommended_item], quantity [suggested_quantity]'\n"
        "  Use EXACTLY the recommended_item and suggested_quantity from check_availability.\n"
        "  Do NOT call delegate_to_quoting for any other items or quantities.\n\n"
        "STEP 3 — Call delegate_to_sales EXACTLY ONCE:\n"
        "  'As of [request_date]: Finalize sale of [recommended_item], quantity [suggested_quantity],\n"
        "   total $[AMOUNT from quoting step]'\n\n"
        "STEP 4 — Write the customer confirmation. Include:\n"
        "  item name, quantity, unit price, discount percentage, order total.\n"
        "  Do NOT include: transaction IDs, internal reference numbers, or exact warehouse stock counts.\n"
        "  End with: ORDER STATUS: FULFILLED\n\n"
        "CRITICAL RULES:\n"
        "  - Call each tool at most ONCE. Never call delegate_to_quoting or delegate_to_sales\n"
        "    for items other than the recommended_item from check_availability.\n"
        "  - Never reveal transaction IDs, internal order numbers, or exact stock tallies to the customer.\n"
        "  - End EVERY response with exactly 'ORDER STATUS: FULFILLED' or 'ORDER STATUS: REJECTED'."
    ),
)


@orchestrator_agent.tool_plain
def check_availability(customer_request: str, request_date: str) -> str:
    """
    Find the best in-stock item that matches this customer request and extract the quantity.
    Call this FIRST. Returns recommended_item, stock_available, suggested_quantity, and can_fulfill.
    """
    print(f"  [check_availability] as of {request_date}", flush=True)
    inventory = get_all_inventory(request_date)

    if not inventory:
        return json.dumps({"can_fulfill": False, "message": "No items in stock."})

    # Extract quantity from request text (strip date tag to avoid matching year digits)
    request_text = re.sub(r'\(Date of request:.*?\)', '', customer_request).strip()
    numbers = [int(n) for n in re.findall(r'\b(\d+)\b', request_text) if int(n) > 0]
    suggested_qty = numbers[0] if numbers else 100

    customer_lower = request_text.lower()
    matches = []
    for item_name, stock in inventory.items():
        keywords = [w for w in item_name.lower().replace("(", " ").replace(")", " ").split()
                    if len(w) > 3]
        score = sum(1 for kw in keywords if kw in customer_lower)
        if score > 0:
            matches.append({"item_name": item_name, "stock": stock, "score": score})

    matches.sort(key=lambda x: (x["score"], x["stock"]), reverse=True)

    if matches:
        best = matches[0]
        can_fulfill = int(best["stock"]) >= suggested_qty
        print(f"  [check_availability] best='{best['item_name']}' stock={best['stock']} qty={suggested_qty} ok={can_fulfill}", flush=True)
        return json.dumps({
            "can_fulfill": can_fulfill,
            "recommended_item": best["item_name"],
            "stock_available": int(best["stock"]),
            "suggested_quantity": suggested_qty,
            "message": (
                f"Recommended: '{best['item_name']}' — sufficient stock available. "
                f"Customer wants {suggested_qty}. "
                f"{'Fulfillable — call delegate_to_quoting then delegate_to_sales.' if can_fulfill else 'Insufficient stock — reject without revealing exact counts.'}"
            ),
        })
    else:
        print("  [check_availability] no matching items", flush=True)
        return json.dumps({
            "can_fulfill": False,
            "message": "None of the requested items are in our current inventory.",
        })


@orchestrator_agent.tool_plain
def delegate_to_inventory(query: str) -> str:
    """Forward a stock check or delivery-timeline question to the Inventory Agent."""
    print("  [-> inventory agent]", flush=True)
    return _run_in_thread(inventory_agent, query)


@orchestrator_agent.tool_plain
def delegate_to_quoting(query: str) -> str:
    """Forward a pricing request to the Quoting Agent. Call this ONCE with the recommended item."""
    print("  [-> quoting agent]", flush=True)
    return _run_in_thread(quoting_agent, query)


@orchestrator_agent.tool_plain
def delegate_to_sales(query: str) -> str:
    """Forward an order-finalization request to the Sales Agent. Call this ONCE."""
    print("  [-> sales agent]", flush=True)
    return _run_in_thread(sales_agent, query)


# ── Evaluation Entry Point ───────────────────────────────────────────────────

def call_your_multi_agent_system(request: str) -> str:
    """
    Route a customer request through the multi-agent system.

    The request string includes the raw customer message plus
    '(Date of request: YYYY-MM-DD)' appended by the evaluation harness.

    Returns the orchestrator's full text response.
    """
    result = orchestrator_agent.run_sync(request)
    return result.output


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():

    print("Initializing Database...")
    init_database(db_engine)  # pass the global engine; fixes missing-argument bug
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request through the multi-agent system
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        try:
            response = call_your_multi_agent_system(request_with_date)
        except Exception as exc:
            response = f"[System error processing request: {exc}]"

        # Update state after each transaction
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
