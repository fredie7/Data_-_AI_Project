import sys
from pathlib import Path
import pandas as pd

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "silver"))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / ""))

# Load transformation functions at module level
# from data_warehouse.etl.gold import dim_category, dim_currency, dim_customers, dim_dates
# from data_warehouse.etl.gold import dim_dates
from transform_transactiions_data import transform_transactions_data
from transform_customers_data import transform_customers_data

from utils.helper_functions import logger

# Define output path at module level
output_path = (
    Path(__file__).resolve().parent.parent.parent
    / "processed_data"
    / "fact_transactions.csv"
)

# Import dimension tables (assumed to be available in the current namespace)
from dim_customers import dim_customer
from dim_currency import dim_currency
from dim_category import dim_category
from dim_dates import dim_date





def build_fact_transactions(
    transform_transactions_fn,
    transform_customers_fn,
    dim_customer: pd.DataFrame,
    dim_currency: pd.DataFrame,
    dim_category: pd.DataFrame,
    dim_date: pd.DataFrame,
    output_path: Path,
):

    # -----------------------------
    #  Load transformed data
    # -----------------------------
    transactions_df = transform_transactions_fn()
    customers_df = transform_customers_fn()  # (kept if needed upstream)

    # -----------------------------
    #  Standardize & Rename
    # -----------------------------
    transactions_df = transactions_df.rename(columns={
        "timestamp": "transaction_timestamp",
        "amount_eur": "transaction_amount_eur",
        "exchange_rate": "current_exchange_rate",
    })

    transactions_df["transaction_timestamp"] = pd.to_datetime(
        transactions_df["transaction_timestamp"]
    )

    transactions_df["transaction_date"] = (
        transactions_df["transaction_timestamp"].dt.date
    )

    transactions_df["is_high_value_transaction"] = (
        transactions_df["transaction_amount_eur"] > 500
    ).astype(int)

    # -----------------------------
    #  Resolve SCD Type-2 Customer Dimension
    # -----------------------------

    # Ensure date types
    dim_customer["effective_from"] = pd.to_datetime(dim_customer["effective_from"])
    dim_customer["effective_to"] = pd.to_datetime(
        dim_customer["effective_to"]
    ).fillna(pd.Timestamp("2099-12-31"))

    transactions_df["transaction_date"] = pd.to_datetime(
        transactions_df["transaction_date"]
    )

    # Merge on business key first
    transactions_df = transactions_df.merge(
        dim_customer,
        on="customer_id",
        how="left",
    )

    # Filter correct historical record
    transactions_df = transactions_df[
        (transactions_df["transaction_date"] >= transactions_df["effective_from"]) &
        (transactions_df["transaction_date"] <= transactions_df["effective_to"])
    ]

    # Enforce 1 transaction → 1 customer version
    assert transactions_df.groupby("transaction_id").size().max() == 1, \
        "SCD2 resolution failed: duplicate customer versions detected."

    # Drop SCD tracking columns & natural key
    transactions_df = transactions_df.drop(
        columns=["customer_id", "effective_from", "effective_to"]
    )

    # -----------------------------
    #  Merge Static Dimensions
    # -----------------------------

    # Currency
    transactions_df = transactions_df.merge(
        dim_currency[["transaction_currency", "currency_key"]],
        on="transaction_currency",
        how="left",
        validate="many_to_one",
    )

    # Category
    transactions_df = transactions_df.merge(
        dim_category[["category", "category_key"]],
        on="category",
        how="left",
        validate="many_to_one",
    )

    # Date
    dim_date["date"] = pd.to_datetime(dim_date["date"])

    transactions_df = transactions_df.merge(
        dim_date[["date", "date_key"]],
        left_on="transaction_date",
        right_on="date",
        how="left",
        validate="many_to_one",
    )

    # -----------------------------
    #  Final Fact Table
    # -----------------------------
    fact_transactions = transactions_df[[
        "transaction_id",
        "transaction_key",
        "customer_key",   
        "currency_key",
        "category_key",
        "date_key",
        "transaction_timestamp",
        "transaction_amount_eur",
        "current_exchange_rate",
        "is_high_value_transaction",
    ]]

    # Final integrity checks
    assert fact_transactions["customer_key"].isnull().sum() == 0, \
        "Missing customer_key detected."
    assert fact_transactions["currency_key"].isnull().sum() == 0, \
        "Missing currency_key detected."
    assert fact_transactions["category_key"].isnull().sum() == 0, \
        "Missing category_key detected."
    assert fact_transactions["date_key"].isnull().sum() == 0, \
        "Missing date_key detected."

    # -----------------------------
    #  Persist
    # -----------------------------
    fact_transactions.to_csv(output_path, index=False)

    logger.info(
        f"Fact table 'fact_transactions' built successfully "
        f"with shape {fact_transactions.shape} and saved to {output_path}."
    )

    return fact_transactions


# -----------------------------
# Usage
# -----------------------------
fact_transactions = build_fact_transactions(
    transform_transactions_data,
    transform_customers_data,
    dim_customer,
    dim_currency,
    dim_category,
    dim_date,
    Path("fact_transactions.csv"),
)
fact_transactions.head(3)