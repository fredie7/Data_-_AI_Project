import sys
from pathlib import Path
import pandas as pd

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "silver"))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / ""))

# Load transformation functions at module level
from transform_transactiions_data import transform_transactions_data
from transform_customers_data import transform_customers_data

from utils.helper_functions import logger

# Define output path at module level
output_path = (
    Path(__file__).resolve().parent.parent.parent
    / "processed_data"
    / "fact_transactions.csv"
)



def build_fact_transactions(
    transform_transactions_fn,
    transform_customers_fn,
    output_path: Path,
):
    # Load transformed data
    transactions_df = transform_transactions_fn()
    customers_df = transform_customers_fn()

    # Import dimension tables (assumed to be available in the current namespace)
    from dim_customers import dim_customer
    from dim_currency import dim_currency
    from dim_category import dim_category
    from dim_dates import dim_date

    # Ensure timestamp is datetime
    # transactions_df["timestamp"] = pd.to_datetime(transactions_df["timestamp"])

    # Rename columns for clarity
    transactions_df = transactions_df.rename(columns={
        "timestamp": "transaction_timestamp",
        "amount_eur": "transaction_amount_eur",
        "exchange_rate": "current_exchange_rate",
    })

    # Add a transaction date column
    transactions_df["transaction_date"] = transactions_df["transaction_timestamp"].dt.date

    # High‑value transaction flag (> 500 EUR)
    transactions_df["is_high_value_transaction"] = (
        transactions_df["transaction_amount_eur"] > 500
    ).astype(int)

    # Merge dimension keys into the fact table
    transactions_df = (
        transactions_df
        .merge(
            dim_customer[["customer_id", "customer_key"]],
            on="customer_id",
            how="left",
        )
        .merge(
            dim_currency[["transaction_currency", "currency_key"]],
            on="transaction_currency",
            how="left",
        )
        .merge(
            dim_category[["category", "category_key"]],
            on="category",
            how="left",
        )
        .merge(
            dim_date[["date", "date_key"]],
            left_on=transactions_df["transaction_timestamp"].dt.date,
            right_on="date",
            how="left",
            validate="many_to_one",
       )
    )

    # Final fact table
    fact_transactions = transactions_df[[
        "transaction_id",
        "transaction_key",
        "customer_id",
        "customer_key",
        "currency_key",
        "category_key",
        "date_key",
        "transaction_timestamp",
        "transaction_amount_eur",
        "current_exchange_rate",
        "is_high_value_transaction",
    ]]

    # Store data as CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fact_transactions.to_csv(output_path, index=False)


    logger.info(f"Fact table 'fact_transactions' built successfully with shape {fact_transactions.shape} and saved to {output_path}.")
    return fact_transactions


# Usage & Quality checks
fact_transactions = build_fact_transactions(
    transform_transactions_data,
    transform_customers_data,
    output_path,
)
# print(fact_transactions.head())
# print(fact_transactions.columns.tolist())
# print(len(fact_transactions))