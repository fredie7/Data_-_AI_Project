import sys
from pathlib import Path
import pandas as pd

# Add parent directory to sys.path at module level
sys.path.append(str(Path(__file__).resolve().parent.parent / "silver"))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / ""))

from transform_transactiions_data import transform_transactions_data

from utils.helper_functions import logger

# Output path for the dimension table
output_path = (
    Path(__file__).resolve().parent.parent.parent
    / "processed_data"
    / "dim_dates.csv"
)


def build_dim_date(transform_transactions_fn):
    # Load and prepare transactions
    transactions_df = transform_transactions_fn()
    
    # If there are no transactions, return an empty DataFrame with the expected columns
    if transactions_df.empty:
        return pd.DataFrame(
            columns=[
                "date_key",
                "date",
                "transaction_day",
                "transaction_month",
                "transaction_year",
                "transaction_weekday",
                "transaction_timestamp",
            ]
        )

    # Extract unique dates from transaction_timestamp
    unique_dates = (
        transactions_df["transaction_timestamp"]
        .dt.date
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    # Create dimension table for date for time series analysis
    logger.info("Building dim_date dimension table...")

    dim_date = pd.DataFrame({"date": unique_dates})
    dim_date["date_key"] = range(1, len(dim_date) + 1)
    dim_date["transaction_day"] = pd.to_datetime(dim_date["date"]).dt.day
    dim_date["transaction_month"] = pd.to_datetime(dim_date["date"]).dt.month
    dim_date["transaction_month_name"] = pd.to_datetime(dim_date["date"]).dt.strftime("%b")
    dim_date["transaction_year"] = pd.to_datetime(dim_date["date"]).dt.year
    dim_date["transaction_weekday"] = pd.to_datetime(dim_date["date"]).dt.strftime("%a")

    # Propagate transaction timestamp
    dim_date["transaction_timestamp"] = transactions_df["transaction_timestamp"]
    # dim_date["transaction_key"] = transactions_df["transaction_key"]

    # Final column order
    dim_date = dim_date[
        [
            "date_key",
            "date",
            "transaction_day",
            "transaction_month",
            "transaction_month_name",
            "transaction_year",
            "transaction_weekday",
            "transaction_timestamp"
        ]
    ]

    # Save dimension table to CSV
    logger.info(f"Saving dim_date dimension table to {output_path}...")
    dim_date.to_csv(output_path, index=False)

    return dim_date


# Build the dimension table and save it to CSV

dim_date = build_dim_date(transform_transactions_data)
# print(dim_date.head())
# print(len(dim_date))
# print(dim_date.info())
