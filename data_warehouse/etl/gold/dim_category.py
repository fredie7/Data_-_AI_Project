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
    / "dim_categories.csv"
)


def build_dim_category(transform_transactions_fn):
    # Prepare transactions
    transactions_df = transform_transactions_fn()
    
    logger.info(f"Transformed transactions data shape: {transactions_df.shape}")
    if transactions_df.empty:
        logger.warning("No transactions data available to build dim_category.")
        return pd.DataFrame(
            columns=[
                "category_key",
                "category",
                "is_refundable",
                "return_window_days",
                "transaction_timestamp",
                # "transaction_key",
            ]
        )

    # Create dimension table for category
    dim_category = (
        transactions_df[["category"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    dim_category["category_key"] = range(1, len(dim_category) + 1)

    # Refundability rules
    dim_category["is_refundable"] = dim_category["category"].map(
        {
            "Electronics": True,
            "Food": False,
        }
    ).fillna(False).astype(bool)

    # Return window rules
    dim_category["return_window_days"] = dim_category["category"].map(
        {
            "Electronics": 30,
            "Food": 0,
        }
    ).fillna(0).astype(int)

    # Propagate transaction timestamp and key for later merging
    first_timestamp = transactions_df["transaction_timestamp"].iloc[0] if not transactions_df.empty else None
    dim_category["transaction_timestamp"] = first_timestamp
    # dim_category["transaction_key"] = transactions_df["transaction_key"].iloc[0] if not transactions_df.empty else None

    # Final column order
    dim_category = dim_category[
        [
            "category_key",
            "category",
            "is_refundable",
            "return_window_days",
            "transaction_timestamp",
        ]
    ]

    # Save dimension table to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dim_category.to_csv(output_path, index=False)

    return dim_category


# Usage & Quality checks

logger.info("Building dim_category dimension table...")
dim_category = build_dim_category(transform_transactions_data)
print(dim_category.head())
print(len(dim_category))
# print(dim_category["transaction_key"].nunique())
# print("INFO==>>",dim_category.info())
