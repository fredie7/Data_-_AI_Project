import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "bronze"))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / ""))

import pandas as pd

from extract_data import extract_data

from utils.helper_functions import (
    standardize_columns,
    EXCHANGE_RATES,
    DuplicateDataError,
    logger
)


def transform_transactions_data() -> pd.DataFrame:
    """
    Extract and transform transactions data:
    - Standardize columns
    - Deduplicate transactions
    - Clean and impute missing values
    - Normalize currencies to EUR
    """

    logger.info("Retrieving transactions data...")
    _, transactions_df = extract_data()
    logger.info(f"Initial transactions shape: {transactions_df.shape}")
    
    logger.info("Standardizing column names...")
    transactions_df = standardize_columns(transactions_df)

    logger.info("Replacing food with Food in the category column...")
    transactions_df["category"] = transactions_df["category"].str.replace("food", "Food")

    logger.info("Replacing electronics with Electronics in the category column...")
    transactions_df["category"] = transactions_df["category"].str.replace("electronics", "Electronics")
    
    logger.info("Converting timestamp to datetime...")
    transactions_df["timestamp"] = pd.to_datetime(
        transactions_df["timestamp"],
        errors="coerce"
    )

    logger.info("Converting customer_id to Int64...")
    transactions_df["customer_id"] = transactions_df["customer_id"].astype("Int64")

    # --------------------------------------------------
    # Duplicate handling
    # --------------------------------------------------
    logger.info("Checking for duplicate transaction_id values...")
    duplicate_count = transactions_df.duplicated("transaction_id").sum()

    if duplicate_count > 0:
        logger.warning(f"Found {duplicate_count} duplicate transaction_id records")

        dup_df = transactions_df[transactions_df.duplicated("transaction_id", keep=False)]

        identical_timestamps = (
            dup_df
            .groupby("transaction_id")["timestamp"]
            .nunique()
            .eq(1)
            .any()
        )

        if identical_timestamps:
            logger.info(
                "Some duplicate transaction_id entries share identical timestamps "
                "(likely pipeline retries)"
            )

        logger.info(
            "Dropping duplicate transaction_id records, keeping the latest timestamp"
        )
        transactions_df = (
            transactions_df
            .sort_values("timestamp")
            .drop_duplicates(subset=["transaction_id"], keep="last")
            .reset_index(drop=True)
        )

    remaining_duplicates = transactions_df.duplicated("transaction_id").sum()
    if remaining_duplicates > 0:
        raise DuplicateDataError(
            f"Duplicate transaction_id values remain after deduplication: "
            f"{remaining_duplicates}"
        )

    # --------------------------------------------------
    # Null handling & imputations
    # --------------------------------------------------
    logger.info("Dropping records with null customer_id...")
    transactions_df = transactions_df.dropna(subset=["customer_id"])

    logger.info("Imputing missing currency values...")
    # Create a flag to indicate which records had missing currency values before imputation
    transactions_df["currency_imputed"] = transactions_df["currency"].isna()
    
    # Fill missing currency values with "EUR" and standardize the format
    transactions_df["currency"] = (
        transactions_df["currency"]
        .fillna("EUR")
        .str.strip()
        .str.upper()
    )

    logger.info("Imputing missing category values to preserve transaction completeness...")
    transactions_df["category"] = transactions_df["category"].fillna("unknown")

    # --------------------------------------------------
    # Currency normalization
    # --------------------------------------------------
    logger.info("Mapping exchange rates...")
    transactions_df["exchange_rate"] = transactions_df["currency"].map(EXCHANGE_RATES)

    logger.info("Calculating amount in euros...")
    transactions_df["amount_eur"] = (
        transactions_df["amount"] * transactions_df["exchange_rate"]
    ).round(2)

    # Define the base currency
    transactions_df["base_currency"] = "EUR"

    # Quality check to ensure no null values remain in critical columns after transformations
    null_summary = transactions_df.isnull().sum()
    logger.info(f"Final null value summary:\n{null_summary}")

    logger.info(f"Final transactions shape: {transactions_df.shape}")
    logger.info("Transactions data transformation completed successfully.")
    
    # --------------------------------------------------
    # Add a surrogate key for transactions for future merging with dimension tables
    # --------------------------------------------------
    transactions_df["transaction_key"] = range(1, len(transactions_df) + 1)
    
    transactions_df = transactions_df.rename(columns={
        "currency": "transaction_currency",
        "timestamp": "transaction_timestamp",
    })
    # transactions_df["date"] = transactions_df["transaction_timestamp"].dt.normalize()
    transactions_df["date"] = transactions_df["transaction_timestamp"].dt.date

    # Create a new 'date' column by extracting the date component from the transaction_timestamp
    # transactions_df["transaction_timestamp"] = pd.to_datetime(
    #     transactions_df["transaction_timestamp"],
    #     errors="coerce"
    # )

    # # Create a new 'date' column by flooring the transaction_timestamp to the nearest day
    # transactions_df["date"] = transactions_df["transaction_timestamp"].dt.floor("D")

    
    # print(transactions_df.columns)   
    # print(transactions_df.head()) 
    return transactions_df


if __name__ == "__main__":
    transformed_transactions = transform_transactions_data()
