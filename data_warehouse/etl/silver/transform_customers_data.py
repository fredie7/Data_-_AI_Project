import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "bronze"))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / ""))

import pandas as pd

from extract_data import extract_data
from utils.helper_functions import standardize_columns, DuplicateDataError, logger


def transform_customers_data() -> pd.DataFrame:
    """
    Extract and transform customer data:
    - Standardize column names
    - Convert signup_date to datetime
    - Remove duplicate customers, keeping the latest signup_date
    """

    logger.info("Retrieving customer data...")
    customers_df, _ = extract_data()

    logger.info("Standardizing customer data columns...")
    customers_df = standardize_columns(customers_df)

    logger.info("Converting signup_date to datetime...")
    customers_df["signup_date"] = pd.to_datetime(
        customers_df["signup_date"],
        errors="coerce"
    )

    logger.info("Checking for null values...")
    null_counts = customers_df.isnull().sum()
    logger.info(f"Null value summary:\n{null_counts}")

    duplicate_customers = customers_df.duplicated(subset=["customer_id"]).sum()
    logger.info(f"Found {duplicate_customers} duplicate customer_id values before deduplication.")
    
    # Preventive measure: If there are duplicate customer_id values, log the details and raise an exception
    logger.info("Removing duplicate customer records...")
    customers_df = (
        customers_df
        # Sort by signup_date to ensure the most recent record is kept when dropping duplicates
        .sort_values("signup_date")
        # Keep the last occurrence of each customer_id (the most recent signup_date)
        .drop_duplicates(subset=["customer_id"], keep="last")
        # Reset index after dropping duplicates
        .reset_index(drop=True)
    )

    # Quality check to ensure no duplicate customer_id values remain after deduplication
    duplicate_count = customers_df.duplicated(subset=["customer_id"]).sum()
    if duplicate_count > 0:
        raise DuplicateDataError(
            logger.error(f"Duplicate customer_id values found after deduplication: {duplicate_count}")
        )
   
    logger.info("Customer data transformation completed successfully.")
    return customers_df


if __name__ == "__main__":
    transformed_customers = transform_customers_data()
