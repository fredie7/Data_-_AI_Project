import sys
from pathlib import Path
import pandas as pd

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "silver"))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / ""))

from utils.helper_functions import logger

from transform_customers_data import transform_customers_data

# Load customers
customers_df = transform_customers_data()

# Output path
output_path = (
    Path(__file__).resolve().parent.parent.parent
    / "processed_data"
    / "dim_customers.csv"
)

# Empty dimension definition
dim_customer = pd.DataFrame(columns=[
    "customer_key",
    "customer_id",
    "country",
    "signup_date",
    "effective_from",
    "effective_to",
    "is_current"
])


def scd2_upsert_customer(
    dim_customer: pd.DataFrame,
    stg_customers: pd.DataFrame,
    run_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    """
    Perform SCD Type 2 upsert for customer dimension.
    """

    run_date = run_date or pd.Timestamp.today().normalize()

    # Ensure uniqueness of business key
    stg_customers = stg_customers.drop_duplicates(subset=["customer_id"]).copy()

    # ------------------------------------------------------------------
    # CASE 1: Initial load
    # ------------------------------------------------------------------
    if dim_customer.empty:
        inserts = stg_customers.copy()

        inserts["customer_key"] = range(1, len(inserts) + 1)
        inserts["effective_from"] = inserts["signup_date"]
        inserts["effective_to"] = pd.NaT
        inserts["is_current"] = True

        result = inserts[[
            "customer_key",
            "customer_id",
            "country",
            "signup_date",
            "effective_from",
            "effective_to",
            "is_current"
        ]]

        # Tests
        assert result["customer_key"].is_unique
        # Since this is the initial load, all records should be current and have no effective_to date
        assert result["is_current"].all()
        assert result["effective_to"].isna().all()

        logger.info("Initial load completed successfully.")
        return result

    # ------------------------------------------------------------------
    # CASE 2: Incremental SCD2
    # ------------------------------------------------------------------
    dim_current = dim_customer[dim_customer["is_current"]].copy()

    merged = stg_customers.merge(
        dim_current,
        on="customer_id",
        how="left",
        suffixes=("_stg", "_dim")
    )

    is_new = merged["customer_key"].isna()

    country_changed = (
        merged["country_stg"].fillna("") != merged["country_dim"].fillna("")
    )

    is_changed = ~is_new & country_changed

    # ------------------------------------------------------------------
    # Expire changed records
    # ------------------------------------------------------------------
    keys_to_expire = merged.loc[is_changed, "customer_key"]

    if not keys_to_expire.empty:
        dim_customer.loc[
            dim_customer["customer_key"].isin(keys_to_expire),
            ["effective_to", "is_current"]
        ] = [run_date, False]

        expired = dim_customer.loc[
            dim_customer["customer_key"].isin(keys_to_expire)
        ]

        assert expired["effective_to"].notna().all()
        assert not expired["is_current"].any()

    # ------------------------------------------------------------------
    # Insert new & changed records
    # ------------------------------------------------------------------
    inserts = merged.loc[is_new | is_changed, [
        "customer_id",
        "country_stg",
        "signup_date_stg"
    ]].rename(columns={
        "country_stg": "country",
        "signup_date_stg": "signup_date"
    })

    if inserts.empty:
        return dim_customer

    logger.info(f"Inserting {len(inserts)} new/changed records into dim_customer...")
    next_key = dim_customer["customer_key"].max() + 1

    inserts["customer_key"] = range(next_key, next_key + len(inserts))
    inserts["effective_from"] = run_date
    inserts["effective_to"] = pd.NaT
    inserts["is_current"] = True

    dim_customer = pd.concat([dim_customer, inserts], ignore_index=True)

    # Final integrity checks
    assert dim_customer["customer_key"].is_unique
    assert dim_customer.loc[
        dim_customer["is_current"], "customer_id"
    ].is_unique
    assert dim_customer.loc[
        dim_customer["is_current"], "effective_to"
    ].isna().all()

    logger.info("SCD2 upsert completed successfully.")
    return dim_customer


# Run SCD2
dim_customer = scd2_upsert_customer(dim_customer, customers_df)

# Save
dim_customer.to_csv(output_path, index=False)
# print(dim_customer.info())
