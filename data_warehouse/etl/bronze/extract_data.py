# =========================================
# Load the data from the data directory
# =========================================

# Import dependencies

import os
import pandas as pd
import logging
import datetime as datetime

# Define directory containing csv files
DATA_DIR = "../../raw_data"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Function to extract data
def extract_data():
    """Load structured CSVs."""
    logger.info("=====Loading customers.csv and transactions.csv=====")

    # Load CSV files
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))

    # Add created_at to both DataFrames to presrve the timestamp of when the data was loaded into the data warehouse
    now = datetime.datetime.now()
    customers["created_at"] = now
    transactions["created_at"] = now

    # Log the shape of the loaded data
    logger.info(f"=====Loaded {len(customers)} customers and {len(transactions)} transactions=====")
    # Return the loaded data
    return customers, transactions

extract_data()