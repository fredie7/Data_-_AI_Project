import logging
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# standardize columns
def standardize_columns(input_data):
    # Convert all column names to lowercase
    input_data.columns = input_data.columns.str.lower()
    
    # Replace white spaces with underscores in column names
    input_data.columns = input_data.columns.str.replace(' ', '_')
    
    return input_data

# Currency exchange rates
EXCHANGE_RATES = {
    "EUR": 1.0,
    "SEK": 0.09496,
    "NOK": 0.08787
}

# Custom exception for duplicate data
class DuplicateDataError(Exception):
    """Custom exception for duplicate data errors."""
    pass
