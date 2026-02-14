"""
LangChain tools exposed to the agent.
Each tool is callable by the LLM during reasoning.
"""

from langchain_core.tools import tool
from core.data_loader import load_business_data
from rag.retriever import build_policy_retriever
from typing import Dict

# Load business data once at startup
business_data = load_business_data()

# Build policy retriever once
policy_retriever = build_policy_retriever()

@tool
def policy_lookup(question: str) -> str:
    """
    Answer questions related to company policies.
    """
    result = policy_retriever.invoke({"query": question})
    return result["result"]

@tool
def average_transaction_amount() -> str:
    """
    Returns the average transaction amount in EUR.
    """
    avg = business_data["transaction_amount_eur"].mean()
    return f"The average transaction amount is {avg:.2f} EUR."

@tool
def get_transaction_field(transaction_id: int, field: str) -> str:
    """
    Get a specific field for a transaction.
    Valid fields:
    transaction_id, customer_id, transaction_amount_eur,
    base_currency, transaction_currency, transaction_timestamp,
    category, country, signup_date, is_high_value_transaction
    """
    if field not in business_data.columns:
        return f"Invalid field. Available fields are: {list(business_data.columns)}"

    row = business_data.loc[business_data["transaction_id"] == transaction_id]
    if row.empty:
        return "Transaction not found."

    return str(row.iloc[0][field])

@tool
def get_customer_transactions(customer_id: int) -> str:
    """
    Get a summary of transactions for a given customer.
    """
    rows = business_data.loc[business_data["customer_id"] == customer_id]

    if rows.empty:
        return "No transactions found for this customer."

    summary = rows[[
    "transaction_id",
    "transaction_amount_eur",
    "base_currency",
    "country",
    "category"
    ]]


    return summary.to_string(index=False)
@tool
def get_transaction_summary(transaction_id: int) -> str:
    """
    Get a human-readable summary of a transaction.
    """
    row = business_data.loc[business_data["transaction_id"] == transaction_id]
    if row.empty:
        return "Transaction not found."

    r = row.iloc[0]

    return (
        f"Transaction {r.transaction_id}:\n"
        f"- Customer ID: {r.customer_id}\n"
        f"- Amount: {r.transaction_amount_eur} {r.base_currency}\n"
        f"- Category: {r.category}\n"
        f"- Country: {r.country}\n"
        f"- Date: {r.transaction_timestamp}\n"
        f"- Is current customer: {r.is_current}"
    )

@tool
def list_transaction_categories() -> str:
    """
    List all transaction categories.
    """
    categories = sorted(business_data["category"].dropna().unique())
    return "Transaction categories: " + ", ".join(categories)

@tool
def get_customer_spending_by_category(customer_id: int) -> str:
    """
    Get spending breakdown by category for a customer.
    """
    rows = business_data.loc[business_data["customer_id"] == customer_id]

    if rows.empty:
        return "No transactions found."

    summary = (
        rows.groupby("category")["transaction_amount_eur"]
        .sum()
        .sort_values(ascending=False)
    )

    return summary.to_string()


@tool
def list_supported_countries() -> str:
    """
    List all countries where transactions have occurred.
    """
    countries = sorted(business_data["country"].dropna().unique())
    return "Supported countries: " + ", ".join(countries)

@tool
def check_high_value_transaction(transaction_id: int, threshold_eur: float = 500) -> str:
    """
    Check if transaction exceeds a EUR threshold.
    """
    row = business_data.loc[business_data["transaction_id"] == transaction_id]
    if row.empty:
        return "Transaction not found."

    transaction_amount_eur = row.iloc[0]["transaction_amount_eur"]

    if transaction_amount_eur > threshold_eur:
        return (
            f"Transaction {transaction_id} is high-value "
            f"({transaction_amount_eur} EUR) and may be reviewed."
        )
    
@tool
def check_cross_border(transaction_id: int) -> str:
    """
    Check if a transaction is cross-border.
    """
    row = business_data.loc[business_data["transaction_id"] == transaction_id]
    if row.empty:
        return "Transaction not found."

    country = row.iloc[0]["country"]

    return (
        f"Transaction {transaction_id} occurred in {country}. "
        "Cross-border transactions may require additional review."
    )

@tool
def list_supported_currencies() -> str:
    """
    List all supported currencies in the system.
    """
    currencies = sorted(business_data["transaction_currency"].dropna().unique())
    return "Supported currencies: " + ", ".join(currencies)

@tool
def get_recent_transactions(customer_id: int, limit: int = 5) -> str:
    """
    Get recent transactions for a customer.
    """
    rows = business_data.loc[business_data["customer_id"] == customer_id]

    if rows.empty:
        return "No transactions found."

    rows = rows.sort_values("transaction_timestamp", ascending=False).head(limit)

    return rows[[
        "transaction_id",
        "transaction_amount_eur",
        "base_currency",
        "country",
        "category",
        "transaction_timestamp",
    ]].to_string(index=False)

def get_customer_profile(customer_id: int) -> str:
    """
    Get customer profile and activity summary.
    """
    rows = business_data.loc[business_data["customer_id"] == customer_id]
    if rows.empty:
        return "Customer not found."

    first = rows.iloc[0]

    return (
        f"Customer {customer_id}:\n"
        f"- Signup date: {first.signup_date}\n"
        f"- Total transactions: {len(rows)}\n"
        f"- Countries used: {rows.country.nunique()}"
    )

@tool
def high_value_by_spend() -> Dict:
    """
    Returns the top 5 customers with the highest total spend.
    """
    stats = business_data.groupby("customer_id").agg(
        transaction_count=("transaction_id", "count"),
        total_spend=("transaction_amount_eur", "sum")
    ).reset_index()

    # Top 5 customers by spend
    top_customers = stats.sort_values("total_spend", ascending=False).head(5)
    
    return {
        "high_value_by_spend": top_customers[["customer_id", "total_spend"]].to_dict(orient="records")
    }

@tool
def platform_statistics() -> str:
    """
    Get high-level platform statistics.
    """
    return (
        f"Platform statistics:\n"
        f"- Total transactions: {len(business_data)}\n"
        f"- Total customers: {business_data['customer_id'].nunique()}\n"
        f"- Average amount (EUR): {business_data['transaction_amount_eur'].mean():.2f}\n"
        f"- Countries served: {business_data['country'].nunique()}"
    )

@tool
def high_value_by_frequency() -> Dict:
    """
    Returns the top 5 customers with the highest number of transactions.
    """
    stats = business_data.groupby("customer_id").agg(
        transaction_count=("transaction_id", "count"),
        total_spend=("transaction_amount_eur", "sum")
    ).reset_index()

    # Top 5 customers by transaction count
    top_customers = stats.sort_values("transaction_count", ascending=False).head(5)
    
    return {
        "high_value_by_frequency": top_customers[["customer_id", "transaction_count"]].to_dict(orient="records")
    }


tools = [
    policy_lookup,
    average_transaction_amount,
    get_transaction_field,
    get_customer_transactions,
    get_transaction_summary,
    list_transaction_categories,
    get_customer_spending_by_category,
    list_supported_countries,
    check_high_value_transaction,
    check_cross_border
]