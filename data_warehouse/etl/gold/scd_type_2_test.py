import pandas as pd
from dim_customers import scd2_upsert_customer, dim_customer,customers_df
import os

DATA_DIR = "../../raw_data"
customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))

customers_df = customers.copy()

customers_changed = customers_df.copy()
customers_changed.loc[
    customers_changed["customer_id"] == 2386,
    "country"
] = "FI"

dim_customer = scd2_upsert_customer(dim_customer, customers_changed)
print(dim_customer.sort_values(
["customer_id", "effective_from"])
)

CHANGED_CUSTOMER_ID = 2386

cust_history = (
    dim_customer
    .loc[dim_customer["customer_id"] == CHANGED_CUSTOMER_ID]
    .sort_values("effective_from")
)

print(cust_history)

comparison = cust_history[[
    "customer_key",
    # "email",
    "country",
    "effective_from",
    "effective_to",
    "is_current"
]]

print(comparison)

old_row = cust_history[cust_history["is_current"] == False]
new_row = cust_history[cust_history["is_current"] == True]

print("==>> OLD VERSION")
print(old_row[["country", "effective_to"]])
# print(old_row[["email", "country", "effective_to"]])

print("\n==>> NEW VERSION")
print(new_row[["country", "effective_from"]])
# print(new_row[["email", "country", "effective_from"]])

assert old_row["country"].values[0] != new_row["country"].values[0], \
    "Country did not change"

assert old_row["effective_to"].values[0] == new_row["effective_from"].values[0], \
    "Timeline break detected"

print("Change tracked and verified for customer_id =", CHANGED_CUSTOMER_ID)

changed_today = dim_customer[
    dim_customer["effective_from"] == pd.Timestamp.today().normalize()
]

print(changed_today[[
    "customer_id",
    "country",
    "effective_from",
    "is_current"
]])


