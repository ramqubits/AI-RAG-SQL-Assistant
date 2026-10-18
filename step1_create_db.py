import sqlite3
import chromadb

# Connect (creates the file if it doesn't exist)
conn = sqlite3.connect("sql_assistant.db")
cursor = conn.cursor()

# ---- Create tables (matching your vector DB schema chunks) ----

cursor.execute("""
CREATE TABLE IF NOT EXISTS Customer (
    customer_id INTEGER PRIMARY KEY,
    customer_name TEXT,
    city TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS Transactions (
    transaction_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product_id INTEGER,
    transaction_date TEXT,
    amount REAL,
    status TEXT,
    FOREIGN KEY (customer_id) REFERENCES Customer(customer_id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    final_answer TEXT,
    sql_used TEXT,
    created_at TEXT DEFAULT (datetime('now'))
)
""")

# ---- Insert sample data ----

cursor.execute("DELETE FROM Customer")
cursor.execute("DELETE FROM Transactions")

cursor.executemany(
    "INSERT INTO Customer (customer_id, customer_name, city) VALUES (?, ?, ?)",
    [
        (1, "Arun", "Chennai"),
        (2, "Priya", "Bangalore"),
        (3, "Karthik", "Chennai"),
    ]
)

cursor.executemany(
    """INSERT INTO Transactions
       (transaction_id, customer_id, product_id, transaction_date, amount, status)
       VALUES (?, ?, ?, ?, ?, ?)""",
    [
        (101, 1, 10, "2026-08-01", 2500.0, "Completed"),
        (102, 1, 11, "2026-07-25", 1000.0, "Cancelled"),
        (103, 2, 10, "2026-07-20", 3200.0, "Completed"),
        (104, 3, 12, "2026-06-15", 750.0, "Completed"),
    ]
)

conn.commit()

print("Database created: sql_assistant.db")
print("Customer rows:", cursor.execute("SELECT COUNT(*) FROM Customer").fetchone()[0])
print("Transaction rows:", cursor.execute("SELECT COUNT(*) FROM Transactions").fetchone()[0])

# ---- Manually written query (this is what step 3 will replace with Gemini's SQL) ----

query = """
SELECT customer_id, SUM(amount) AS total_completed_amount
FROM Transactions
WHERE status = 'Completed'
GROUP BY customer_id
"""

cursor.execute(query)
rows = cursor.fetchall()

print("Query:\n", query)
print("\nResults:")
for row in rows:
    print(row)


# ---- 1. Create and populate Product table ----
 
cursor.execute("""
CREATE TABLE IF NOT EXISTS Product (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT,
    category TEXT,
    price REAL
)
""")
 
# Matches the product_ids already referenced in your existing Transactions
# rows (10, 11, 12) plus a couple more for variety in future test data.
sample_products = [
    (10, "Wireless Mouse", "Electronics", 1600.0),
    (11, "Desk Lamp", "Furniture", 1000.0),
    (12, "Office Chair", "Furniture", 750.0),
    (13, "Bluetooth Speaker", "Electronics", 2200.0),
    (14, "Notebook Set", "Stationery", 300.0),
]
 
cursor.execute("SELECT COUNT(*) FROM Product")
if cursor.fetchone()[0] == 0:
    cursor.executemany(
        "INSERT INTO Product (product_id, product_name, category, price) VALUES (?, ?, ?, ?)",
        sample_products,
    )
    print(f"Inserted {len(sample_products)} products.")
else:
    print("Product table already has data - skipping insert.")
 
conn.commit()
conn.close()
 
# ---- 2. Add more business rules to ChromaDB ----
 
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="sql_knowledge")
 
new_rules = [
    {
        "id": "rule_high_value_transaction",
        "doc": "Business rule: A transaction is considered high-value if its amount exceeds 2000.",
    },
    {
        "id": "rule_cancelled_exclusion",
        "doc": "Business rule: Cancelled transactions must always be excluded from revenue calculations, even if they appear in the Transactions table.",
    },
    {
        "id": "rule_category_classification",
        "doc": "Business rule: Product categories are Electronics, Furniture, and Stationery. Every product belongs to exactly one category.",
    },
]
 
existing_ids = set(collection.get()["ids"])
to_add = [r for r in new_rules if r["id"] not in existing_ids]
 
if to_add:
    collection.add(
        ids=[r["id"] for r in to_add],
        documents=[r["doc"] for r in to_add],
    )
    print(f"Added {len(to_add)} new business rules to ChromaDB.")
else:
    print("Business rules already present - skipping insert.")
 
print("\nDone. Re-run your agent - search_schema and search_business_rules "
      "should now return richer results.")

conn.close()	