import sqlite3
import chromadb
from google import genai
import os
import re

# ---- Connect to vector DB (created earlier) ----

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="sql_knowledge")

# ---- Connect to SQLite DB (created in step 1) ----

conn = sqlite3.connect("sql_assistant.db")
cursor = conn.cursor()

# ---- Retrieval tools the agent can call ----

question = input("Enter your question: ")

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def search_schema(question):
    results = collection.query(query_texts=[question], n_results=5)
    docs = results["documents"][0]
    schema_docs = [doc for doc in docs if "table contains" in doc.lower()]
    return "\n".join(schema_docs)


def search_business_rules(question):
    results = collection.query(query_texts=[question], n_results=5)
    docs = results["documents"][0]
    rule_docs = [doc for doc in docs if "business rule" in doc.lower()]
    return "\n".join(rule_docs)


# ---- Agent loop: Gemini decides which tool to call, or when it's done ----

context = ""
MAX_ITERATIONS = 5   # safety cap so a confused agent can't loop forever and burn API calls
iterations = 0

while iterations < MAX_ITERATIONS:
    iterations += 1

    agent_prompt = f"""
You are an AI SQL assistant acting as an agent.

User question:
{question}

Information collected so far:
{context}

Available actions:

SEARCH_SCHEMA
Use this when you need table names, columns, or relationships.

SEARCH_BUSINESS_RULES
Use this when you need definitions or business rules.

GENERATE_SQL
Use this only when you have enough information to generate the SQL.

Choose the next action.

Return ONLY one of:
SEARCH_SCHEMA
SEARCH_BUSINESS_RULES
GENERATE_SQL
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash",
        contents=agent_prompt
    )

    decision = response.text.strip()

    print(f"\n--- Agent decision (iteration {iterations}) ---")
    print(decision)

    if decision == "SEARCH_SCHEMA":
        result = search_schema(question)
        print("\n--- Schema tool result ---")
        print(result)
        context += f"\nSchema information:\n{result}\n"

    elif decision == "SEARCH_BUSINESS_RULES":
        result = search_business_rules(question)
        print("\n--- Business rule tool result ---")
        print(result)
        context += f"\nBusiness rules:\n{result}\n"

    elif decision == "GENERATE_SQL":
        break

    else:
        print("Unexpected agent decision. Stopping.")
        break

else:
    # while loop hit MAX_ITERATIONS without breaking
    print(f"\nAgent could not decide within {MAX_ITERATIONS} steps. Stopping to avoid wasting API calls.")
    conn.close()
    exit()

# ---- Generate final SQL using whatever the agent collected ----

final_prompt = f"""
You are an AI SQL assistant.

Generate SQL to answer the user's question.

Use ONLY the information collected by the agent.

Do not invent:
- tables
- columns
- relationships
- business rules

Actual SQLite table names:
Customer, Transactions

Collected information:
{context}

User question:
{question}

Return ONLY the raw SQL query.
"""

response = gemini_client.models.generate_content(
    model="gemini-3.7-flash",
    contents=final_prompt
)

generated_sql = response.text.strip()

generated_sql = re.sub(
    r"^```sql\s*|^```\s*|```$",
    "",
    generated_sql,
    flags=re.MULTILINE
).strip()

print("\n--- Final SQL ---")
print(generated_sql)

# ---- Validate before executing (never trust generated SQL blindly) ----

BLOCKED_WORDS = ["drop", "delete", "update", "insert", "alter"]

def validate(sql: str) -> bool:
    sql_lower = sql.lower()
    if not sql_lower.strip().startswith("select"):
        return False
    for word in BLOCKED_WORDS:
        if word in sql_lower:
            return False
    return True

if not validate(generated_sql):
    print("\n--- BLOCKED ---")
    print("Gemini generated a query that isn't a safe SELECT statement. Refusing to run it.")
    conn.close()
    exit()

# ---- Execute the agent's SQL against the real database ----

try:
    cursor.execute(generated_sql)
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]

    print("\n--- Query Results ---")
    print(column_names)
    for row in rows:
        print(row)

except sqlite3.Error as e:
    print("\n--- SQL Execution Error ---")
    print(e)
    print("(This means Gemini generated SQL that doesn't match the real table structure.)")

conn.close()