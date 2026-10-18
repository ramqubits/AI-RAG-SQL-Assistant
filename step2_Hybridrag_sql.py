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

# ---- Retrieval ----

question = input("Enter your question: ")

# ---- Hybrid Retrieval: Vector search + Keyword search, combined ----

# 1. Vector (semantic) search — same as before
vector_results = collection.query(
    query_texts=[question],
    n_results=3
)
vector_docs = vector_results["documents"][0]
vector_distances = vector_results["distances"][0]

print("\n--- Vector search candidates ---")
for doc, dist in zip(vector_docs, vector_distances):
    print(f"[distance={dist:.4f}] {doc}")

# 2. Keyword search — catches exact-word matches vector search might rank low
#    (e.g. an exact status value like "Completed", or a table name)
all_stored = collection.get(include=["documents"])
all_chunks = all_stored["documents"]

def keyword_search(question: str, chunks: list[str]) -> list[str]:
    question_words = set(re.sub(r"[^a-z0-9\s]", "", question.lower()).split())
    matches = []
    for chunk in chunks:
        chunk_words = set(re.sub(r"[^a-z0-9\s]", "", chunk.lower()).split())
        overlap = question_words & chunk_words
        if overlap:
            matches.append((chunk, len(overlap)))
    matches.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in matches]

keyword_matches = keyword_search(question, all_chunks)

print("\n--- Keyword search matches ---")
for doc in keyword_matches:
    print("-", doc)

# 3. Combine both result sets, removing duplicates, vector matches first
#    (vector search stays primary since it understands meaning; keyword
#    search just adds anything exact-match it found that vector missed)
combined_docs = list(vector_docs)
for doc in keyword_matches:
    if doc not in combined_docs:
        combined_docs.append(doc)

print("\n--- Combined (hybrid) candidates ---")
for doc in combined_docs:
    print("-", doc)

# ---- Retrieval distances only exist for vector results, so build a
#      lookup and give any keyword-only match a "safe" distance of 0.0
#      (meaning: trust it, since it was an exact word match, not a guess)
distance_lookup = dict(zip(vector_docs, vector_distances))
retrieved_docs = combined_docs
retrieved_distances = [distance_lookup.get(doc, 0.0) for doc in combined_docs]

# ---- reject chunks that are too far from the question ----
# Chroma's default distance is smaller = more similar (0 = identical meaning).
# This threshold is a starting point — you WILL need to tune it by testing
# your own questions and watching which distances come back.
DISTANCE_THRESHOLD = 1.3

relevant_chunks = [
    doc for doc, dist in zip(retrieved_docs, retrieved_distances)
    if dist <= DISTANCE_THRESHOLD
]

print("\n--- Relevant chunks (after filtering) ---")
for doc in relevant_chunks:
    print("-", doc)

# ---- NEW: if nothing survives filtering, stop here — don't call Gemini at all ----
if not relevant_chunks:
    print("\nI don't have enough information.")
    conn.close()
    exit()

retrieved_chunks = relevant_chunks
context_text = "\n".join(retrieved_chunks)

# ---- Build the prompt (same as before) ----

prompt = f"""You are an AI SQL assistant.

Generate a SQL query to answer the user's question.

Use ONLY the database knowledge provided below.
Do not invent tables, columns, relationships, or business rules.

The actual SQLite table names are: Customer, Transactions.

If the required information is not available in the context,
say "I don't have enough information."

Database Knowledge:
{context_text}

User Question:
{question}

Return ONLY the raw SQL query. No explanation, no markdown formatting, no ```sql fences.
"""

# ---- Call Gemini ----

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = gemini_client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt
)

generated_sql = response.text.strip()

# Gemini sometimes wraps SQL in ```sql ... ``` even when told not to.
# Strip that off so sqlite3 doesn't choke on it.
generated_sql = re.sub(r"^```sql\s*|^```\s*|```$", "", generated_sql, flags=re.MULTILINE).strip()

print("\n--- Gemini-generated SQL ---")
print(generated_sql)

# ---- Validate before executing (never trust generated SQL blindly) ----

BLOCKED_WORDS = ["drop", "delete", "update", "insert", "alter"]

def validate(sql: str) -> bool:
    sql_lower = sql.lower()

    # Rule 1: must start with SELECT
    if not sql_lower.strip().startswith("select"):
        return False

    # Rule 2: must not contain dangerous keywords
    for word in BLOCKED_WORDS:
        if word in sql_lower:
            return False

    return True

if not validate(generated_sql):
    print("\n--- BLOCKED ---")
    print("Gemini generated a query that isn't a safe SELECT statement. Refusing to run it.")
    conn.close()
    exit()

# ---- Execute the generated SQL against the real database ----

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