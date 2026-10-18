import chromadb
from google import genai
import os

# Create a local Chroma database
client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection
try:
    client.delete_collection(name="sql_knowledge")
except Exception:
    pass

collection = client.create_collection(
    name="sql_knowledge"
)

print("Vector DB created!")
print("Collection:", collection.name)

collection.add(
    documents=[
        "The Customer table contains customer_id, customer_name, and city.",
        "The Transactions table contains transaction_id, customer_id, product_id, transaction_date, amount, and status. customer_id joins to Customer.customer_id. Only transactions with status 'Completed' should be included when calculating total transaction amount.",
        "The Product table contains product_id, product_name, category, and price.",
        "The Order table contains order_id, customer_id, and order_date.",
        "Business rule: A customer is considered active if they have made at least one transaction in the last 90 days.",
    ],
    ids=[
        "customer_table",
        "transaction_table",
        "product_table",
        "order_table",
        "business_rule_active_customer",
    ]
)

print("Knowledge chunk added!")

print("Number of items:", collection.count())


#print("Stored data:")
#print(collection.get(include=["documents", "embeddings"]))

#question = "Which fields are available for a customer?"
question = input("Enter your question: ")

results = collection.query(
    query_texts=[question],
    n_results=3
)

#print("\nQuestion:", question)
#print("Retrieved:", results["documents"])
#print("Distance:", results["distances"])

print("\n--- RAG CONTEXT ---")

for doc in results["documents"][0]:
    print(doc)

# ---- NEW: build the prompt that carries retrieved context to Gemini ----
retrieved_chunks = results["documents"][0] 
context_text = "\n".join(retrieved_chunks)
 
prompt = f"""You are an AI SQL assistant.

Generate a SQL query to answer the user's question.

Use ONLY the database knowledge provided below.
Do not invent tables, columns, relationships, or business rules.

If the required information is not available in the context,
say "I don't have enough information."

Database Knowledge:
{context_text}

User Question:
{question}

Return only the SQL query.
"""
 
print("\n--- Prompt sent to Gemini ---")
print(prompt)


# ---- NEW: call Gemini with that prompt ----
# Connect to Gemini
gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

response = gemini_client.models.generate_content(
    #model="gemini-3.7-flash",
    model="models/gemini-3-flash-preview",
    contents=prompt
)

print("\n--- Gemini's Answer ---")
print(response.text)