"""
mcp_server.py

Exposes your SQL Assistant's capabilities as MCP tools, so Claude
(Claude Desktop, or any MCP-compatible client) can call them directly —
deciding on its own when to look up schema, when to look up business
rules, and when to run a query. No agent-decision loop needed on your
side anymore; the MCP client (Claude) IS the agent.

Install first:
    pip install mcp chromadb

Run it directly to sanity-check it starts without errors:
    python mcp_server.py

To actually use it from Claude Desktop, add it to your
claude_desktop_config.json (see notes at the bottom of this file).
"""

import sqlite3
import re
import chromadb
from mcp.server.mcpserver import MCPServer
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

chroma_client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))

# ---- Connect to your existing databases (created in earlier steps) ----

#chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="sql_knowledge")

# NOTE: SQLite connections are not thread-safe by default, and MCP
# servers may handle multiple calls. check_same_thread=False is fine
# here since this is a single-user local learning project — for
# anything beyond that you'd want a connection-per-call pattern instead.
#conn = sqlite3.connect("sql_assistant.db", check_same_thread=False)
conn = sqlite3.connect(os.path.join(BASE_DIR, "sql_assistant.db"), check_same_thread=False)

# ---- Create the MCP server ----

mcp = MCPServer("sql-assistant")


# ---- Tool 1: schema lookup ----

@mcp.tool()
def search_schema(question: str) -> str:
    """Look up table names, columns, and relationships relevant to a
    natural-language question about the database."""
    results = collection.query(query_texts=[question], n_results=5)
    docs = results["documents"][0]
    schema_docs = [doc for doc in docs if "table contains" in doc.lower()]
    return "\n".join(schema_docs) if schema_docs else "No matching schema information found."


# ---- Tool 2: business rule lookup ----

@mcp.tool()
def search_business_rules(question: str) -> str:
    """Look up business rules or definitions relevant to a
    natural-language question (e.g. what counts as an 'active' customer)."""
    results = collection.query(query_texts=[question], n_results=5)
    docs = results["documents"][0]
    rule_docs = [doc for doc in docs if "business rule" in doc.lower()]
    return "\n".join(rule_docs) if rule_docs else "No matching business rules found."


# ---- Tool 3: run a SQL query, with the same safety validation as before ----

BLOCKED_WORDS = ["drop", "delete", "update", "insert", "alter"]


def _validate(sql: str) -> bool:
    sql_lower = sql.lower()
    if not sql_lower.strip().startswith("select"):
        return False
    for word in BLOCKED_WORDS:
        if word in sql_lower:
            return False
    return True


@mcp.tool()
def run_sql(sql: str) -> str:
    """Execute a read-only SELECT query against the SQLite database and
    return the results. Only SELECT statements are allowed — anything
    else (DROP, DELETE, UPDATE, INSERT, ALTER) is refused."""
    # Strip markdown fences in case the caller includes them
    cleaned = re.sub(r"^```sql\s*|^```\s*|```$", "", sql.strip(), flags=re.MULTILINE).strip()

    if not _validate(cleaned):
        return "REFUSED: only safe SELECT statements are allowed."

    try:
        cursor = conn.cursor()
        cursor.execute(cleaned)
        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        return f"Columns: {column_names}\nRows: {rows}"
    except sqlite3.Error as e:
        return f"SQL execution error: {e}"


# ---- Tool 4: what tables actually exist (helps the caller avoid guessing) ----

@mcp.tool()
def list_tables() -> str:
    """List the actual table names available in the SQLite database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    return ", ".join(tables)


# ---- Start the server ----

if __name__ == "__main__":
    mcp.run()

# ---------------------------------------------------------------
# To connect this to Claude Desktop:
#
# 1. Find your claude_desktop_config.json:
#      Windows: %APPDATA%\Claude\claude_desktop_config.json
#      Mac:     ~/Library/Application Support/Claude/claude_desktop_config.json
#
# 2. Add an entry like this (adjust the path to where this file lives):
#
# {
#   "mcpServers": {
#     "sql-assistant": {
#       "command": "python",
#       "args": ["D:/Ram/Project/AI_Projects/AI_RAG_SQL_Assistant/mcp_server.py"]
#     }
#   }
# }
#
# 3. Restart Claude Desktop. You should see "sql-assistant" tools
#    available (look for the hammer/tools icon in the chat input).
#
# 4. Ask Claude something like: "What tables are in my database, and
#    what are the total completed transaction amounts per customer?"
#    Watch Claude decide, on its own, to call list_tables, then
#    search_schema, then run_sql — that decision-making is now
#    Claude's job instead of your agent_prompt loop.
# ---------------------------------------------------------------