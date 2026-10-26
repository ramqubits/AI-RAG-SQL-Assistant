import sqlite3
import os
import re
import json
import time
import logging
from datetime import date

import chromadb
from google import genai
from google.genai import types

# ---- Structured logging ----
# Prints still happen everywhere they did before (console UX unchanged).
# This additionally writes a reviewable file log, so a run can be inspected
# after the fact instead of only living in a scrolled-away terminal.

logger = logging.getLogger("data_analyst_agent")
logger.setLevel(logging.INFO)
if not logger.handlers:  # avoid duplicate handlers if this module is imported twice
    file_handler = logging.FileHandler("agent_activity.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)


# ---- API call / cost tracking ----
# Gemini counts a request against your quota whether it succeeds or fails
# (confirmed by your own logs - failed 429/503 attempts still count). So we
# track every attempt, persisted to disk, to give visibility into the daily
# cap BEFORE you hit it, not just after.

USAGE_FILE = "api_usage.json"


def _load_usage():
    today = str(date.today())
    if os.path.exists(USAGE_FILE):
        try:
            with open(USAGE_FILE, "r") as f:
                data = json.load(f)
            if data.get("date") == today:
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"date": today, "count": 0}


def record_api_call():
    """Call once per actual API attempt (success or failure). Persists a
    running daily count to disk so it survives across separate script runs
    on the same day, and returns today's count after recording this call."""
    data = _load_usage()
    data["count"] += 1
    try:
        with open(USAGE_FILE, "w") as f:
            json.dump(data, f)
    except OSError as e:
        logger.warning(f"Could not persist API usage count: {e}")
    return data["count"]


def get_api_usage_today():
    return _load_usage()["count"]


# ---- Connections ----
# On hosted platforms (e.g. Streamlit Cloud) files checked out from GitHub can
# be read-only, which makes SQLite fail with "attempt to write a readonly
# database" as soon as the agent saves history. If a data file/folder isn't
# writable, we transparently use a copy in the system temp directory instead.
# (That copy is ephemeral: it resets whenever the hosted app restarts.)

import shutil
import tempfile


def _is_writable(path):
    try:
        if os.path.isdir(path):
            probe_dir = path
            files = [os.path.join(path, "chroma.sqlite3")]
        else:
            probe_dir = os.path.dirname(os.path.abspath(path))
            files = [path]
        probe = os.path.join(probe_dir, ".write_test")
        with open(probe, "w") as f:
            f.write("x")
        os.remove(probe)
        return all(os.access(f, os.W_OK) for f in files if os.path.exists(f))
    except OSError:
        return False


def writable_path(path):
    """Return `path` if it's writable, else a writable temp copy of it."""
    if _is_writable(path):
        return path
    abs_path = os.path.abspath(path)
    target = os.path.join(tempfile.gettempdir(), "agent_data", os.path.basename(abs_path))
    if not os.path.exists(target):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.isdir(abs_path):
            shutil.copytree(abs_path, target)
        else:
            shutil.copy2(abs_path, target)
        for root, dirs, files in os.walk(target):  # copies can inherit read-only bits
            for name in files:
                os.chmod(os.path.join(root, name), 0o644)
    logger.warning(f"{path} is read-only here; using writable copy at {target}")
    return target


chroma_client = chromadb.PersistentClient(path=writable_path("./chroma_db"))
collection = chroma_client.get_or_create_collection(name="sql_knowledge")

conn = sqlite3.connect(writable_path("sql_assistant.db"), check_same_thread=False)  # Streamlit reruns scripts on varying threads
cursor = conn.cursor()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(api_version="v1beta"),
)

#MODEL = "gemini-3.8-flash"
MODEL = "gemini-3.1-flash-lite" # "gemini-3.8-flash"

call_counts = {}
last_result = {}
last_successful_query_result = None
last_successful_sql = None  # track the actual SQL too, for saving to history

MAX_API_RETRIES = 3
DEFAULT_RETRY_WAIT_SECONDS = 5
HISTORY_LIMIT = 5  # how many past exchanges to load as context


# ---- Tool 1: search schema ----

def search_schema(question):
    try:
        results = collection.query(query_texts=[question], n_results=5)
    except Exception as e:
        return f"SEARCH_ERROR: could not search schema ({e})"

    docs = results["documents"][0]
    schema_docs = [doc for doc in docs if "table contains" in doc.lower()]
    return "\n".join(schema_docs) if schema_docs else "No matching schema found."


# ---- Tool 2: search business rules ----

def search_business_rules(question):
    try:
        results = collection.query(query_texts=[question], n_results=5)
    except Exception as e:
        return f"SEARCH_ERROR: could not search business rules ({e})"

    docs = results["documents"][0]
    rule_docs = [doc for doc in docs if "business rule" in doc.lower()]
    return "\n".join(rule_docs) if rule_docs else "No matching business rules found."


# ---- SQL safety check ----

BLOCKED_WORDS = ["drop", "delete", "update", "insert", "alter"]

def validate_sql(sql):
    sql_lower = sql.lower().strip()
    if not (sql_lower.startswith("select") or sql_lower.startswith("with")):
        return False, "Query must start with SELECT or WITH (for CTEs)"
    if ";" in sql_lower.rstrip(";"):
        return False, "Multiple statements are not allowed"
    for word in BLOCKED_WORDS:
        if re.search(rf"\b{word}\b", sql_lower):
            return False, f"Blocked word found: {word}"
    return True, "OK"


# ---- Tool 3: generate + run SQL ----

def generate_sql(sql):
    global last_successful_query_result, last_successful_sql

    is_valid, reason = validate_sql(sql)
    if not is_valid:
        return f"BLOCKED: {reason}"

    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description]
        result = json.dumps({"columns": columns, "rows": rows}, default=str)

        last_successful_query_result = result
        last_successful_sql = sql

        return result
    except sqlite3.Error as e:
        return f"SQL_ERROR: {e}"


TOOL_FUNCTIONS = {
    "search_schema": search_schema,
    "search_business_rules": search_business_rules,
    "generate_sql": generate_sql,
}

tools = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="search_schema",
        description="Search the database schema for tables, columns, and relationships.",
        parameters_json_schema={
            "type": "object",
            "properties": {"question": {"type": "string", "description": "The user's database question."}},
            "required": ["question"],
        },
    ),
    types.FunctionDeclaration(
        name="search_business_rules",
        description="Search for business rules or definitions relevant to the question.",
        parameters_json_schema={
            "type": "object",
            "properties": {"question": {"type": "string", "description": "The user's database question."}},
            "required": ["question"],
        },
    ),
    types.FunctionDeclaration(
        name="generate_sql",
        description=(
            "Call this only once you have enough schema and business rule info. "
            "Provide a single SELECT statement, no markdown."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A single SELECT statement."}},
            "required": ["sql"],
        },
    ),
])


def extract_retry_delay(error_message):
    match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", str(error_message))
    if match:
        return float(match.group(1))
    return DEFAULT_RETRY_WAIT_SECONDS


def is_daily_quota_error(error):
    """True if the error is a per-day quota exhaustion, which retrying
    within this run can never fix."""
    text = str(error)
    return "RESOURCE_EXHAUSTED" in text and "PerDay" in text


def call_gemini_with_retry(contents, system_instruction, use_tools=True, tool_list=None):
    last_error = None
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            config_kwargs = {"system_instruction": system_instruction}
            if tool_list is not None:
                config_kwargs["tools"] = tool_list
            elif use_tools:
                config_kwargs["tools"] = [tools]
            today_count = record_api_call()
            logger.info(f"API call attempt {attempt}/{MAX_API_RETRIES} (call #{today_count} today)")
            return client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as e:
            last_error = e
            logger.warning(f"API call failed (attempt {attempt}/{MAX_API_RETRIES}): {e}")
            if is_daily_quota_error(e):
                print("[error] Daily free-tier quota exhausted - not retrying. "
                      "Try again after the quota resets, or use a different key/model.")
                logger.error("Daily quota exhausted - aborting without further retries.")
                raise RuntimeError(f"Daily quota exhausted: {e}")
            wait_seconds = extract_retry_delay(e)
            print(f"[warning] Gemini call failed (attempt {attempt}/{MAX_API_RETRIES}): {e}")
            if attempt < MAX_API_RETRIES:
                print(f"[info] Waiting {wait_seconds:.0f}s before retrying (as requested by the API)...")
                time.sleep(wait_seconds)

    logger.error(f"Gemini API failed after {MAX_API_RETRIES} attempts: {last_error}")
    raise RuntimeError(f"Gemini API failed after {MAX_API_RETRIES} attempts: {last_error}")


def get_fallback_result_text():
    """Build a printable/returnable string from the last successful query,
    or return None if there isn't one."""
    if not last_successful_query_result:
        return None
    parsed = json.loads(last_successful_query_result)
    lines = ["--- Showing last successful query result instead ---",
             f"Columns: {parsed['columns']}"]
    for row in parsed["rows"]:
        lines.append(str(row))
    return "\n".join(lines)


# ---- Load recent history from the database ----

def load_recent_history(limit=HISTORY_LIMIT):
    cursor.execute(
        "SELECT question, final_answer FROM conversation_history "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cursor.fetchall()
    rows.reverse()  # oldest first, so it reads naturally
    return rows


# ---- Save this exchange to the database ----

def save_to_history(question, final_answer, sql_used):
    # History is a convenience, not the answer itself - if saving fails
    # (read-only storage, locked file, etc.) log it and carry on.
    try:
        cursor.execute(
            "INSERT INTO conversation_history (question, final_answer, sql_used) "
            "VALUES (?, ?, ?)",
            (question, final_answer, sql_used),
        )
        conn.commit()
    except sqlite3.Error as e:
        logger.warning(f"Could not save conversation history: {e}")


# ---- The agent loop ----

def run_agent(question, max_iterations=6):
    """Runs the agent loop for a single question and RETURNS the final
    answer text (in addition to printing progress along the way), so
    callers like the planner's run_plan() can chain results together."""
    global last_successful_query_result, last_successful_sql

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = ", ".join(
        row[0] for row in cursor.fetchall() if row[0] != "conversation_history"
    )

    # Build a short "previous conversation" block from history
    history_rows = load_recent_history()
    if history_rows:
        history_text = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in history_rows
        )
        history_block = f"""
Recent previous questions and answers in this session's history (for context
only - only use these if the current question clearly refers back to them,
e.g. "same as before", "what about last month"):

{history_text}
"""
    else:
        history_block = ""

    system_instruction = f"""You are an AI SQL assistant with access to tools.
Actual SQLite table names available: {table_names}

When giving your final answer, always show the SQL query first, then the results.

Use search_schema and search_business_rules to gather context.
Do not invent tables, columns, or business rules.
Before searching again, check if you already have the answer - don't repeat a search.
Once you have enough information, call generate_sql with one SELECT statement.
If generate_sql returns an error, fix the query and call it again.

If, after searching schema and business rules, you find no information
relevant to the question, do NOT guess, and do NOT browse tables blindly
with SELECT * to look for something that might fit. Instead, stop and tell
the user clearly that this information is not available in the database.
{history_block}"""

    # reset per-run state
    last_successful_query_result = None
    last_successful_sql = None
    call_counts.clear()
    last_result.clear()

    contents = [types.Content(role="user", parts=[types.Part(text=question)])]

    for i in range(max_iterations):
        try:
            response = call_gemini_with_retry(contents, system_instruction)
        except RuntimeError as e:
            print(f"\n--- Agent stopped: {e} ---")
            fallback_text = get_fallback_result_text()
            if fallback_text:
                print(fallback_text)
                final_answer = f"(agent stopped due to API failure)\n{fallback_text}"
            else:
                print("No partial results were available either.")
                final_answer = "(agent stopped due to API failure, no partial results available)"
            save_to_history(question, final_answer, last_successful_sql)
            return final_answer

        candidate = response.candidates[0]
        contents.append(candidate.content)

        if not response.function_calls:
            print("\n--- Final answer ---")
            print(response.text)
            save_to_history(question, response.text, last_successful_sql)
            return response.text

        response_parts = []
        for fc in response.function_calls:
            name = fc.name
            args = dict(fc.args)

            call_counts[name] = call_counts.get(name, 0) + 1
            if call_counts[name] > 2:
                result = f"Call limit reached for {name}. Use what you already have."
            else:
                func = TOOL_FUNCTIONS.get(name)
                if func:
                    try:
                        result = func(**args)
                    except Exception as e:
                        result = f"TOOL_ERROR: {name} failed unexpectedly ({e})"
                else:
                    result = f"Unknown function: {name}"

                if last_result.get(name) == result:
                    result += "\n(Note: same as your last search on this tool - no new info. Move on.)"
                last_result[name] = result

            print(f"\n--- Iteration {i+1}: {name}({args}) ---")
            print(f"Result: {result}")

            response_parts.append(
                types.Part.from_function_response(name=name, response={"result": result})
            )

        contents.append(types.Content(role="user", parts=response_parts))

    print(f"\nAgent did not finish within {max_iterations} iterations.")
    fallback_text = get_fallback_result_text()
    if fallback_text:
        print(fallback_text)
        final_answer = f"(agent did not finish - ran out of iterations)\n{fallback_text}"
    else:
        final_answer = "(agent did not finish - ran out of iterations, no partial results available)"
    save_to_history(question, final_answer, last_successful_sql)
    return final_answer


if __name__ == "__main__":
    try:
        question = input("Enter your question: ")
        run_agent(question)
    finally:
        conn.close()