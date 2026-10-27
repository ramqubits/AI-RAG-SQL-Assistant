"""
Milestone 1 - Autonomous Data Investigation Agent.

Different mode from planner.py: instead of deciding all steps upfront,
this agent decides its next move after seeing each result, and is
explicitly told to keep digging when a change looks significant
(e.g. "sales dropped 18%" should trigger "let's see which category
drove that", not stop at the headline number).

Reuses the low-level tools and plumbing from step4_function_calling.py;
only adds one new tool (analyze_results) and its own decision loop.
"""

import json

from google.genai import types
from step4_function_calling import (
    client, MODEL, conn, cursor, call_gemini_with_retry,
    search_schema, search_business_rules, generate_sql,
    save_to_history, load_recent_history, get_fallback_result_text,
    get_api_usage_today,
)

FREE_TIER_DAILY_LIMIT = 20  # adjust if your quota differs

SIGNIFICANT_CHANGE_PCT = 10  # a change at or above this is worth investigating further


# ---- New tool: analyze_results ----
# Deliberately deterministic (plain arithmetic), not another LLM call -
# exact math beats an LLM estimating a percentage.

def analyze_results(value_a, value_b, label_a="Value A", label_b="Value B"):
    try:
        value_a = float(value_a)
        value_b = float(value_b)
    except (TypeError, ValueError):
        return "ANALYZE_ERROR: value_a and value_b must be numeric"

    if value_b == 0:
        return json.dumps({
            "label_a": label_a, "label_b": label_b,
            "pct_diff": None,
            "note": f"{label_b} is zero, percent difference is undefined",
        })

    pct_diff = (value_a - value_b) / value_b * 100
    if pct_diff > 0:
        direction = f"{label_a} is higher than {label_b}"
    elif pct_diff < 0:
        direction = f"{label_a} is lower than {label_b}"
    else:
        direction = f"{label_a} and {label_b} are equal"

    return json.dumps({
        "label_a": label_a,
        "value_a": value_a,
        "label_b": label_b,
        "value_b": value_b,
        "pct_diff": round(pct_diff, 2),
        "direction": direction,
        "significant": abs(pct_diff) >= SIGNIFICANT_CHANGE_PCT,
    })


TOOL_FUNCTIONS = {
    "search_schema": search_schema,
    "search_business_rules": search_business_rules,
    "generate_sql": generate_sql,
    "analyze_results": analyze_results,
}

investigation_tools = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="search_schema",
        description="Search the database schema for tables, columns, and relationships.",
        parameters_json_schema={
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    ),
    types.FunctionDeclaration(
        name="search_business_rules",
        description="Search for business rules or definitions relevant to the question.",
        parameters_json_schema={
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    ),
    types.FunctionDeclaration(
        name="generate_sql",
        description="Run a single SELECT statement and return the results.",
        parameters_json_schema={
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        },
    ),
    types.FunctionDeclaration(
        name="analyze_results",
        description=(
            "Compare two numeric values and get the exact percent difference and "
            "whether it counts as significant. Works for ANY two-way comparison - "
            "two time periods, two customer segments, two categories, etc. Use this "
            "instead of estimating a percentage yourself."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "value_a": {"type": "number"},
                "value_b": {"type": "number"},
                "label_a": {"type": "string", "description": "What value_a represents, e.g. 'Active customers' or 'July 2026'"},
                "label_b": {"type": "string", "description": "What value_b represents, e.g. 'Inactive customers' or 'June 2026'"},
            },
            "required": ["value_a", "value_b"],
        },
    ),
])


INVESTIGATION_SYSTEM_PROMPT_TEMPLATE = """You are an AI Data Analyst Agent. Unlike a
simple SQL assistant, your job is to INVESTIGATE business questions, not just
answer them with one query.

Actual SQLite table names available: {table_names}

Process:
1. Use search_schema and search_business_rules to understand what data is available.
   Trust what they return - do NOT run exploratory queries like "SELECT * FROM table
   LIMIT 5" to peek at the data or re-discover columns you were already given.
2. Use generate_sql to get the headline number(s) the question is really asking about,
   directly - do not run a throwaway query first to check today's date or sample rows.
   SQLite's date('now') and standard date math can be used inline within your real
   query (e.g. strftime('%Y-%m', 'now')) without a separate lookup call.
2b. If a comparison comes back as 0 and 0 (or both suspiciously small/empty), verify
   your date logic ONCE with a single breakdown query (e.g. group by month), then
   stop checking. If the breakdown confirms there's genuinely no data in the periods
   being compared, report that directly as your finding - e.g. "there are no
   completed transactions in the current or prior month; the available data only
   covers [the months you found]" - rather than re-running date checks again.
2c. When a change is significant, one breakdown dimension (e.g. by category) may not
   be the real story. If the first breakdown shows the change is spread roughly
   evenly across every group (no single group stands out as the driver), try ONE
   other dimension your schema supports (e.g. by customer segment, by region, by
   time-of-month) before concluding. Cap yourself at 2 breakdown dimensions total -
   if neither explains it clearly, say so honestly in your final answer ("the change
   does not appear to be concentrated in any single [category/customer/etc.] - it
   looks broad-based") rather than forcing a causal story onto a weak signal.
3. Use analyze_results whenever you're comparing two numbers (a period-over-period
   change, a before/after) - never estimate a percentage yourself, always call the tool.
   This applies to EVERY percentage you state in your final answer, including
   secondary metrics (e.g. if you compare transaction counts AND total amounts, call
   analyze_results once for each pair). If you haven't called the tool for a pair of
   numbers, state the two numbers and the plain difference, but do not state a percentage.
4. If analyze_results reports "significant": true (change >= {threshold}%), do NOT
   stop there. Decide what breakdown would explain it (by category, region, product,
   customer segment - whatever the schema supports) and run another generate_sql +
   analyze_results on that breakdown before concluding.
5. Only stop investigating when you have enough evidence to explain WHY the number
   changed, not just THAT it changed. If you cannot find a breakdown that explains
   it, say so honestly rather than guessing.
6. Do not invent tables, columns, or business rules.
6b. "Amount" and "price" are NOT the same thing. Transactions.amount is what was
   actually paid in that specific transaction (can reflect quantity, discounts, or
   partial payments). Product.price is the product's catalog/list price. If a
   question asks specifically about a product's "price" (not what a transaction was
   worth), you must query Product.price directly - do not assume a transaction's
   amount equals the product's price.
7. Give your final answer as a short business explanation, not raw SQL/JSON - name
   the specific numbers and drivers you found.
{history_block}"""


MAX_WORKING_MEMORY_ENTRIES = 40  # cap on contents list length, oldest trimmed first


def investigate(question, max_iterations=10, working_memory=None, on_step=None):
    """Runs the investigation loop for one question and returns (answer, contents).

    working_memory: the `contents` list returned by a PRIOR call in this same
    session, or None to start fresh. Passing it back in lets a follow-up
    question ("now break that down by city instead") reuse everything the
    agent already found - tool calls, SQL results, reasoning - instead of
    re-running schema search and re-deriving numbers from scratch.

    This is in-session working memory only: it lives in the caller's
    variable, not on disk. For continuity ACROSS separate runs of this
    script, conversation_history (loaded below as history_block) is still
    the mechanism - the two are complementary, not duplicates.
    """
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = ", ".join(
        row[0] for row in cursor.fetchall() if row[0] != "conversation_history"
    )

    history_rows = load_recent_history()
    if history_rows:
        history_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in history_rows)
        history_block = f"\nRecent previous questions/answers (context only):\n{history_text}\n"
    else:
        history_block = ""

    system_instruction = INVESTIGATION_SYSTEM_PROMPT_TEMPLATE.format(
        table_names=table_names,
        threshold=SIGNIFICANT_CHANGE_PCT,
        history_block=history_block,
    )

    call_counts = {}
    last_result = {}
    contents = list(working_memory) if working_memory else []
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

    for i in range(max_iterations):
        try:
            response = call_gemini_with_retry(
                contents, system_instruction, tool_list=[investigation_tools],
            )
        except RuntimeError as e:
            print(f"\n--- Investigation stopped: {e} ---")
            fallback_text = get_fallback_result_text()
            error_detail = str(e)[:600]  # keep the real cause visible, but bounded
            final_answer = (
                f"(investigation stopped due to API failure)\n{fallback_text}\n\nError: {error_detail}"
                if fallback_text
                else f"(investigation stopped due to API failure, no partial results available)\n\nError: {error_detail}"
            )
            save_to_history(question, final_answer, None)
            return final_answer, contents

        candidate = response.candidates[0]
        contents.append(candidate.content)

        if not response.function_calls:
            save_to_history(question, response.text, None)
            if len(contents) > MAX_WORKING_MEMORY_ENTRIES:
                contents = contents[-MAX_WORKING_MEMORY_ENTRIES:]
            return response.text, contents

        response_parts = []
        for fc in response.function_calls:
            name = fc.name
            args = dict(fc.args)

            call_counts[name] = call_counts.get(name, 0) + 1
            if call_counts[name] > 6:
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
                    result += "\n(Note: same as your last call on this tool - no new info. Move on.)"
                last_result[name] = result

            print(f"\n--- Iteration {i+1}: {name}({args}) ---")
            print(f"Result: {result}")
            if on_step:
                on_step(i + 1, name, args, result)

            response_parts.append(
                types.Part.from_function_response(name=name, response={"result": result})
            )

        contents.append(types.Content(role="user", parts=response_parts))

    print(f"\nInvestigation did not finish within {max_iterations} iterations.")
    fallback_text = get_fallback_result_text()
    final_answer = (
        f"(investigation did not finish - ran out of iterations)\n{fallback_text}"
        if fallback_text
        else "(investigation did not finish - ran out of iterations, no partial results available)"
    )
    save_to_history(question, final_answer, None)
    return final_answer, contents


if __name__ == "__main__":
    print("AI Data Analyst Agent. Type 'reset' to clear working memory, 'exit' to quit.")
    working_memory = None
    try:
        while True:
            question = input("\nEnter your business question: ").strip()
            if question.lower() in ("exit", "quit"):
                break
            if question.lower() == "reset":
                working_memory = None
                print("Working memory cleared.")
                continue
            if not question:
                continue

            answer, working_memory = investigate(question, working_memory=working_memory)
            print("\n--- Final answer ---")
            print(answer)

            used = get_api_usage_today()
            remaining = FREE_TIER_DAILY_LIMIT - used
            print(f"\n[usage] API calls today: {used}/{FREE_TIER_DAILY_LIMIT} "
                  f"({'OK' if remaining > 3 else 'LOW - ' + str(remaining) + ' left'})")
    finally:
        conn.close()