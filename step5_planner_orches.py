"""
Planning / Orchestration layer for the SQL Assistant.

Sits in front of the existing run_agent() loop from your main script.
Import run_agent, generate_sql, etc. from there — this file only adds
the decomposition step; none of your existing tool functions change.
"""

import json
import re

from google.genai import types
from step4_function_calling import client, MODEL, call_gemini_with_retry, run_agent, conn

PLANNER_MODEL_INSTRUCTION = """You are a query planner for a SQL assistant.

Given a user's question, decide if it needs to be broken into smaller
independent sub-questions before a SQL agent can answer it.

Rules:
- If the question is already a single, simple lookup, return a plan with
  exactly one step: the original question, unchanged.
- If the question compares periods, groups, or conditions (e.g. "this month
  vs last month", "flag regions where X"), break it into ordered steps where
  each step is answerable with its own SQL query, and later steps may refer
  to earlier results in plain English (e.g. "using the two totals above").
- Do not invent table or column names — steps should describe what data is
  needed in business terms, not SQL.
- Keep the plan as short as possible. Prefer 1-4 steps.

Return ONLY valid JSON, no markdown, no preamble, in this exact shape:
{"steps": ["first sub-question", "second sub-question", ...]}
"""


def plan_question(question, client, model):
    """Ask the model to decompose a question into ordered sub-questions.

    Falls back to a single-step plan (the original question) if the model
    output can't be parsed, or if the API call fails even after retries —
    planning should never be a hard failure point for the whole request.
    """
    try:
        response = call_gemini_with_retry(
            contents=[types.Content(role="user", parts=[types.Part(text=question)])],
            system_instruction=PLANNER_MODEL_INSTRUCTION,
            use_tools=False,
        )
        raw = response.text.strip()
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        steps = parsed.get("steps", [])
        if not steps:
            raise ValueError("empty steps list")
        print(f"[planner] Plan produced: {len(steps)} step(s)")
        return steps
    except Exception as e:
        print(f"[planner] Planning failed, falling back to single-step plan: {e}")
        print("[planner] Proceeding directly with the original question, unplanned.")
        return [question]


def run_plan(question, run_agent_fn, client, model):
    """Decompose a question, then run each step through the existing agent
    loop, passing prior results forward as plain-text context.

    run_agent_fn should be your existing run_agent() from the main script,
    but adapted to *return* the final answer string instead of only
    printing it (small change: replace `print(response.text)` with
    `return response.text` in the no-function-calls branch, and similarly
    return the fallback/timeout messages instead of just printing them).
    """
    steps = plan_question(question, client, model)

    if len(steps) == 1:
        # Simple case: no real decomposition needed, run as before.
        return run_agent_fn(steps[0])

    print(f"\n--- Plan ({len(steps)} steps) ---")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")

    results = []
    for i, step in enumerate(steps, 1):
        if results:
            context = "\n".join(
                f"Step {j+1} ('{steps[j]}') result: {r}"
                for j, r in enumerate(results)
            )
            step_question = (
                f"{step}\n\nContext from earlier steps in this plan:\n{context}"
            )
        else:
            step_question = step

        print(f"\n--- Executing step {i}/{len(steps)}: {step} ---")
        result = run_agent_fn(step_question)
        results.append(result)

        # If a step failed outright, later steps depend on it - stop now
        # instead of burning more API calls on doomed steps.
        if result.startswith("(agent stopped") or result.startswith("(agent did not finish"):
            print(f"\n[planner] Step {i} failed - aborting the remaining steps.")
            summary = "\n".join(
                f"Step {j+1} ({steps[j]}): {r}" for j, r in enumerate(results)
            )
            return f"Plan aborted at step {i}/{len(steps)}.\n\n{summary}"

    # Final synthesis: ask the model to combine step results into one answer
    # rather than just dumping the last step's output.
    synthesis_prompt = (
        f"Original question: {question}\n\n"
        + "\n".join(f"Step {i+1} result: {r}" for i, r in enumerate(results))
        + "\n\nCombine these into one clear final answer for the user."
    )
    try:
        final = call_gemini_with_retry(
            contents=[types.Content(role="user", parts=[types.Part(text=synthesis_prompt)])],
            system_instruction="You combine sub-question results into one clear final answer.",
            use_tools=False,
        )
        return final.text
    except Exception as e:
        print(f"[planner] Synthesis failed ({e}) - returning raw step results instead.")
        return "\n\n".join(
            f"Step {i+1} ({steps[i]}):\n{r}" for i, r in enumerate(results)
        )


if __name__ == "__main__":
    try:
        question = input("Enter your question: ")
        answer = run_plan(question, run_agent, client, MODEL)
        print("\n--- Final answer ---")
        print(answer)
    finally:
        conn.close()