"""
Streamlit UI for the AI Data Analyst Agent.

Run with:  streamlit run app.py

Wraps the existing investigate() loop - no agent logic lives here.
- Working memory is held in st.session_state (same idea as the terminal loop)
- Each tool call is shown live while the agent works, then kept in an
  expander under the answer so you can audit how it got there
- Sidebar shows today's API usage and a reset button
"""

import streamlit as st

# NOTE: change this import if your agent file has a different name.
from step6_DI_AgenticAI import investigate, FREE_TIER_DAILY_LIMIT
from step4_function_calling import get_api_usage_today, MODEL

st.set_page_config(page_title="AI Data Analyst Agent", page_icon="📊", layout="wide")

# ---- Session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []      # [{"role", "content", "steps"}]
if "working_memory" not in st.session_state:
    st.session_state.working_memory = None


def render_steps(steps):
    """Show a list of tool-call steps (used both live and for history replay)."""
    for step in steps:
        st.markdown(f"**Step {step['iteration']} — `{step['name']}`**")
        if step["name"] == "generate_sql" and "sql" in step["args"]:
            st.code(step["args"]["sql"], language="sql")
        else:
            st.json(step["args"], expanded=False)
        result = step["result"]
        st.caption("Result")
        st.code(result if len(result) <= 1500 else result[:1500] + " ...[truncated]", language="json")
        st.divider()


# ---- Sidebar ----
with st.sidebar:
    st.header("Agent")
    st.caption(f"Model: `{MODEL}`")

    used = get_api_usage_today()
    st.metric("API calls today", f"{used} / {FREE_TIER_DAILY_LIMIT}")
    st.progress(min(used / FREE_TIER_DAILY_LIMIT, 1.0))
    if FREE_TIER_DAILY_LIMIT - used <= 3:
        st.warning("Daily quota almost used up.")

    st.divider()
    memory_active = st.session_state.working_memory is not None
    st.caption("Working memory: " + ("active — follow-ups reuse prior findings" if memory_active else "empty"))
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.working_memory = None
        st.rerun()

# ---- Main ----
st.title("📊 AI Data Analyst Agent")
st.caption("Ask a business question. The agent investigates with tools and explains what it finds.")

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps"):
            with st.expander(f"How the agent got here ({len(msg['steps'])} tool calls)"):
                render_steps(msg["steps"])

question = st.chat_input("e.g. Why did average transaction value change from June to July 2026?")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "steps": []})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        collected_steps = []

        with st.status("Investigating...", expanded=True) as status:
            def on_step(iteration, name, args, result):
                step = {"iteration": iteration, "name": name, "args": args, "result": str(result)}
                collected_steps.append(step)
                render_steps([step])

            try:
                answer, st.session_state.working_memory = investigate(
                    question,
                    working_memory=st.session_state.working_memory,
                    on_step=on_step,
                )
                status.update(label=f"Done — {len(collected_steps)} tool calls", state="complete", expanded=False)
            except Exception as e:
                answer = f"Something went wrong while investigating: `{e}`"
                status.update(label="Failed", state="error")

        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "steps": collected_steps}
    )
    st.rerun()  # refresh sidebar usage counter

#How to Run the Streamlit App
# 1. Save this code in a file named `ui_app.py`.    
# 2. Open a terminal and navigate to the directory where `ui_app.py` is located.
# 3. Run the command: `streamlit run ui_app.py`.