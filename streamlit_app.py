import streamlit as st
from app.intent_parser import parse_intent
from app.followup import is_critical_missing, get_followup_question
from app.job_response_final import JobRetriever

st.set_page_config(page_title="Job Assistant AI", page_icon="🤖")

st.title("🤖 Job Assistant AI")
st.write("Tell me what kind of job you're looking for! I'll guide you and fetch top matching roles.")

# --- SESSION STATE ---
if "intent" not in st.session_state:
    st.session_state.intent = None

if "history" not in st.session_state:
    st.session_state.history = []

if "followup" not in st.session_state:
    st.session_state.followup = None

if "job_results" not in st.session_state:
    st.session_state.job_results = None

# --- INPUT FORM ---
with st.form("job_input_form", clear_on_submit=True):
    user_input = st.text_input("Your job search query", placeholder="e.g. Looking for a Data Analyst job in California")
    submitted = st.form_submit_button("Submit")

if submitted and user_input:
    # First user query or follow-up
    parsed = parse_intent(user_input)

    if st.session_state.intent is None:
        st.session_state.intent = parsed
    else:
        # Update missing fields
        for k, v in parsed.items():
            if v:
                st.session_state.intent[k] = v

    # Add to history
    st.session_state.history.append(("You", user_input))

    # Check if we still need follow-up
    if is_critical_missing(st.session_state.intent):
        followup = get_followup_question(st.session_state.intent)
        st.session_state.followup = followup
        st.session_state.history.append(("AI", followup))
    else:
        st.session_state.followup = None
        # Trigger job retrieval
        retriever = JobRetriever()
        st.session_state.job_results = retriever.retrieve_jobs(st.session_state.intent)

# --- CONVERSATION HISTORY ---
if st.session_state.history:
    with st.expander("🗨️ Conversation History"):
        for speaker, msg in st.session_state.history:
            st.markdown(f"**{speaker}:** {msg}")

# --- JOB RESULTS ---
if st.session_state.job_results:
    st.subheader("🎯 Top Job Matches")
    for i, summary in enumerate(st.session_state.job_results, 1):
        st.markdown(f"{i}. {summary}")
