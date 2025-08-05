import sys
import os

# Enable absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Phase 1: Intent Parsing
from app.intent_parser import parse_intent
from app.followup import get_followup_question, is_critical_missing
from app.memory import get_session, reset_session

# Phase 2: Retrieval
from app.job_retriever import JobRetriever

MAX_ATTEMPTS = 3
REQUIRED_FIELDS = {"role", "location", "salary"}


def merge_intents(existing: dict, new: dict) -> dict:
    return {k: new.get(k) or existing.get(k) for k in existing}


def run_cli_assistant(user_id="cli_user"):
    print("🔍 Welcome to the Job Intent Assistant!\nDescribe the kind of job you're looking for.")
    reset_session(user_id)
    session = get_session(user_id)

    # Step 1: Multi-turn intent collection
    for attempt in range(MAX_ATTEMPTS):
        user_input = input("\nYou: ")
        new_intent = parse_intent(user_input)
        session["intent"] = merge_intents(session["intent"], new_intent)

        if not is_critical_missing(session["intent"]):
            break

        followup = get_followup_question(session["intent"])
        print(f"\n🤖 Assistant: {followup}")

    if is_critical_missing(session["intent"]):
        print("\n❌ Sorry, your intent is still unclear after 3 tries. Please try again.")
        return

    print("\n✅ All required fields received!")
    print("🎯 Parsed Job Intent:")
    print(session["intent"])

    # Optional refinement
    followup_optional = get_followup_question(session["intent"])
    if followup_optional:
        print(f"\n💡 Optional refinement: {followup_optional}")
    else:
        print("\n👍 Looks like we have all the information needed!")

    # Step 2: Perform retrieval
    import json

    # Step 2: Perform retrieval
    print("\n🔎 Retrieving top 30 matching jobs...\n")
    retriever = JobRetriever()
    top_30 = retriever.retrieve_jobs(session["intent"], top_k=30)
    filtered = retriever.filter_by_location(top_30, session["intent"]["location"])

    print(f"\n📊 Retrieved {len(filtered)} jobs matching location '{session['intent']['location']}'")

    print("\n🎯 Structured Results:")
    print(json.dumps(filtered, indent=2))


    print("\n🎯 TOP 10 JOBS:")
    for job in filtered[:10]:
        print(f"{job['rank']}. {job['title']} at {job['company']} ({job['location']}) — Score: {job['similarity_score']}")

if __name__ == "__main__":
    run_cli_assistant()
