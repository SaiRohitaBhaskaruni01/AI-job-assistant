# scripts/CLI_job_assistant.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.intent_parser import parse_intent
from app.followup import get_followup_question, is_critical_missing
from app.memory import get_session, reset_session

MAX_ATTEMPTS = 3

# ✅ UPDATED required fields
REQUIRED_FIELDS = {"role", "location", "salary"}

def merge_intents(existing: dict, new: dict) -> dict:
    return {k: new.get(k) or existing.get(k) for k in existing}

def run_cli_assistant(user_id="cli_user"):
    print("🔍 Welcome to the Job Intent Assistant!\nDescribe the kind of job you're looking for.")
    reset_session(user_id)
    session = get_session(user_id)

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
    print("🎯 Parsed Job Intent:\n")
    print(session["intent"])

    followup_optional = get_followup_question(session["intent"])
    if followup_optional:
        print(f"\n💡 Optional refinement: {followup_optional}")
    else:
        print("\n👍 Looks like we have all the information needed!")

if __name__ == "__main__":
    run_cli_assistant()
