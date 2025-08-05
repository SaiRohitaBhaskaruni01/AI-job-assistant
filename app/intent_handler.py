# app/intent_handler.py - this orcheastrates everything

from app.intent_parser import parse_intent
from app.followup import get_followup_question, is_critical_missing
from app.memory import get_session, update_session, append_chat_history

MAX_ATTEMPTS = 3

def handle_user_query(user_input: str, user_id: str = "default_user") -> dict:
    session = get_session(user_id)

    # Update chat history
    append_chat_history(user_id, "user", user_input)

    # Parse current input
    new_intent = parse_intent(user_input)

    # Merge with existing intent
    for k, v in new_intent.items():
        if v:
            session["intent"][k] = v

    # Check if still missing required fields
    if is_critical_missing(session["intent"]):
        session["attempts"] += 1

        if session["attempts"] >= MAX_ATTEMPTS:
            return {
                "type": "error",
                "message": "I'm sorry, your request is still unclear after 3 tries. Please rephrase or try again.",
                "intent": session["intent"]
            }

        followup = get_followup_question(session["intent"])
        append_chat_history(user_id, "assistant", followup)
        return {
            "type": "followup_required",
            "followup": followup,
            "intent": session["intent"]
        }

    # Proceed — intent is usable
    followup_optional = get_followup_question(session["intent"])
    message = None
    if followup_optional:
        message = "I have found some matches, but I can refine them further with more information."

    return {
        "type": "intent_complete",
        "intent": session["intent"],
        "followup": followup_optional,
        "message": message,
        "chat_history": session["chat_history"]
    }
