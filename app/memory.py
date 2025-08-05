# app/memory.py

# Simple in-memory store for session tracking
user_sessions = {}

def get_session(user_id: str) -> dict:
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "intent": {
                "role": None,
                "location": None,
                "salary": None,
                "domain": None,
                "remote": None
            },
            "attempts": 0,
            "chat_history": []
        }
    return user_sessions[user_id]

def update_session(user_id: str, key: str, value):
    session = get_session(user_id)
    session[key] = value

def append_chat_history(user_id: str, role: str, message: str):
    session = get_session(user_id)
    session["chat_history"].append({"role": role, "message": message})

def reset_session(user_id: str):
    user_sessions.pop(user_id, None)
