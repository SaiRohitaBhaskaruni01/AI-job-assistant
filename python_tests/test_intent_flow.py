# tests/test_intent_flow.py


# python_tests/test_intent_flow.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.intent_handler import handle_user_query
from app.memory import reset_session


# Simulated user interaction
def test_full_intent_flow():
    user_id = "test_user"
    reset_session(user_id)

    # ROUND 1: Missing required fields (e.g., domain + location)
    input1 = "Looking for a data analyst role"
    result1 = handle_user_query(input1, user_id)
    print("\n--- Round 1 ---")
    print("Type:", result1["type"])
    print("Follow-up:", result1.get("followup"))
    print("Intent:", result1["intent"])

    # ROUND 2: Add domain only (still missing location)
    input2 = "Preferably at a fintech company"
    result2 = handle_user_query(input2, user_id)
    print("\n--- Round 2 ---")
    print("Type:", result2["type"])
    print("Follow-up:", result2.get("followup"))
    print("Intent:", result2["intent"])

    # ROUND 3: Still missing location
    input3 = "I want something remote"
    result3 = handle_user_query(input3, user_id)
    print("\n--- Round 3 ---")
    print("Type:", result3["type"])
    print("Follow-up:", result3.get("followup"))
    print("Intent:", result3["intent"])

    # ROUND 4: Still not clear → should fail after max attempts
    input4 = "Salary doesn’t matter"
    result4 = handle_user_query(input4, user_id)
    print("\n--- Round 4 ---")
    print("Type:", result4["type"])
    print("Message:", result4.get("message"))
    print("Intent:", result4["intent"])

def test_successful_flow_with_optional_missing():
    user_id = "test_success_user"
    reset_session(user_id)

    # Required fields are present
    input1 = "Looking for a data analyst role in New York in a startup"
    result = handle_user_query(input1, user_id)
    print("\n--- Complete Intent ---")
    print("Type:", result["type"])
    print("Message:", result.get("message"))
    print("Follow-up (optional):", result.get("followup"))
    print("Intent:", result["intent"])

if __name__ == "__main__":
    test_full_intent_flow()
    test_successful_flow_with_optional_missing()
