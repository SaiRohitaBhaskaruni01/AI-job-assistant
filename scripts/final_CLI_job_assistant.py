import os
import json
from app.intent_parser import parse_intent
from app.followup import is_critical_missing, get_followup_question
from app.job_response_final import JobRetriever

def main():
    print("👋 Welcome to the Job Assistant!")
    print("Type your job search query below (e.g., 'Looking for a high-paying data analyst role in New York')")

    # 1. Take initial user input
    user_input = input("📝 Your Query: ").strip()

    # Save raw input
    os.makedirs("history", exist_ok=True)
    with open("history/user_input.txt", "w") as f:
        f.write(user_input)

    # 2. Initial intent parse
    intent = parse_intent(user_input)

    # 3. Multi-turn loop until required fields (role, location, salary) are filled
    while is_critical_missing(intent):
        followup_q = get_followup_question(intent)
        print("🤖", followup_q)
        user_reply = input("🧠 You: ").strip()
        followup_intent = parse_intent(user_reply)

        # Update intent with new values
        for key, val in followup_intent.items():
            if val:
                intent[key] = val

    print("\n✅ Final Parsed Intent:")
    print(json.dumps(intent, indent=2))

    # Save intent
    with open("history/final_intent.json", "w") as f:
        json.dump(intent, f, indent=2)

    # 4. Retrieve and rerank jobs, heart of the RAG system
    retriever = JobRetriever()
    results = retriever.retrieve_jobs(intent)

    print("\n🎯 Final Top Jobs")
    for i, job in enumerate(results, 1):
        print(f"{i}. {job}")

if __name__ == "__main__":
    main()
