# scripts/CLI_job_assistant.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from app.intent_parser import parse_intent
from app.followup import get_followup_question, is_critical_missing
from app.memory import get_session, reset_session
from app.job_retriever import JobRetriever
from app.response_chain import rerank_jobs

MAX_ATTEMPTS = 3
REQUIRED_FIELDS = {"role", "location", "salary"}

def merge_intents(existing: dict, new: dict) -> dict:
    return {k: new.get(k) or existing.get(k) for k in existing}

def run_cli_assistant(user_id="cli_user"):
    print("\U0001F50D Welcome to the Job Intent Assistant!\nDescribe the kind of job you're looking for.")
    reset_session(user_id)
    session = get_session(user_id)

    for attempt in range(MAX_ATTEMPTS):
        user_input = input("\nYou: ")
        new_intent = parse_intent(user_input)
        session["intent"] = merge_intents(session["intent"], new_intent)

        if not is_critical_missing(session["intent"]):
            break

        followup = get_followup_question(session["intent"])
        print(f"\n\U0001F916 Assistant: {followup}")

    if is_critical_missing(session["intent"]):
        print("\n❌ Sorry, your intent is still unclear after 3 tries. Please try again.")
        return

    print("\n✅ All required fields received!")
    print("\U0001F3AF Parsed Job Intent:")
    for key, value in session["intent"].items():
        print(f"   {key}: {value}")

    followup_optional = get_followup_question(session["intent"])
    if followup_optional:
        print(f"\n\U0001F4A1 Optional refinement: {followup_optional}")
    else:
        print("\n\U0001F44D Looks like we have all the information needed!")

    # NEW: Run job search pipeline
    print("\n🚀 Running job search and ranking pipeline...")
    
    try:
        # Step 1: Retrieve top 30 jobs
        print("🔍 Step 1: Retrieving similar jobs...")
        retriever = JobRetriever()
        top_30_jobs = retriever.retrieve_jobs(session["intent"], top_k=30)
        
        if not top_30_jobs:
            print("❌ No jobs found matching your criteria.")
            return
        
        print(f"✅ Found {len(top_30_jobs)} similar jobs")
        
        # Step 2: Rerank to get top 10
        print("🤖 Step 2: AI reranking to select best 10...")
        top_10_jobs = rerank_jobs(top_30_jobs, session["intent"])
        
        print(f"✅ Selected top {len(top_10_jobs)} jobs")
        
        # Step 3: Display results
        print("\n🏆 YOUR TOP 10 RECOMMENDED JOBS:")
        print("=" * 50)
        
        for job in top_10_jobs:
            rank = job.get('final_rank', job.get('rank', '?'))
            title = job.get('title', 'N/A')
            company = job.get('company', 'N/A')
            location = job.get('location', 'N/A')
            similarity = job.get('similarity_score', 'N/A')
            reason = job.get('selection_reason', 'Good match')
            
            print(f"\n#{rank}. {title}")
            print(f"    🏢 Company: {company}")
            print(f"    📍 Location: {location}")
            print(f"    📊 Similarity Score: {similarity}")
            print(f"    💡 Why: {reason}")
        
        print("\n" + "=" * 50)
        print("💡 TIP: Apply to these jobs in the order listed!")
        
    except Exception as e:
        print(f"\n❌ Error during job search: {str(e)}")
        print("Please check your setup:")
        print("1. ChromaDB exists at ./chroma_db")
        print("2. OPENAI_API_KEY is set")
        print("3. All dependencies are installed")

def test_with_sample_intent():
    """Quick test function with predefined intent"""
    print("🧪 TESTING WITH SAMPLE INTENT")
    print("=" * 40)
    
    sample_intent = {
        "role": "Software Engineer",
        "location": "San Francisco",
        "salary": "150000",
        "domain": "Technology",
        "remote": "yes"
    }
    
    print("Sample Intent:")
    for key, value in sample_intent.items():
        print(f"   {key}: {value}")
    
    try:
        print("\n🔍 Retrieving jobs...")
        retriever = JobRetriever()
        top_30 = retriever.retrieve_jobs(sample_intent, top_k=30)
        
        print(f"✅ Found {len(top_30)} jobs")
        
        print("\n🤖 Reranking...")
        top_10 = rerank_jobs(top_30, sample_intent)
        
        print(f"✅ Selected {len(top_10)} top jobs")
        
        print("\n🏆 TOP 5 RESULTS:")
        for job in top_10[:5]:
            print(f"{job.get('final_rank', job.get('rank'))}: {job['title']} at {job['company']}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    load_dotenv()
    
    # Ask user which mode to run
    print("Choose mode:")
    print("1. Interactive CLI")
    print("2. Test with sample data")
    
    choice = input("\nEnter 1 or 2: ").strip()
    
    if choice == "2":
        test_with_sample_intent()
    else:
        run_cli_assistant()