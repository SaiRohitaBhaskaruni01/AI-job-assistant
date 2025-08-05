"""
Simple job reranker using OpenAI to select top 10 from top 30 retrieved jobs.
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

class SimpleReranker:
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def rerank_jobs(self, jobs: List[Dict[str, Any]], intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rerank top 30 jobs and return top 10 with reasons.
        """
        if len(jobs) > 30:
            jobs = jobs[:30]

        # Create prompt
        intent_str = f"Role: {intent.get('role', 'N/A')}, Location: {intent.get('location', 'N/A')}, Salary: {intent.get('salary', 'N/A')}"
        
        jobs_text = ""
        for i, job in enumerate(jobs, 1):
            jobs_text += f"{i}. {job['title']} at {job['company']} - {job['location']} (Score: {job['similarity_score']})\n"

        prompt = f"""You are a job matching expert. Given the user's preferences: {intent_str}

Here are 30 job candidates:
{jobs_text}

Select the TOP 10 jobs that best match the user's requirements. Consider role fit, location match, company quality, and overall alignment.

Return ONLY a JSON array with this format:
[
  {{
    "rank": 1,
    "job_number": 5,
    "reason": "Perfect role match and location fit"
  }},
  {{
    "rank": 2, 
    "job_number": 12,
    "reason": "Strong company with relevant experience"
  }}
]

Return only the JSON array, no other text."""

        try:
            response = self.model.invoke(prompt)
            rankings = json.loads(response.content)
            
            # Build final results
            final_jobs = []
            for rank_info in rankings:
                job_idx = rank_info["job_number"] - 1
                if 0 <= job_idx < len(jobs):
                    job = jobs[job_idx].copy()
                    job["final_rank"] = rank_info["rank"]
                    job["selection_reason"] = rank_info["reason"]
                    final_jobs.append(job)
            
            return final_jobs[:10]
            
        except Exception as e:
            print(f"❌ Reranking failed: {e}")
            return jobs[:10]  # Fallback to top 10 by similarity


def rerank_jobs(jobs: List[Dict[str, Any]], intent: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Simple function interface"""
    reranker = SimpleReranker()
    return reranker.rerank_jobs(jobs, intent)


def test_reranker():
    """Test the reranker with sample data"""
    print("🧪 TESTING RERANKER")
    print("=" * 30)
    
    # Sample intent
    sample_intent = {
        "role": "Data Scientist",
        "location": "New York",
        "salary": "120000",
        "domain": "AI/ML", 
        "remote": "yes"
    }
    
    # Sample jobs (simulating retriever output)
    sample_jobs = [
        {
            "rank": 1,
            "job_id": "001",
            "title": "Senior Data Scientist",
            "company": "Google",
            "location": "New York, NY",
            "similarity_score": 0.95,
            "description": "Machine learning and AI research"
        },
        {
            "rank": 2,
            "job_id": "002", 
            "title": "Data Analyst",
            "company": "Meta",
            "location": "Remote",
            "similarity_score": 0.92,
            "description": "Data analysis and visualization"
        },
        {
            "rank": 3,
            "job_id": "003",
            "title": "Machine Learning Engineer", 
            "company": "Netflix",
            "location": "New York, NY",
            "similarity_score": 0.90,
            "description": "Build ML models for recommendations"
        },
        {
            "rank": 4,
            "job_id": "004",
            "title": "Research Scientist",
            "company": "OpenAI",
            "location": "San Francisco, CA",
            "similarity_score": 0.88,
            "description": "AI research and development"
        },
        {
            "rank": 5,
            "job_id": "005",
            "title": "Data Engineer",
            "company": "Uber",
            "location": "New York, NY", 
            "similarity_score": 0.85,
            "description": "Build data pipelines and infrastructure"
        }
    ]
    
    print("Sample Intent:")
    for key, value in sample_intent.items():
        print(f"   {key}: {value}")
    
    print(f"\nSample Jobs ({len(sample_jobs)} jobs):")
    for job in sample_jobs:
        print(f"   {job['rank']}. {job['title']} at {job['company']} - {job['location']}")
    
    try:
        print("\n🤖 Running reranker...")
        ranked_jobs = rerank_jobs(sample_jobs, sample_intent)
        
        print(f"\n✅ Reranking successful! Got {len(ranked_jobs)} ranked jobs")
        
        print("\n🏆 RERANKED RESULTS:")
        for job in ranked_jobs:
            rank = job.get('final_rank', '?')
            title = job.get('title', 'N/A')
            company = job.get('company', 'N/A')
            reason = job.get('selection_reason', 'No reason provided')
            
            print(f"   #{rank}. {title} at {company}")
            print(f"       💡 Reason: {reason}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_reranker()