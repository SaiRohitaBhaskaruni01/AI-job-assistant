"""
Simple job reranker using OpenAI to select top 10 from top 30 retrieved jobs.
Compatible with existing rerank_prompt.txt format.
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

class SimpleReranker:
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Load your existing prompt template
        try:
            with open("prompts/rerank_prompt.txt", "r") as f:
                self.prompt_template = f.read()
        except FileNotFoundError:
            # Fallback prompt if file not found
            self.prompt_template = """You are a helpful job assistant AI.

Given the user's job-seeking intent:
{intent}

And a list of 30 job postings:
{jobs}

Please analyze and select the **top 10 most relevant jobs** for this user.

Be thoughtful in matching based on:
- Role/title relevance
- Location preference (on-site vs remote)
- Salary expectations
- Domain (if specified)

Return your results as a JSON array like this:

[
  {{
    "title": "...",
    "company": "...",
    "location": "...",
    "reason": "..."
  }}
]

Do NOT include any commentary, headers, or extra text — return only valid JSON."""

    def rerank_jobs(self, jobs: List[Dict[str, Any]], intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rerank top 30 jobs and return top 10 with reasons.
        """
        if len(jobs) > 30:
            jobs = jobs[:30]

        # Format intent for prompt
        intent_str = ""
        for key, value in intent.items():
            if value:
                intent_str += f"- {key}: {value}\n"

        # Format jobs for prompt
        jobs_str = ""
        for i, job in enumerate(jobs, 1):
            title = job.get('title', 'Unknown')
            company = job.get('company', 'Unknown')
            location = job.get('location', 'Unknown')
            similarity = job.get('similarity_score', 0)
            jobs_str += f"{i}. {title} at {company} ({location}) - Similarity: {similarity}\n"

        # Create prompt using your existing template
        prompt = self.prompt_template.format(
            intent=intent_str.strip(),
            jobs=jobs_str.strip()
        )

        try:
            print(f"🤖 Calling OpenAI to rank {len(jobs)} jobs...")
            response = self.model.invoke(prompt)
            
            content = response.content.strip()
            print(f"📝 LLM Response length: {len(content)} characters")
            print(f"📝 First 100 chars: {repr(content[:100])}")
            
            if not content:
                print("⚠️ Empty response from LLM, using fallback")
                return self._create_fallback_ranking(jobs)
            
            # Clean and extract JSON array from response
            # Remove any markdown formatting
            content = content.replace('```json', '').replace('```', '')
            
            # Find JSON array bounds
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                print("⚠️ No JSON array found, using fallback")
                print(f"Content preview: {content[:100]}...")
                return self._create_fallback_ranking(jobs)
            
            json_str = content[start_idx:end_idx]
            print(f"📝 Extracted JSON: {repr(json_str[:100])}")
            
            # Clean up common JSON formatting issues
            json_str = json_str.replace('\n', ' ').replace('\t', ' ')
            
            # Try to parse JSON
            try:
                rankings = json.loads(json_str)
                print(f"✅ JSON parsed successfully, got {len(rankings)} items")
            except json.JSONDecodeError as parse_error:
                print(f"⚠️ JSON parse error: {parse_error}")
                print(f"Problematic JSON: {repr(json_str[:200])}")
                return self._create_fallback_ranking(jobs)
            
            if not isinstance(rankings, list):
                print("⚠️ Response is not a list, using fallback")
                return self._create_fallback_ranking(jobs)
            
            # Match ranked jobs with original jobs based on title and company
            final_jobs = []
            for i, ranked_job in enumerate(rankings[:10], 1):
                if not isinstance(ranked_job, dict):
                    print(f"⚠️ Item {i} is not a dict: {type(ranked_job)}")
                    continue
                
                ranked_title = ranked_job.get("title", "").strip()
                ranked_company = ranked_job.get("company", "").strip()
                reason = ranked_job.get("reason", "Good match for your requirements")
                
                # Find matching job from original list
                matched_job = None
                for original_job in jobs:
                    orig_title = original_job.get('title', '').strip()
                    orig_company = original_job.get('company', '').strip()
                    
                    # Simple matching logic
                    if (ranked_title.lower() in orig_title.lower() or orig_title.lower() in ranked_title.lower()) and \
                       (ranked_company.lower() in orig_company.lower() or orig_company.lower() in ranked_company.lower()):
                        matched_job = original_job.copy()
                        break
                
                if matched_job:
                    matched_job["final_rank"] = i
                    matched_job["selection_reason"] = reason
                    final_jobs.append(matched_job)
                else:
                    # If no match found, create a placeholder (shouldn't happen often)
                    placeholder_job = {
                        "final_rank": i,
                        "job_id": f"ranked_{i}",
                        "title": ranked_title,
                        "company": ranked_company,
                        "location": ranked_job.get("location", "Unknown"),
                        "similarity_score": 0.0,
                        "selection_reason": reason
                    }
                    final_jobs.append(placeholder_job)
            
            print(f"✅ Successfully ranked {len(final_jobs)} jobs")
            return final_jobs[:10]
            
        except json.JSONDecodeError as e:
            print(f"❌ Final JSON parsing failed: {e}")
            print(f"Raw response preview: {content[:200]}...")
            return self._create_fallback_ranking(jobs)
        except Exception as e:
            print(f"❌ Reranking failed: {e}")
            return self._create_fallback_ranking(jobs)

    def _create_fallback_ranking(self, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create fallback ranking when LLM fails"""
        print("📋 Using fallback ranking by similarity score")
        fallback_jobs = []
        for i, job in enumerate(jobs[:10], 1):
            job_copy = job.copy()
            job_copy["final_rank"] = i
            job_copy["selection_reason"] = f"High similarity match (score: {job.get('similarity_score', 'N/A')})"
            fallback_jobs.append(job_copy)
        return fallback_jobs


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
        "role": "Software Engineer",
        "location": "Bay Area",
        "salary": "100K",
        "domain": None,
        "remote": "no"
    }
    
    # Sample jobs
    sample_jobs = [
        {
            "rank": 1,
            "job_id": "001",
            "title": "Software Engineer",
            "company": "Google",
            "location": "Mountain View, CA",
            "similarity_score": 0.95,
        },
        {
            "rank": 2,
            "job_id": "002", 
            "title": "Senior Software Engineer",
            "company": "Meta",
            "location": "Menlo Park, CA",
            "similarity_score": 0.92,
        },
        {
            "rank": 3,
            "job_id": "003",
            "title": "Data Scientist",
            "company": "Apple",
            "location": "Cupertino, CA",
            "similarity_score": 0.88,
        }
    ]
    
    print("Sample Intent:")
    for key, value in sample_intent.items():
        if value:
            print(f"   {key}: {value}")
    
    try:
        print("\n🤖 Running reranker...")
        ranked_jobs = rerank_jobs(sample_jobs, sample_intent)
        
        print(f"\n✅ Got {len(ranked_jobs)} ranked jobs")
        
        print("\n🏆 RERANKED RESULTS:")
        for job in ranked_jobs:
            rank = job.get('final_rank', '?')
            title = job.get('title', 'N/A')
            company = job.get('company', 'N/A')
            reason = job.get('selection_reason', 'No reason')
            
            print(f"   #{rank}. {title} at {company}")
            print(f"       💡 {reason}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_reranker()


# """
# Simple job reranker using OpenAI to select top 10 from top 30 retrieved jobs.
# """

# import os
# import json
# from typing import List, Dict, Any
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI

# load_dotenv()

# class SimpleReranker:
#     def __init__(self):
#         self.model = ChatOpenAI(
#             model="gpt-4o",
#             temperature=0.2,
#             api_key=os.getenv("OPENAI_API_KEY")
#         )

#     def rerank_jobs(self, jobs: List[Dict[str, Any]], intent: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """
#         Rerank top 30 jobs and return top 10 with reasons.
#         """
#         if len(jobs) > 30:
#             jobs = jobs[:30]

#         # Create prompt
#         intent_str = f"Role: {intent.get('role', 'N/A')}, Location: {intent.get('location', 'N/A')}, Salary: {intent.get('salary', 'N/A')}"
        
#         jobs_text = ""
#         for i, job in enumerate(jobs, 1):
#             jobs_text += f"{i}. {job['title']} at {job['company']} - {job['location']} (Score: {job['similarity_score']})\n"

#         prompt = f"""You are a job matching expert. Given the user's preferences: {intent_str}

# Here are 30 job candidates:
# {jobs_text}

# Select the TOP 10 jobs that best match the user's requirements. Consider role fit, location match, company quality, and overall alignment.

# Return ONLY a JSON array with this format:
# [
#   {{
#     "rank": 1,
#     "job_number": 5,
#     "reason": "Perfect role match and location fit"
#   }},
#   {{
#     "rank": 2, 
#     "job_number": 12,
#     "reason": "Strong company with relevant experience"
#   }}
# ]

# Return only the JSON array, no other text."""

#         try:
#             response = self.model.invoke(prompt)
#             rankings = json.loads(response.content)
            
#             # Build final results
#             final_jobs = []
#             for rank_info in rankings:
#                 job_idx = rank_info["job_number"] - 1
#                 if 0 <= job_idx < len(jobs):
#                     job = jobs[job_idx].copy()
#                     job["final_rank"] = rank_info["rank"]
#                     job["selection_reason"] = rank_info["reason"]
#                     final_jobs.append(job)
            
#             return final_jobs[:10]
            
#         except Exception as e:
#             print(f"❌ Reranking failed: {e}")
#             return jobs[:10]  # Fallback to top 10 by similarity


# def rerank_jobs(jobs: List[Dict[str, Any]], intent: Dict[str, Any]) -> List[Dict[str, Any]]:
#     """Simple function interface"""
#     reranker = SimpleReranker()
#     return reranker.rerank_jobs(jobs, intent)