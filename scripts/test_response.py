import sys
import os

# Add project root (parent of 'scripts/') to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.response_chain import rerank_jobs_with_llm
from app.job_retriever import JobRetriever

# Assume intent comes from CLI or parser
intent = {
    "role": "Data Analyst",
    "location": "New York",
    "salary": "80K",
    "domain": None,
    "remote": "no"
}

retriever = JobRetriever()
top30 = retriever.retrieve_jobs(intent, top_k=30)
filtered = retriever.filter_by_location(top30, intent["location"])

top10 = rerank_jobs_with_llm(intent, filtered)

# Pretty print
import json
print(json.dumps(top10, indent=2))
