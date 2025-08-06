"""
Job Retriever Module
Performs semantic similarity search on ChromaDB vector store and returns top 30 job matches.
Focuses on role (parsed title) and full user query (which includes required fields).
Then passes results through LLM for reranking with natural language recommendations.
"""

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import json
import os
import re


class JobRetriever:
    def __init__(self, chroma_db_path: str = "./chroma_db", collection_name: str = "job_postings"):
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = Chroma(
            persist_directory=self.chroma_db_path,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )

        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    def create_search_query(self, intent: Dict[str, Any]) -> str:
        role = intent.get("role", "")
        raw_query = intent.get("raw_query", "")
        query = f"{role} - {raw_query}".strip()
        print(f"🔍 Search Query: {query}")
        return query

    def retrieve_jobs(self, intent: Dict[str, Any], top_k: int = 30) -> List[str]:
        query = self.create_search_query(intent)
        results = self.vectorstore.similarity_search_with_score(query, k=top_k * 2)

        seen = set()
        deduped = []

        for doc, score in results:
            key = (doc.metadata.get("title"), doc.metadata.get("company"), doc.metadata.get("location"))
            if key in seen:
                continue
            seen.add(key)

            deduped.append({
                "title": doc.metadata.get("title"),
                "company": doc.metadata.get("company"),
                "location": doc.metadata.get("location"),
                "short_description": doc.page_content[:500],
                "similarity_score": round(score, 4)
            })

            if len(deduped) >= top_k:
                break

        print(f"✅ Retrieved {len(deduped)} jobs")

        return self.rerank_with_llm(deduped, intent)

    def rerank_with_llm(self, jobs: List[Dict[str, Any]], intent: Dict[str, Any]) -> List[str]:
        prompt_template_path = os.path.join(os.path.dirname(__file__), "../prompts/new_rerank_prompt.txt")
        with open(prompt_template_path, "r", encoding="utf-8") as f:
            template = f.read()

        compact_jobs = [
            {
                "title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "description": job["short_description"],
                "score": job["similarity_score"]
            }
            for job in jobs
        ]

        prompt = template.format(
            intent=json.dumps(intent, indent=2),
            jobs=json.dumps(compact_jobs, indent=2)
        )

        print("\n📤 Sending prompt to LLM...")
        response = self.llm.invoke(prompt)
        raw_output = response.content.strip()

        # Clean up triple-backtick JSON formatting
        if raw_output.startswith("```json"):
            raw_output = raw_output[len("```json"):].strip()
        elif raw_output.startswith("```"):
            raw_output = raw_output[len("```"):].strip()

        if raw_output.endswith("```"):
            raw_output = raw_output[:-3].strip()

        try:
            parsed = json.loads(raw_output)  # Expecting list of strings
            return parsed
        except Exception as e:
            print("❌ Failed to parse LLM response:", e)
            print("Raw output:", raw_output)
            return []


if __name__ == "__main__":
    intent = {
        "role": "Data Analyst",
        "location": "New York",
        "salary": "120000",
        "domain": "Analytics",
        "remote": "no",
        "raw_query": "Looking for a high-paying analytics role in New York. Prefer Data Analyst, not remote."
    }

    retriever = JobRetriever()
    top_jobs = retriever.retrieve_jobs(intent)

    print("\n🎯 Final Top Jobs")
    for i, job in enumerate(top_jobs, 1):
        print(f"{i}. {job}")


# """
# Job Retriever Module
# Performs semantic similarity search on ChromaDB vector store and returns top 30 job matches.
# Focuses on role (parsed title) and full user query (which includes required fields).
# Then passes results through LLM for reranking with reasoning.
# """

# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import ChatOpenAI
# from typing import List, Dict, Any
# import json
# import os
# from collections import OrderedDict


# class JobRetriever:
#     def __init__(self, chroma_db_path: str = "./chroma_db", collection_name: str = "job_postings"):
#         self.chroma_db_path = chroma_db_path
#         self.collection_name = collection_name

#         self.embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2"
#         )

#         self.vectorstore = Chroma(
#             persist_directory=self.chroma_db_path,
#             embedding_function=self.embeddings,
#             collection_name=self.collection_name
#         )

#         self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

#     def create_search_query(self, intent: Dict[str, Any]) -> str:
#         role = intent.get("role", "")
#         raw_query = intent.get("raw_query", "")
#         query = f"{role} - {raw_query}".strip()
#         print(f"🔍 Search Query: {query}")
#         return query

#     def retrieve_jobs(self, intent: Dict[str, Any], top_k: int = 30) -> List[Dict[str, Any]]:
#         query = self.create_search_query(intent)
#         results = self.vectorstore.similarity_search_with_score(query, k=top_k * 2)

#         seen = set()
#         deduped = []

#         for doc, score in results:
#             key = (doc.metadata.get("title"), doc.metadata.get("company"), doc.metadata.get("location"))
#             if key in seen:
#                 continue
#             seen.add(key)

#             deduped.append({
#                 "title": doc.metadata.get("title"),
#                 "company": doc.metadata.get("company"),
#                 "location": doc.metadata.get("location"),
#                 "short_description": doc.page_content[:500],  # truncate to 500 chars
#                 "similarity_score": round(score, 4)
#             })

#             if len(deduped) >= top_k:
#                 break

#         print(f"✅ Retrieved {len(deduped)} jobs")

#         return self.rerank_with_llm(deduped, intent)

#     def rerank_with_llm(self, jobs: List[Dict[str, Any]], intent: Dict[str, Any]) -> List[Dict[str, Any]]:
#         prompt_template_path = os.path.join(os.path.dirname(__file__), "../prompts/rerank_prompt.txt")
#         with open(prompt_template_path, "r", encoding="utf-8") as f:
#             template = f.read()

#         compact_jobs = [
#             {
#                 "title": job["title"],
#                 "company": job["company"],
#                 "location": job["location"],
#                 "description": job["short_description"],
#                 "score": job["similarity_score"]
#             }
#             for job in jobs
#         ]

#         prompt = template.format(
#             intent=json.dumps(intent, indent=2),
#             jobs=json.dumps(compact_jobs, indent=2)
#         )

#         print("\n📤 Sending prompt to LLM...")
#         response = self.llm.invoke(prompt)

#         try:
#             parsed = json.loads(response.content)
#             return parsed
#         except Exception as e:
#             print("❌ Failed to parse LLM response:", e)
#             print("Raw output:", response.content)
#             return []


# if __name__ == "__main__":
#     intent = {
#         "role": "Data Analyst",
#         "location": "New York",
#         "salary": "120000",
#         "domain": "Analytics",
#         "remote": "no",
#         "raw_query": "Looking for a high-paying analytics role in New York. Prefer Data Analyst, not remote."
#     }

#     retriever = JobRetriever()
#     top_jobs = retriever.retrieve_jobs(intent)

#     print("\n🎯 Final Top Jobs with Reasons")
#     for i, job in enumerate(top_jobs, 1):
#         print(f"{i}. {job['title']} at {job['company']} ({job['location']}) — {job['reason']}")
