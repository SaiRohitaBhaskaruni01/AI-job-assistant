"""
Job Retriever Module
Performs semantic similarity search on ChromaDB vector store and returns top 30 job matches.
"""

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any


class JobRetriever:
    def __init__(self, chroma_db_path: str = "./chroma_db", collection_name: str = "job_postings"):
        """
        Initialize the job retriever with ChromaDB connection.
        """
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name

        # Must match the embedding model used in embedding_store.py
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Connect to existing ChromaDB store
        self.vectorstore = Chroma(
            persist_directory=self.chroma_db_path,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )

    def create_search_query(self, intent: Dict[str, Any]) -> str:
        """
        Create a formatted query string from the user intent.
        """
        parts = []

        if intent.get("role"):
            parts.append(f"Role: {intent['role']}")
        if intent.get("domain"):
            parts.append(f"Domain: {intent['domain']}")
        if intent.get("location"):
            parts.append(f"Location: {intent['location']}")
        if intent.get("remote") == "yes":
            parts.append("Remote friendly")
        if intent.get("salary"):
            parts.append(f"Salary: {intent['salary']}")

        query = " ".join(parts)
        print(f"🔍 Search Query: {query}")
        return query

    def retrieve_jobs(self, intent: Dict[str, Any], top_k: int = 30) -> List[Dict[str, Any]]:
        """
        Retrieve top-K jobs from vector DB using similarity search.
        """
        query = self.create_search_query(intent)
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)

        formatted = []
        for i, (doc, score) in enumerate(results):
            formatted.append({
                "rank": i + 1,
                "job_id": doc.metadata.get("id"),
                "title": doc.metadata.get("title"),
                "company": doc.metadata.get("company"),
                "location": doc.metadata.get("location"),
                "description": doc.page_content,  # this is title_normalized + description
                "similarity_score": round(score, 4),
                "metadata": doc.metadata
            })
        print(f"✅ Retrieved {len(formatted)} jobs")
        return formatted

    def filter_by_location(self, jobs: List[Dict[str, Any]], target_location: str) -> List[Dict[str, Any]]:
        """
        Optional strict location filter (after retrieval).
        """
        if not target_location:
            return jobs

        target = target_location.lower()
        filtered = [
            job for job in jobs
            if target in job.get("location", "").lower() or "remote" in job.get("location", "").lower()
        ]
        print(f"📍 Location filter: {len(filtered)} jobs match '{target_location}'")
        return filtered

    def get_retrieval_summary(self, jobs: List[Dict[str, Any]], intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return stats and insights from the retrieval.
        """
        if not jobs:
            return {"total_jobs": 0, "avg_similarity_score": 0, "top_companies": []}

        avg_score = sum(job["similarity_score"] for job in jobs) / len(jobs)
        top_companies = {}
        for job in jobs:
            c = job["company"]
            top_companies[c] = top_companies.get(c, 0) + 1

        sorted_companies = sorted(top_companies.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_jobs": len(jobs),
            "avg_similarity_score": round(avg_score, 4),
            "top_companies": [{"company": c, "job_count": n} for c, n in sorted_companies],
            "score_range": {
                "highest": jobs[0]["similarity_score"],
                "lowest": jobs[-1]["similarity_score"]
            },
            "search_intent": intent
        }


def main():
    """Demo: Run retrieval for sample intent"""
    intent = {
        "role": "Data Scientist",
        "location": "New York",
        "salary": "120000",
        "domain": "AI/ML",
        "remote": "no"
    }

    retriever = JobRetriever()
    top30 = retriever.retrieve_jobs(intent)
    filtered = retriever.filter_by_location(top30, intent["location"])
    summary = retriever.get_retrieval_summary(filtered, intent)

    print("\n📊 RETRIEVAL SUMMARY")
    print(summary)

    print("\n🎯 TOP 5 JOBS")
    for job in filtered[:5]:
        print(f"{job['rank']}. {job['title']} at {job['company']} ({job['location']}) — Score: {job['similarity_score']}")


if __name__ == "__main__":
    main()
