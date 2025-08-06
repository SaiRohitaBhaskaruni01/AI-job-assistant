
# 🧠 Job Assistant AI (RAG-based)

A conversational job assistant that helps users find the most relevant job opportunities through natural language input. It leverages **GPT-4o**, **LangChain**, and **ChromaDB** to parse user intent, retrieve similar job postings, and rerank them using an LLM with explanations.

---

## 🧩 Features

- Multi-turn **Conversational AI** using GPT-4o.
- Parses natural language job queries and follows up for missing fields.
- Embeds and stores jobs in **ChromaDB** using sentence-transformers.
- Retrieves and reranks top matching jobs based on parsed intent.
- Presents top job matches with natural language explanations.
- Streamlit frontend and CLI support.

---

## ⚙️ Project Structure

```
new_job_assistant/
│
├── app/
│   ├── intent_parser.py         # Parses user query to structured job intent
│   ├── followup.py              # Generates follow-up questions using GPT-4o
│   ├── embedding_store.py       # Embeds jobs and stores in ChromaDB
│   ├── job_repsonser_final.py   # Full RAG pipeline: retrieve → rerank → respond
│   ├── job_retriever.py         # Only retrieves top 30 based on intent
│   └── memory.py                # (Optional) session/memory logic
│
├── prompts/                     # Prompt templates for each stage
│   ├── intent_prompt.txt
│   ├── followup_prompt.txt
│   ├── rerank_prompt.txt
│   └── new_rerank_prompt.txt
│
├── data/                        # CSV job datasets (cleaned + minimal)
│   ├── clean_jobs.csv
│   └── jobs_minimal.csv
│
├── history/                     # Saved inputs & final intent
│   ├── user_input.txt
│   └── final_intent.json
│
├── scripts/                     # CLI runners for debugging or backend
│   ├── CLI_job_assistant.py
│   ├── final_CLI_job_assistant.py
│   └── test_response.py
│
├── streamlit_app.py             # Frontend app
├── preprocess_data.py           # Script to clean/prepare job CSV
├── requirements.txt
└── .env                         # OpenAI key etc.
```

---

## 🧠 How this is a RAG system

This project is a **Retrieval-Augmented Generation (RAG)** setup applied to job search.

| Component     | Implementation                                                                 |
|--------------|----------------------------------------------------------------------------------|
| **Retriever**  | ChromaDB + sentence-transformers/all-MiniLM-L6-v2 embedding for job documents |
| **Generator**  | GPT-4o via LangChain for reranking and natural explanations                    |
| **Bridge**     | Top 30 retrieved jobs passed into GPT for final response with reasoning        |

This hybrid setup gives you relevance (via vector search) + nuance (via LLM reasoning).

---

## 🛠 Tech Stack & Why

| Technology        | Purpose                                                             |
|------------------|---------------------------------------------------------------------|
| **GPT-4o**        | Natural language understanding, follow-ups, reranking explanation  |
| **LangChain**     | LLM chaining and prompt management                                  |
| **ChromaDB**      | Vector DB to store + retrieve job embeddings                        |
| **sentence-transformers** | Convert jobs to vector form for semantic search            |
| **Streamlit**     | Interactive frontend                                                 |
| **Python**        | Backend and CLI logic                                                |

---

## ✅ Example Workflow

1. User: “Looking for a high-paying data analyst job in California”  
2. Assistant asks follow-up: “What salary range are you expecting?”  
3. Once required fields (role, location, salary) are captured, the top 30 jobs are retrieved from ChromaDB.
4. GPT-4o reranks jobs based on semantic fit + intent match.
5. Final top 10 jobs shown with explanation.

---

## 📥 Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env

# Build vector store
python app/embedding_store.py

# Run CLI version
PYTHONPATH=. python scripts/final_CLI_job_assistant.py

# Run Streamlit app
streamlit run streamlit_app.py
```

---

## 📌 Coming Soon

- Filter by company size, remote/hybrid
- Save favorite jobs
- Resume parsing for personalization
- Export job results to PDF/CSV

---

## 🧑‍💻 Author : Sai Rohita Bhaskaruni

