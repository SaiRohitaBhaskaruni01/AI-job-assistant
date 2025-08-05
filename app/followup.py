# app/followup.py

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ UPDATED required fields
REQUIRED_FIELDS = {"role", "location", "salary"}

model = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=OPENAI_API_KEY)

with open("prompts/followup_prompt.txt", "r") as f:
    followup_prompt_template = f.read()

prompt = PromptTemplate.from_template(followup_prompt_template)
chain = prompt | model

def get_missing_fields(intent: dict):
    return [k for k, v in intent.items() if v is None]

def is_critical_missing(intent: dict) -> bool:
    return any(field in REQUIRED_FIELDS for field, val in intent.items() if val is None)

def get_followup_question(intent: dict) -> str | None:
    missing_fields = get_missing_fields(intent)
    if not missing_fields:
        return None

    try:
        result = chain.invoke({
            "missing_fields": ", ".join(missing_fields),
            "current_intent": intent
        })
        return result.content.strip()
    except Exception as e:
        print("❌ Follow-up generation failed:", e)
        return None
