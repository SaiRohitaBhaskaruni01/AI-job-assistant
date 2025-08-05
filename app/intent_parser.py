import os
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define structured schema
class JobIntent(BaseModel):
    role: str | None = Field(..., description="Job title or position")
    location: str | None = Field(..., description="Desired job location (city, remote, etc.)")
    salary: str | None = Field(..., description="Expected salary (e.g., 100000 or '100k')")
    domain: str | None = Field(..., description="Industry/domain like healthcare, startup, etc.")
    remote: str | None = Field(..., description="'yes', 'no', or 'hybrid'")

# Create output parser
parser = JsonOutputParser(pydantic_object=JobIntent)

# Load prompt from file
with open("prompts/intent_prompt.txt", "r") as f:
    prompt_template = f.read()

# Build prompt
prompt = PromptTemplate.from_template(
    prompt_template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Initialize OpenAI GPT-4o
model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

# Compose LangChain pipeline
chain = prompt | model | parser

# app/intent_parser.py

from pydantic import BaseModel, Field

# (your JobIntent model definition remains unchanged...)

def parse_intent(user_input: str) -> dict:
    """
    Extract job intent using LLM. Ensures all fields are returned, even if null.
    """
    try:
        result = chain.invoke({"user_input": user_input})
        
        # ✅ Normalize result to always include all fields
        full_intent = {
            "role": result.get("role"),
            "location": result.get("location"),
            "salary": result.get("salary"),
            "domain": result.get("domain"),
            "remote": result.get("remote"),
        }

        return full_intent

    except Exception as e:
        print("❌ Intent parsing failed:", e)
        return {
            "role": None,
            "location": None,
            "salary": None,
            "domain": None,
            "remote": None,
        }
# Test
if __name__ == "__main__":
    query = "Looking for a remote data scientist role in a healthcare startup"
    print(parse_intent(query))
