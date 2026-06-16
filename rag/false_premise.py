from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
import re

load_dotenv()

class FalsePremiseResult(BaseModel):
    is_false_premise: bool = Field(description="Whether the question contains a factually incorrect assumption about the project")
    explanation: str = Field(description="A brief explanation of why the premise is false, or empty if it's a valid question")

from rag.llm import get_local_llm

def get_false_premise_detector():
    llm = get_local_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert on FastAPI. Your task is to detect if a user's question contains a 'false premise'.

        A false premise is an assumption in the question that is factually wrong regarding FastAPI's capabilities, architecture, or ecosystem.

        Examples:
        - "Why does FastAPI require Java?" (is_false_premise: true)
        - "What is a path parameter?" (is_false_premise: false)

        Output ONLY a JSON object with keys: {{"is_false_premise": boolean}} and {{"explanation": string}}.
        Example: {{"is_false_premise": true, "explanation": "FastAPI is Python-based, not Java."}}"""),
        ("user", "Question: {query}")
    ])

    parser = JsonOutputParser(pydantic_object=FalsePremiseResult)
    chain = prompt | llm | parser
    return chain

def detect_false_premise(query: str) -> FalsePremiseResult:
    llm = get_local_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a technical validator for FastAPI. Your task is to detect if a question contains a 'false premise'—a specific technical misconception about how FastAPI works.

        A false premise is ONLY when the user assumes a technical fact that is fundamentally wrong (e.g., wrong programming language, fake mandatory features, non-existent pricing).

        Examples of False Premises (Return true):
        - "Why does FastAPI require Java?"
        - "How do I pay the monthly subscription for FastAPI?"

        Examples of VALID Questions (Return false):
        - "How do I define a path parameter?"
        - "What is the difference between Query and Path parameters?"
        - "How to use Pydantic models?"
        - "Can I use async def for path operations?"

        A question is only a false premise if it makes a CLEAR and SPECIFIC technical claim that is factually wrong. If you are unsure, return is_false_premise: false.

        Output ONLY a JSON object."""),
        ("user", "Question: {query}")
    ])
    chain = prompt | llm | StrOutputParser()

    try:
        raw_result = chain.invoke({"query": query})
        
        # Robust parsing for small models
        # First, try to find anything that looks like JSON in a potentially echoed output
        json_match = re.search(r'\{.*\}', raw_result, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                # Basic cleanup if there's junk around the JSON
                if "{" in json_str and "}" in json_str:
                    json_str = json_str[json_str.find("{"):json_str.rfind("}")+1]
                
                result_dict = json.loads(json_str)
                # Handle common key variations
                is_fp = result_dict.get("is_false_premise", result_dict.get("has_false_premise", False))
                explanation = result_dict.get("explanation", "")
                return FalsePremiseResult(is_false_premise=is_fp, explanation=explanation)
            except json.JSONDecodeError:
                pass
        
        # Fallback: simple string matching in the full text
        lower_raw = raw_result.lower()
        if "is_false_premise\": true" in lower_raw or "\"is_false_premise\":true" in lower_raw or "is_false_premise: true" in lower_raw:
            return FalsePremiseResult(is_false_premise=True, explanation="The question contains a false assumption about FastAPI.")
            
        return FalsePremiseResult(is_false_premise=False, explanation="")
    except Exception as e:
        print(f"Error in false premise detection: {e}")
        return FalsePremiseResult(is_false_premise=False, explanation="")
