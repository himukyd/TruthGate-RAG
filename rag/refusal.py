from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.documents import Document
from rag.llm import get_local_llm
import json
import re

class ContextSufficiency(BaseModel):
    is_sufficient: bool = Field(description="True if the context contains enough information to answer the question accurately")
    missing_info: Optional[str] = Field(default="", description="Briefly describe what information is missing if not sufficient")

def get_context_grader():
    llm = get_local_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Task: Is the context sufficient to answer the question?
        Only answer in JSON format: {{"is_sufficient": true, "missing_info": ""}} or {{"is_sufficient": false, "missing_info": "description"}}.
        Be very lenient. If the context mentions the topic at all, return true."""),
        ("user", "Context: {context}\n\nQuestion: {query}")
    ])
    parser = JsonOutputParser(pydantic_object=ContextSufficiency)
    chain = prompt | llm | parser
    return chain

def check_context_sufficiency(query: str, documents: List[Document]) -> ContextSufficiency:
    llm = get_local_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Task: Evaluate if the provided context contains enough information to answer the user's question accurately.

        Guidelines:
        1. If the context contains information that helps answer the question, return is_sufficient: true.
        2. Even if the context only covers *part* of the question, return is_sufficient: true.
        3. Only return is_sufficient: false if the context is completely irrelevant or the question is about a feature that definitely doesn't exist in the docs.
        4. When in doubt, return is_sufficient: true and let the generation node decide.

        Examples of INSUFFICIENT Context:
        - Question: "How do I use mind-reading in FastAPI?"
          Context: "FastAPI is a modern web framework..."
          Output: {{"is_sufficient": false, "missing_info": "Mind-reading is not a real feature."}}

        Examples of SUFFICIENT Context:
        - Question: "How do I define a path parameter?"
          Context: "You can define path parameters using curly braces {{}} in the path."
          Output: {{"is_sufficient": true, "missing_info": ""}}

        - Question: "What are Pydantic models?"
          Context: "FastAPI uses Pydantic for data validation."
          Output: {{"is_sufficient": true, "missing_info": ""}}

        Output ONLY a JSON object."""),
        ("user", "Context: {context}\n\nQuestion: {query}")
    ])
    chain = prompt | llm | StrOutputParser()
    
    context_text = "\n\n".join([f"Source: {d.metadata.get('source')}\nContent: {d.page_content}" for d in documents])
    
    try:
        raw_result = chain.invoke({"context": context_text, "query": query})
        
        # Robust parsing for small models
        # First, try to find anything that looks like JSON
        json_match = re.search(r'\{.*\}', raw_result, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                result_dict = json.loads(json_str)
                # Ensure keys exist
                is_sufficient = result_dict.get("is_sufficient", False)
                missing_info = result_dict.get("missing_info", "")
                return ContextSufficiency(is_sufficient=is_sufficient, missing_info=missing_info)
            except json.JSONDecodeError:
                pass
        
        # Fallback: simple string matching if JSON fails
        if "is_sufficient\": true" in raw_result.lower() or "\"is_sufficient\":true" in raw_result.lower():
            return ContextSufficiency(is_sufficient=True, missing_info="")
            
        return ContextSufficiency(is_sufficient=False, missing_info="Context does not provide enough information.")
    except Exception as e:
        print(f"Error in context grading: {e}")
        return ContextSufficiency(is_sufficient=False, missing_info="Internal error during validation")
