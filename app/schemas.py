from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., example="How do I create a path parameter in FastAPI?")

class Citation(BaseModel):
    source: str
    content: str
    section: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation] = []
    status: str = Field(..., description="Success, Refused, or False-Premise")
    cost: float
    latency: float

class EvalRequest(BaseModel):
    questions_path: str = "evaluation/questions.json"
