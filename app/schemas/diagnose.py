from pydantic import BaseModel
from typing import List, Optional

class DiagnosisRequest(BaseModel):
    symptoms: str
    age: Optional[int] = None
    gender: Optional[str] = None
    history: Optional[str] = None

class DiagnosisResponse(BaseModel):
    predictions: List[dict]
    advice: str
