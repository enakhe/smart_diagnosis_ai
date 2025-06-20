from fastapi import APIRouter
from app.schemas.diagnose import DiagnosisRequest, DiagnosisResponse
from app.models.predictor import predict_disease

router = APIRouter()

@router.post("/", response_model=DiagnosisResponse)
async def diagnose(request: DiagnosisRequest):
    return predict_disease(request)