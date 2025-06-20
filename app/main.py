from fastapi import FastAPI
from app.routes import diagnose

app = FastAPI()

app.include_router(diagnose.router, prefix="/diagnose", tags=["Diagnosis"])
