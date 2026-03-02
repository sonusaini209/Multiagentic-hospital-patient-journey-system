from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import re

from hospital_multiagent import ask, reset, setup_all_databases, graph

app = FastAPI(
    title="Hospital Patient Journey System",
    description="Multi-Agent Hospital Intelligence API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup databases on startup
@app.on_event("startup")
async def startup_event():
    setup_all_databases()

# Serve frontend
@app.get("/")
async def serve_ui():
    return FileResponse("index.html")

# ── API Routes ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    patient_id: str = ""
    remember: bool = True

class ResetRequest(BaseModel):
    pass

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/api/patients")
async def get_patients():
    return {
        "patients": [
            {"id": "P001", "name": "Rahul Singh",  "age": 34, "status": "Active",    "ward": "Cardiology Ward"},
            {"id": "P002", "name": "Anjali Patel", "age": 28, "status": "Discharged","ward": "General Ward A"},
            {"id": "P003", "name": "Vikram Rao",   "age": 52, "status": "Active",    "ward": "Orthopedic Ward"},
            {"id": "P004", "name": "Sonal Joshi",  "age": 45, "status": "Discharged","ward": "Neurology Ward"},
            {"id": "P005", "name": "Deepak Nair",  "age": 61, "status": "ICU",       "ward": "ICU"},
            {"id": "P006", "name": "Kavya Menon",  "age": 19, "status": "Active",    "ward": "General Ward B"},
            {"id": "P007", "name": "Harish Bhat",  "age": 38, "status": "Discharged","ward": "Oncology Ward"},
            {"id": "P008", "name": "Nisha Kapoor", "age": 55, "status": "Active",    "ward": "General Ward A"},
        ]
    }

@app.post("/api/query")
async def query_patient(request: QueryRequest):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        answer = ask(request.query, request.patient_id, request.remember)
        return {"status": "success", "answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
async def reset_conversation():
    try:
        reset()
        return {"status": "success", "message": "Conversation reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
