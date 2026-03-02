from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
import sys

app = FastAPI(title="Hospital Patient Journey System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Vercel uses /tmp for writable storage ─────────────────────
os.environ["HOSPITAL_DB_DIR"] = "/tmp/hospital_databases"

# ── Lazy imports — don't crash on cold start ──────────────────
_initialized = False

def get_agent():
    global _initialized
    if not _initialized:
        from hospital_multiagent import setup_all_databases
        setup_all_databases()
        _initialized = True
    from hospital_multiagent import ask as _ask, reset as _reset
    return _ask, _reset

# ── Serve frontend ─────────────────────────────────────────────
@app.get("/")
async def serve_ui():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return JSONResponse({"status": "Hospital API running. index.html not found."})

# ── Health ─────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    groq_set = bool(os.environ.get("GROQ_API_KEY"))
    return {
        "status": "healthy",
        "groq_key_set": groq_set,
        "db_dir": os.environ.get("HOSPITAL_DB_DIR"),
        "tmp_writable": os.access("/tmp", os.W_OK),
    }

# ── Patients ───────────────────────────────────────────────────
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

# ── Query ──────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    patient_id: str = ""
    remember: bool = True

@app.post("/api/query")
async def query_patient(request: QueryRequest):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        ask, _ = get_agent()
        answer = ask(request.query, request.patient_id, request.remember)
        return {"status": "success", "answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Reset ──────────────────────────────────────────────────────
@app.post("/api/reset")
async def reset_conversation():
    try:
        _, reset = get_agent()
        reset()
        return {"status": "success", "message": "Conversation reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
