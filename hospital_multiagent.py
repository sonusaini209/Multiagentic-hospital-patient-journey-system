import os
import json
import sqlite3
import operator
from typing import TypedDict, Annotated, Optional
from datetime import datetime, timedelta
import random

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()


# CONFIG
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL_NAME   = "llama-3.3-70b-versatile"

DB_DIR = os.environ.get(
    os.path.join(os.path.dirname(__file__), "hospital_databases")
)
DB_ADMITCORE  = os.path.join(DB_DIR, "db_admitcore.db")
DB_LABTRACK   = os.path.join(DB_DIR, "db_labtrack.db")
DB_PHARMAFLOW = os.path.join(DB_DIR, "db_pharmaflow.db")
DB_BILLDESK   = os.path.join(DB_DIR, "db_billdesk.db")
os.makedirs(DB_DIR, exist_ok=True)True)

# LLM
llm = ChatGroq(model=MODEL_NAME, temperature=0, api_key=GROQ_API_KEY)

# SECTION 1 — SHARED STATE
class HospitalState(TypedDict):
    # Input
    query:      str
    patient_id: Optional[str]

    # Super Agent routing decision
    agents_needed: list[str]
    reasoning:     str
    sub_tasks:     dict

    # Sub-agent results (Annotated = safe parallel writes)
    admitcore_result:  Annotated[list, operator.add]
    labtrack_result:   Annotated[list, operator.add]
    pharmaflow_result: Annotated[list, operator.add]
    billdesk_result:   Annotated[list, operator.add]
    agent_errors:      Annotated[list, operator.add]

    # Output
    final_answer: str
    chat_history: list[dict]


# SECTION 2 — DATABASE SETUP & SYNTHETIC DATA
def _rdate(start=30, end=0):
    d = random.randint(end, start)
    return (datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d")


def setup_admitcore():
    conn = sqlite3.connect(DB_ADMITCORE)
    c = conn.cursor()
    c.executescript("""
        DROP TABLE IF EXISTS Patients;
        DROP TABLE IF EXISTS Doctors;
        DROP TABLE IF EXISTS Admissions;

        CREATE TABLE Patients (
            PatientID TEXT PRIMARY KEY, Name TEXT, Age INTEGER,
            BloodGroup TEXT, ContactNumber TEXT, Email TEXT
        );
        CREATE TABLE Doctors (
            DoctorID TEXT PRIMARY KEY, Name TEXT,
            Specialization TEXT, Department TEXT
        );
        CREATE TABLE Admissions (
            AdmissionID TEXT PRIMARY KEY, PatientID TEXT, DoctorID TEXT,
            AdmitDate TEXT, DischargeDate TEXT, Ward TEXT,
            Status TEXT, Reason TEXT
        );
    """)
    c.executemany("INSERT INTO Doctors VALUES (?,?,?,?)", [
        ("D001","Dr. Arjun Mehta",   "Cardiologist",    "Cardiology"),
        ("D002","Dr. Priya Sharma",  "Neurologist",     "Neurology"),
        ("D003","Dr. Ravi Kumar",    "Orthopedic",      "Orthopedics"),
        ("D004","Dr. Sunita Verma",  "General Physician","General Medicine"),
        ("D005","Dr. Anil Gupta",    "Oncologist",      "Oncology"),
        ("D006","Dr. Meena Iyer",    "Pediatrician",    "Pediatrics"),
    ])
    c.executemany("INSERT INTO Patients VALUES (?,?,?,?,?,?)", [
        ("P001","Rahul Singh",  34,"O+", "9876543210","rahul@email.com"),
        ("P002","Anjali Patel", 28,"A+", "9876543211","anjali@email.com"),
        ("P003","Vikram Rao",   52,"B-", "9876543212","vikram@email.com"),
        ("P004","Sonal Joshi",  45,"AB+","9876543213","sonal@email.com"),
        ("P005","Deepak Nair",  61,"O-", "9876543214","deepak@email.com"),
        ("P006","Kavya Menon",  19,"A-", "9876543215","kavya@email.com"),
        ("P007","Harish Bhat",  38,"B+", "9876543216","harish@email.com"),
        ("P008","Nisha Kapoor", 55,"O+", "9876543217","nisha@email.com"),
    ])
    c.executemany("INSERT INTO Admissions VALUES (?,?,?,?,?,?,?,?)", [
        ("ADM001","P001","D001",_rdate(10,8), None,           "Cardiology Ward","Active",    "Chest Pain"),
        ("ADM002","P002","D004",_rdate(20,18),_rdate(12,10), "General Ward A", "Discharged","Fever & Infection"),
        ("ADM003","P003","D003",_rdate(5,3),  None,           "Orthopedic Ward","Active",    "Fracture - Left Femur"),
        ("ADM004","P004","D002",_rdate(15,13),_rdate(7,5),   "Neurology Ward", "Discharged","Migraine & Seizure"),
        ("ADM005","P005","D001",_rdate(3,1),  None,           "ICU",            "ICU",       "Heart Attack"),
        ("ADM006","P006","D006",_rdate(8,6),  None,           "General Ward B", "Active",    "Appendicitis"),
        ("ADM007","P007","D005",_rdate(25,22),_rdate(15,13), "Oncology Ward",  "Discharged","Chemotherapy Cycle 2"),
        ("ADM008","P008","D004",_rdate(2,1),  None,           "General Ward A", "Active",    "Diabetes & High BP"),
    ])
    conn.commit(); conn.close()


def setup_labtrack():
    conn = sqlite3.connect(DB_LABTRACK)
    c = conn.cursor()
    c.executescript("""
        DROP TABLE IF EXISTS LabTests;
        DROP TABLE IF EXISTS TestOrders;
        DROP TABLE IF EXISTS TestResults;

        CREATE TABLE LabTests (
            TestID TEXT PRIMARY KEY, TestName TEXT,
            Description TEXT, NormalRange TEXT, UnitCost REAL
        );
        CREATE TABLE TestOrders (
            OrderID TEXT PRIMARY KEY, PatientID TEXT, AdmissionID TEXT,
            TestID TEXT, OrderedDate TEXT, Status TEXT
        );
        CREATE TABLE TestResults (
            ResultID TEXT PRIMARY KEY, OrderID TEXT,
            ResultValue TEXT, ReportDate TEXT, Flag TEXT, Notes TEXT
        );
    """)
    c.executemany("INSERT INTO LabTests VALUES (?,?,?,?,?)", [
        ("LT001","Complete Blood Count (CBC)", "Full blood panel",       "WBC: 4-11 K/uL",    350.0),
        ("LT002","Blood Glucose - Fasting",    "Fasting sugar level",    "70-100 mg/dL",      120.0),
        ("LT003","Lipid Profile",              "Cholesterol panel",      "Total < 200 mg/dL", 450.0),
        ("LT004","ECG",                        "Heart electrical activity","Normal Sinus",     200.0),
        ("LT005","MRI Brain",                  "Brain imaging",          "No lesions",       3500.0),
        ("LT006","X-Ray - Left Femur",         "Bone imaging",           "No fracture",       600.0),
        ("LT007","Liver Function Test (LFT)",  "Liver enzyme levels",    "ALT: 7-56 U/L",     380.0),
        ("LT008","Urine Routine",              "Urine analysis",         "No abnormality",    150.0),
        ("LT009","Thyroid Profile",            "Thyroid hormone levels", "TSH: 0.4-4 mIU/L",  420.0),
        ("LT010","Troponin I",                 "Heart attack marker",    "< 0.04 ng/mL",      700.0),
    ])
    c.executemany("INSERT INTO TestOrders VALUES (?,?,?,?,?,?)", [
        ("TO001","P001","ADM001","LT004",_rdate(9,9), "Completed"),
        ("TO002","P001","ADM001","LT010",_rdate(9,9), "Completed"),
        ("TO003","P001","ADM001","LT001",_rdate(8,8), "Completed"),
        ("TO004","P002","ADM002","LT001",_rdate(19,19),"Completed"),
        ("TO005","P002","ADM002","LT008",_rdate(19,19),"Completed"),
        ("TO006","P003","ADM003","LT006",_rdate(4,4), "Completed"),
        ("TO007","P003","ADM003","LT001",_rdate(4,4), "Completed"),
        ("TO008","P004","ADM004","LT005",_rdate(14,14),"Completed"),
        ("TO009","P005","ADM005","LT010",_rdate(2,2), "Completed"),
        ("TO010","P005","ADM005","LT004",_rdate(2,2), "Completed"),
        ("TO011","P006","ADM006","LT001",_rdate(7,7), "Completed"),
        ("TO012","P008","ADM008","LT002",_rdate(1,1), "Completed"),
        ("TO013","P008","ADM008","LT009",_rdate(1,1), "Pending"),
    ])
    c.executemany("INSERT INTO TestResults VALUES (?,?,?,?,?,?)", [
        ("TR001","TO001","Abnormal ST elevation in V2-V4",      _rdate(9,9), "Abnormal","Suggests anterior STEMI"),
        ("TR002","TO002","Troponin I: 2.8 ng/mL",              _rdate(9,9), "Critical","Significantly elevated - cardiac event"),
        ("TR003","TO003","WBC: 12.5 K/uL, RBC: 4.1",           _rdate(8,8), "Abnormal","Elevated WBC - possible infection"),
        ("TR004","TO004","WBC: 9.2, RBC: 4.5, Hb: 13.2",       _rdate(19,19),"Normal", "All parameters within range"),
        ("TR005","TO005","Pus cells: 8-10/HPF",                _rdate(19,19),"Abnormal","UTI suspected"),
        ("TR006","TO006","Fracture at mid-shaft left femur",    _rdate(4,4), "Abnormal","Displaced fracture confirmed"),
        ("TR007","TO007","WBC: 10.1, Hb: 11.8",                _rdate(4,4), "Normal",  "Mild anemia noted"),
        ("TR008","TO008","Small lesion noted in temporal lobe", _rdate(14,14),"Abnormal","Follow-up MRI in 4 weeks"),
        ("TR009","TO009","Troponin I: 5.1 ng/mL",              _rdate(2,2), "Critical","Massive cardiac event - urgent"),
        ("TR010","TO010","Ventricular fibrillation pattern",    _rdate(2,2), "Critical","Life threatening - ICU monitoring"),
        ("TR011","TO011","WBC: 15.2 K/uL",                     _rdate(7,7), "Abnormal","Consistent with appendicitis"),
        ("TR012","TO012","Blood Glucose: 210 mg/dL",           _rdate(1,1), "Abnormal","Significantly above normal"),
    ])
    conn.commit(); conn.close()


def setup_pharmaflow():
    conn = sqlite3.connect(DB_PHARMAFLOW)
    c = conn.cursor()
    c.executescript("""
        DROP TABLE IF EXISTS Medicines;
        DROP TABLE IF EXISTS Prescriptions;
        DROP TABLE IF EXISTS PrescriptionItems;

        CREATE TABLE Medicines (
            MedicineID TEXT PRIMARY KEY, Name TEXT,
            Category TEXT, StockQty INTEGER, UnitPrice REAL
        );
        CREATE TABLE Prescriptions (
            PrescriptionID TEXT PRIMARY KEY, PatientID TEXT,
            DoctorID TEXT, AdmissionID TEXT, PrescribedDate TEXT, Notes TEXT
        );
        CREATE TABLE PrescriptionItems (
            ItemID TEXT PRIMARY KEY, PrescriptionID TEXT,
            MedicineID TEXT, Dosage TEXT, Duration TEXT, Quantity INTEGER
        );
    """)
    c.executemany("INSERT INTO Medicines VALUES (?,?,?,?,?)", [
        ("M001","Aspirin 75mg",         "Antiplatelet",    500, 2.50),
        ("M002","Metoprolol 25mg",      "Beta Blocker",    300, 8.00),
        ("M003","Atorvastatin 40mg",    "Statin",          400,12.00),
        ("M004","Clopidogrel 75mg",     "Antiplatelet",    250,15.00),
        ("M005","Amoxicillin 500mg",    "Antibiotic",      600, 5.00),
        ("M006","Paracetamol 500mg",    "Analgesic",       800, 1.50),
        ("M007","Morphine 10mg",        "Opioid Analgesic",100,45.00),
        ("M008","Insulin Glargine",     "Antidiabetic",    200,85.00),
        ("M009","Metformin 500mg",      "Antidiabetic",    500, 4.00),
        ("M010","Levetiracetam 500mg",  "Anticonvulsant",  150,22.00),
        ("M011","Enoxaparin 40mg",      "Anticoagulant",   180,55.00),
        ("M012","Ondansetron 4mg",      "Antiemetic",      300, 6.00),
        ("M013","Pantoprazole 40mg",    "PPI",             600, 7.00),
        ("M014","Ceftriaxone 1g IV",    "Antibiotic IV",   120,95.00),
    ])
    c.executemany("INSERT INTO Prescriptions VALUES (?,?,?,?,?,?)", [
        ("PR001","P001","D001","ADM001",_rdate(9,9), "Post cardiac event - monitor BP daily"),
        ("PR002","P002","D004","ADM002",_rdate(18,18),"Complete antibiotic course"),
        ("PR003","P003","D003","ADM003",_rdate(4,4), "Post fracture - pain management"),
        ("PR004","P004","D002","ADM004",_rdate(13,13),"Seizure management protocol"),
        ("PR005","P005","D001","ADM005",_rdate(2,2), "Critical cardiac care - ICU protocol"),
        ("PR006","P006","D006","ADM006",_rdate(7,7), "Pre-surgery antibiotics"),
        ("PR007","P008","D004","ADM008",_rdate(1,1), "Diabetes + hypertension management"),
        ("PR008","P001","D001","ADM001",_rdate(7,7), "Updated after abnormal Troponin"),
    ])
    c.executemany("INSERT INTO PrescriptionItems VALUES (?,?,?,?,?,?)", [
        ("PI001","PR001","M001","75mg once daily",      "30 days",30),
        ("PI002","PR001","M002","25mg twice daily",     "30 days",60),
        ("PI003","PR001","M003","40mg at night",        "30 days",30),
        ("PI004","PR001","M004","75mg once daily",      "30 days",30),
        ("PI005","PR002","M005","500mg thrice daily",   "7 days", 21),
        ("PI006","PR002","M006","500mg as needed",      "5 days", 15),
        ("PI007","PR003","M007","10mg IV every 6 hours","3 days", 12),
        ("PI008","PR003","M006","500mg thrice daily",   "5 days", 15),
        ("PI009","PR004","M010","500mg twice daily",    "90 days",180),
        ("PI010","PR005","M011","40mg SC once daily",   "7 days",  7),
        ("PI011","PR005","M002","25mg IV once daily",   "5 days",  5),
        ("PI012","PR006","M014","1g IV pre-surgery",    "1 day",   1),
        ("PI013","PR007","M008","10 units at bedtime",  "30 days", 30),
        ("PI014","PR007","M009","500mg twice daily",    "30 days", 60),
        ("PI015","PR008","M011","40mg SC once daily",   "5 days",  5),
        ("PI016","PR008","M013","40mg once daily",      "7 days",  7),
    ])
    conn.commit(); conn.close()


def setup_billdesk():
    conn = sqlite3.connect(DB_BILLDESK)
    c = conn.cursor()
    c.executescript("""
        DROP TABLE IF EXISTS Bills;
        DROP TABLE IF EXISTS BillItems;
        DROP TABLE IF EXISTS Insurance;

        CREATE TABLE Bills (
            BillID TEXT PRIMARY KEY, PatientID TEXT, AdmissionID TEXT,
            TotalAmount REAL, PaidAmount REAL DEFAULT 0,
            Status TEXT, GeneratedDate TEXT
        );
        CREATE TABLE BillItems (
            ItemID TEXT PRIMARY KEY, BillID TEXT,
            Description TEXT, Amount REAL, Category TEXT
        );
        CREATE TABLE Insurance (
            InsuranceID TEXT PRIMARY KEY, PatientID TEXT, Provider TEXT,
            PolicyNumber TEXT, CoverageAmount REAL, ExpiryDate TEXT, Status TEXT
        );
    """)
    c.executemany("INSERT INTO Insurance VALUES (?,?,?,?,?,?,?)", [
        ("INS001","P001","Star Health",   "SH-2024-001",500000.0,"2026-03-31","Active"),
        ("INS002","P002","HDFC ERGO",     "HE-2024-002",300000.0,"2025-12-31","Active"),
        ("INS003","P003","Bajaj Allianz", "BA-2024-003",400000.0,"2026-06-30","Active"),
        ("INS004","P004","ICICI Lombard", "IL-2023-004",250000.0,"2024-11-30","Expired"),
        ("INS005","P005","Star Health",   "SH-2024-005",1000000.0,"2026-12-31","Active"),
        ("INS006","P006","New India",     "NI-2024-006",200000.0,"2026-01-31","Active"),
        ("INS007","P007","United India",  "UI-2024-007",600000.0,"2025-09-30","Active"),
        ("INS008","P008","Oriental",      "OR-2024-008",350000.0,"2026-04-30","Active"),
    ])
    c.executemany("INSERT INTO Bills VALUES (?,?,?,?,?,?,?)", [
        ("B001","P001","ADM001", 45200.0, 10000.0,"Partial",_rdate(8,8)),
        ("B002","P002","ADM002",  8500.0,  8500.0,"Paid",   _rdate(11,11)),
        ("B003","P003","ADM003", 32000.0, 15000.0,"Partial",_rdate(3,3)),
        ("B004","P004","ADM004", 21000.0, 21000.0,"Paid",   _rdate(6,6)),
        ("B005","P005","ADM005",125000.0,     0.0,"Pending",_rdate(1,1)),
        ("B006","P006","ADM006", 18500.0,  5000.0,"Partial",_rdate(6,6)),
        ("B007","P007","ADM007", 85000.0, 85000.0,"Paid",   _rdate(14,14)),
        ("B008","P008","ADM008", 12400.0,     0.0,"Pending",_rdate(1,1)),
    ])
    c.executemany("INSERT INTO BillItems VALUES (?,?,?,?,?)", [
        ("BI001","B001","ECG",                            200.0,  "Lab"),
        ("BI002","B001","Troponin I Test",                700.0,  "Lab"),
        ("BI003","B001","Complete Blood Count",           350.0,  "Lab"),
        ("BI004","B001","Cardiology Ward - 8 days",     12000.0,  "Room"),
        ("BI005","B001","Dr. Arjun Mehta Consultation",  2000.0,  "Doctor"),
        ("BI006","B001","Aspirin + Metoprolol + Statin",  680.0,  "Pharmacy"),
        ("BI007","B001","Clopidogrel 30 days",            450.0,  "Pharmacy"),
        ("BI008","B001","Enoxaparin injections",          275.0,  "Pharmacy"),
        ("BI009","B001","Cardiac Monitoring Procedure", 28545.0,  "Procedure"),
        ("BI010","B003","X-Ray Left Femur",               600.0,  "Lab"),
        ("BI011","B003","CBC",                            350.0,  "Lab"),
        ("BI012","B003","Orthopedic Ward - 3 days",      4500.0,  "Room"),
        ("BI013","B003","Dr. Ravi Kumar Consultation",   1500.0,  "Doctor"),
        ("BI014","B003","Morphine + Paracetamol",         825.0,  "Pharmacy"),
        ("BI015","B003","Fracture Fixation Surgery",    24225.0,  "Procedure"),
        ("BI016","B005","Troponin I - Urgent",            700.0,  "Lab"),
        ("BI017","B005","ECG",                            200.0,  "Lab"),
        ("BI018","B005","ICU - 2 days",                 20000.0,  "Room"),
        ("BI019","B005","Dr. Arjun Mehta - ICU Care",    5000.0,  "Doctor"),
        ("BI020","B005","Enoxaparin + Metoprolol",        380.0,  "Pharmacy"),
        ("BI021","B005","Emergency Cardiac Procedure",  98720.0,  "Procedure"),
        ("BI022","B008","Blood Glucose Test",             120.0,  "Lab"),
        ("BI023","B008","Thyroid Profile",                420.0,  "Lab"),
        ("BI024","B008","General Ward - 2 days",         3000.0,  "Room"),
        ("BI025","B008","Dr. Sunita Verma Consultation", 1500.0,  "Doctor"),
        ("BI026","B008","Insulin + Metformin",           7360.0,  "Pharmacy"),
    ])
    conn.commit(); conn.close()


def setup_all_databases():
    """Create all 4 databases with synthetic data."""
    print("🏥 Setting up Hospital Databases...")
    setup_admitcore();  print("  ✅ AdmitCore  — Patients, Doctors, Admissions")
    setup_labtrack();   print("  ✅ LabTrack   — Tests, Orders, Results")
    setup_pharmaflow(); print("  ✅ PharmaFlow — Medicines, Prescriptions")
    setup_billdesk();   print("  ✅ BillDesk   — Bills, Insurance")
    print("  All 4 databases ready.\n")

# SECTION 3 — SUPER AGENT (Router + Synthesizer)
AGENT_DESCRIPTIONS = """
You have 4 specialized sub-agents, each with their own isolated database:

1. ADMITCORE  — Patients, Doctors, Admissions
   Knows: patient details, ward, doctor assigned, admission/discharge dates, status

2. LABTRACK   — LabTests, TestOrders, TestResults
   Knows: which tests ordered, test results, normal/abnormal/critical flags

3. PHARMAFLOW — Medicines, Prescriptions, PrescriptionItems
   Knows: medicines prescribed, dosage, duration, prescription history

4. BILLDESK   — Bills, BillItems, Insurance
   Knows: bill total, paid amount, balance, bill breakdown, insurance coverage
"""


def super_agent_router(state: HospitalState) -> dict:
    """Node 1 — Reads query, decides which agents to call and what each should do."""

    query       = state["query"]
    patient_id  = state.get("patient_id", "")
    history     = state.get("chat_history", [])
    history_txt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-4:])

    system = f"""You are the Super Agent orchestrator for a Hospital Management System.
{AGENT_DESCRIPTIONS}

Analyze the user query and return ONLY a valid JSON routing decision.
Format:
{{
  "patient_id": "P001",
  "agents_needed": ["admitcore", "labtrack"],
  "reasoning": "why these agents",
  "sub_tasks": {{
    "admitcore": "specific task for this agent",
    "labtrack":  "specific task for this agent"
  }}
}}

Rules:
- Only include agents actually needed
- Extract PatientID if mentioned (P001 format)
- If not mentioned, use from history context or leave ""
- sub_tasks must be specific, not just repeat the query
"""
    msg = f"History:\n{history_txt}\n\nQuery: {query}\nKnown patient_id: {patient_id}"

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=msg)])

    raw = response.content.strip()
    if "```json" in raw: raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:   raw = raw.split("```")[1].split("```")[0].strip()

    try:
        d = json.loads(raw)
    except Exception:
        # Fallback
        d = {
            "patient_id":    patient_id,
            "agents_needed": ["admitcore","labtrack","pharmaflow","billdesk"],
            "reasoning":     "Fallback: parse error, calling all agents",
            "sub_tasks":     {a: query for a in ["admitcore","labtrack","pharmaflow","billdesk"]},
        }

    print(f"\n{'═'*60}")
    print(f"  🧠  SUPER AGENT — ROUTING DECISION")
    print(f"{'═'*60}")
    print(f"  Query      : {query}")
    print(f"  PatientID  : {d.get('patient_id','—')}")
    print(f"  Agents     : {d.get('agents_needed', [])}")
    print(f"  Reasoning  : {d.get('reasoning','')}")
    print(f"  Sub-Tasks  :")
    for agent, task in d.get("sub_tasks", {}).items():
        print(f"    [{agent:12s}] → {task}")
    print(f"{'═'*60}\n")

    return {
        "patient_id":    d.get("patient_id", patient_id),
        "agents_needed": d.get("agents_needed", []),
        "reasoning":     d.get("reasoning", ""),
        "sub_tasks":     d.get("sub_tasks", {}),
    }


def super_agent_synthesizer(state: HospitalState) -> dict:
    """Node 6 — Combines all sub-agent results into a final human-readable answer."""

    query   = state["query"]
    pid     = state.get("patient_id", "")
    results = ""
    if state.get("admitcore_result"):
        results += f"\n📋 ADMISSIONS:\n{json.dumps(state['admitcore_result'], indent=2)}"
    if state.get("labtrack_result"):
        results += f"\n🧪 LAB RESULTS:\n{json.dumps(state['labtrack_result'], indent=2)}"
    if state.get("pharmaflow_result"):
        results += f"\n💊 PHARMACY:\n{json.dumps(state['pharmaflow_result'], indent=2)}"
    if state.get("billdesk_result"):
        results += f"\n💰 BILLING:\n{json.dumps(state['billdesk_result'], indent=2)}"
    if state.get("agent_errors"):
        results += f"\n⚠️ ERRORS:\n{json.dumps(state['agent_errors'], indent=2)}"

    system = """You are a helpful hospital assistant synthesizing data from multiple hospital systems.
- Answer the patient's question clearly and completely
- Be empathetic and professional — this is healthcare
- Highlight any CRITICAL or ABNORMAL findings prominently with ⚠️
- Format amounts in ₹, dates in readable format
- Do NOT show raw JSON, database IDs, or technical field names
- Speak in plain, friendly language
"""
    msg = f"Patient query: {query}\nPatient ID: {pid}\n\nData from hospital systems:\n{results}"

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=msg)])
    answer   = response.content.strip()

    print(f"\n{'═'*60}")
    print(f"  ✅  FINAL ANSWER")
    print(f"{'═'*60}")
    print(answer)
    print(f"{'═'*60}\n")

    history = state.get("chat_history", [])
    history.append({"role": "user",      "content": query})
    history.append({"role": "assistant", "content": answer})

    return {"final_answer": answer, "chat_history": history}

# SECTION 4 — SUB-AGENTS (Text-to-SQL)
def _run_sql(db_path: str, sql: str) -> list[dict]:
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur  = conn.cursor()
        cur.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        return [{"error": str(e), "sql_attempted": sql}]


def _text_to_sql(agent_name: str, db_path: str, schema: str, task: str, pid: str) -> list[dict]:
    system = f"""You are {agent_name}, a text-to-SQL expert.
You ONLY have access to your own database. You cannot query other databases.

YOUR SCHEMA:
{schema}

STRICT RULES:
- Return ONLY a single valid SQLite SQL query — no explanation, no markdown
- ALWAYS JOIN all relevant tables — never query just one table alone
- Return ALL useful columns (names, dates, values, flags, notes — everything)
- Use LEFT JOIN for result/item tables so rows appear even if sub-data is missing
- Filter by PatientID = '{pid}' where applicable
"""
    msg  = f"Task: {task}\nPatientID: {pid}\n\nWrite the SQL query:"
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=msg)])
    sql  = resp.content.strip()
    if "```sql" in sql: sql = sql.split("```sql")[1].split("```")[0].strip()
    elif "```" in sql:  sql = sql.split("```")[1].split("```")[0].strip()
    return _run_sql(db_path, sql)


# ── Schemas — each agent only knows its own ──────────────────

_ADMITCORE_SCHEMA = """
Patients  (PatientID, Name, Age, BloodGroup, ContactNumber, Email)
Doctors   (DoctorID, Name, Specialization, Department)
Admissions(AdmissionID, PatientID, DoctorID, AdmitDate, DischargeDate, Ward, Status[Active/Discharged/ICU], Reason)
  Admissions.PatientID → Patients.PatientID
  Admissions.DoctorID  → Doctors.DoctorID

Good query example:
  SELECT p.Name, p.Age, p.BloodGroup, p.ContactNumber,
         a.AdmissionID, a.AdmitDate, a.DischargeDate, a.Ward, a.Status, a.Reason,
         d.Name as DoctorName, d.Specialization, d.Department
  FROM Admissions a
  JOIN Patients p ON a.PatientID = p.PatientID
  JOIN Doctors  d ON a.DoctorID  = d.DoctorID
  WHERE a.PatientID = 'P001'
"""

_LABTRACK_SCHEMA = """
LabTests   (TestID, TestName, Description, NormalRange, UnitCost)
TestOrders (OrderID, PatientID, AdmissionID, TestID, OrderedDate, Status[Pending/Completed/Cancelled])
TestResults(ResultID, OrderID, ResultValue, ReportDate, Flag[Normal/Abnormal/Critical], Notes)
  TestOrders.TestID    → LabTests.TestID
  TestResults.OrderID  → TestOrders.OrderID

Good query example:
  SELECT lt.TestName, lt.NormalRange, lt.UnitCost,
         tor.OrderedDate, tor.Status,
         tr.ResultValue, tr.ReportDate, tr.Flag, tr.Notes
  FROM TestOrders tor
  JOIN LabTests    lt ON tor.TestID  = lt.TestID
  LEFT JOIN TestResults tr ON tr.OrderID = tor.OrderID
  WHERE tor.PatientID = 'P001'
"""

_PHARMAFLOW_SCHEMA = """
Medicines        (MedicineID, Name, Category, StockQty, UnitPrice)
Prescriptions    (PrescriptionID, PatientID, DoctorID, AdmissionID, PrescribedDate, Notes)
PrescriptionItems(ItemID, PrescriptionID, MedicineID, Dosage, Duration, Quantity)
  PrescriptionItems.PrescriptionID → Prescriptions.PrescriptionID
  PrescriptionItems.MedicineID     → Medicines.MedicineID

Good query example:
  SELECT m.Name as MedicineName, m.Category, m.UnitPrice,
         pi.Dosage, pi.Duration, pi.Quantity,
         pr.PrescribedDate, pr.Notes
  FROM Prescriptions pr
  JOIN PrescriptionItems pi ON pi.PrescriptionID = pr.PrescriptionID
  JOIN Medicines         m  ON m.MedicineID      = pi.MedicineID
  WHERE pr.PatientID = 'P001'
"""

_BILLDESK_SCHEMA = """
Bills    (BillID, PatientID, AdmissionID, TotalAmount, PaidAmount, Status[Pending/Partial/Paid], GeneratedDate)
BillItems(ItemID, BillID, Description, Amount, Category[Lab/Pharmacy/Room/Doctor/Procedure])
Insurance(InsuranceID, PatientID, Provider, PolicyNumber, CoverageAmount, ExpiryDate, Status[Active/Expired/Claimed])
  BillItems.BillID     → Bills.BillID
  Insurance.PatientID  → filter only

Good query example:
  SELECT b.BillID, b.TotalAmount, b.PaidAmount, b.Status, b.GeneratedDate,
         bi.Description, bi.Amount, bi.Category,
         i.Provider, i.PolicyNumber, i.CoverageAmount, i.ExpiryDate, i.Status as InsuranceStatus
  FROM Bills b
  JOIN BillItems bi       ON bi.BillID    = b.BillID
  LEFT JOIN Insurance i   ON i.PatientID  = b.PatientID
  WHERE b.PatientID = 'P001'
"""


def admitcore_agent(state: HospitalState) -> dict:
    if "admitcore" not in state.get("agents_needed", []):
        return {}
    rows = _text_to_sql("AdmitCore Agent", DB_ADMITCORE, _ADMITCORE_SCHEMA,
                        state.get("sub_tasks", {}).get("admitcore", state["query"]),
                        state.get("patient_id", ""))
    return {"admitcore_result": rows}


def labtrack_agent(state: HospitalState) -> dict:
    if "labtrack" not in state.get("agents_needed", []):
        return {}
    rows = _text_to_sql("LabTrack Agent", DB_LABTRACK, _LABTRACK_SCHEMA,
                        state.get("sub_tasks", {}).get("labtrack", state["query"]),
                        state.get("patient_id", ""))
    return {"labtrack_result": rows}


def pharmaflow_agent(state: HospitalState) -> dict:
    if "pharmaflow" not in state.get("agents_needed", []):
        return {}
    rows = _text_to_sql("PharmaFlow Agent", DB_PHARMAFLOW, _PHARMAFLOW_SCHEMA,
                        state.get("sub_tasks", {}).get("pharmaflow", state["query"]),
                        state.get("patient_id", ""))
    return {"pharmaflow_result": rows}


def billdesk_agent(state: HospitalState) -> dict:
    if "billdesk" not in state.get("agents_needed", []):
        return {}
    rows = _text_to_sql("BillDesk Agent", DB_BILLDESK, _BILLDESK_SCHEMA,
                        state.get("sub_tasks", {}).get("billdesk", state["query"]),
                        state.get("patient_id", ""))
    return {"billdesk_result": rows}


# SECTION 5 — GRAPH ASSEMBLY
def route_to_agents(state: HospitalState) -> list[str]:
    mapping = {
        "admitcore":  "admitcore_agent",
        "labtrack":   "labtrack_agent",
        "pharmaflow": "pharmaflow_agent",
        "billdesk":   "billdesk_agent",
    }
    nodes = [mapping[a] for a in state.get("agents_needed", []) if a in mapping]
    return nodes or list(mapping.values())


def build_graph():
    g = StateGraph(HospitalState)

    g.add_node("super_agent_router",      super_agent_router)
    g.add_node("admitcore_agent",         admitcore_agent)
    g.add_node("labtrack_agent",          labtrack_agent)
    g.add_node("pharmaflow_agent",        pharmaflow_agent)
    g.add_node("billdesk_agent",          billdesk_agent)
    g.add_node("super_agent_synthesizer", super_agent_synthesizer)

    g.add_edge(START, "super_agent_router")

    g.add_conditional_edges(
        "super_agent_router",
        route_to_agents,
        {
            "admitcore_agent":  "admitcore_agent",
            "labtrack_agent":   "labtrack_agent",
            "pharmaflow_agent": "pharmaflow_agent",
            "billdesk_agent":   "billdesk_agent",
        }
    )

    for node in ["admitcore_agent", "labtrack_agent", "pharmaflow_agent", "billdesk_agent"]:
        g.add_edge(node, "super_agent_synthesizer")

    g.add_edge("super_agent_synthesizer", END)

    return g.compile()

# SECTION 6 — RUNNER

# Global conversation history — persists across calls in same session
# This lets the system remember context like:
# Q1: "tell me about patient P001"
# Q2: "what medicines is he on?" ← knows P001 from Q1
_conversation_history = []
_current_patient_id   = ""

def ask(query: str, patient_id: str = "", remember: bool = True) -> str:
    """
    Run a query through the full hospital multi-agent system.

    Args:
        query      : Your natural language question
        patient_id : Optional — auto-detected from query if not given
        remember   : If True, keeps conversation context across calls
                     so follow-up questions work naturally.
                     Set to False to start completely fresh.
    """
    global _conversation_history, _current_patient_id

    # Auto-detect patient ID from query text (e.g. "patient P001")
    import re
    match = re.search(r'\bP\d{3}\b', query)
    if match:
        _current_patient_id = match.group()
    elif patient_id:
        _current_patient_id = patient_id

    # Use history only if remember=True
    # History helps with follow-up questions like:
    # "what about his medicines?" after asking about a patient
    history_to_pass = _conversation_history[-6:] if remember else []

    result = graph.invoke({
        "query":            query,
        "patient_id":       _current_patient_id,
        "agents_needed":    [],
        "reasoning":        "",
        "sub_tasks":        {},
        "admitcore_result":  [],
        "labtrack_result":   [],
        "pharmaflow_result": [],
        "billdesk_result":   [],
        "agent_errors":      [],
        "final_answer":     "",
        "chat_history":     history_to_pass,
    })

    answer = result.get("final_answer", "")

    # Save to history for future follow-up questions
    if remember:
        _conversation_history.append({"role": "user",      "content": query})
        _conversation_history.append({"role": "assistant", "content": answer})

    return answer


def reset():
    """Clear conversation history and patient context. Start fresh."""
    global _conversation_history, _current_patient_id
    _conversation_history = []
    _current_patient_id   = ""


# ══════════════════════════════════════════════════════════════
# SECTION 7 — GRAPH BUILD + DEMO QUERIES
# ══════════════════════════════════════════════════════════════

# Build graph once at import time
graph = build_graph()

# ── Demo queries (run each cell separately in Jupyter) ───────

# DEMO 1 — 3 databases: Admissions + Labs + Pharmacy
# ask("I am patient P001. What is my current ward, what lab tests were done, and what medicines are prescribed?")

# DEMO 2 — All 4 databases
# ask("I am patient P005. I had a heart attack. What are my lab results, medicines, total bill, and does my insurance cover it?")

# DEMO 3 — Follow-up question (history helps here)
# ask("I am patient P004. What is my diagnosis and current medicines?")
# ask("What is my pending bill?")   # ← no need to repeat P004, history remembers

# DEMO 4 — Fresh query, no history needed

#ask("Show complete details for patient P003", remember=False)

