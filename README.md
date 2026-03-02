#  Multiagentic Hospital Patient Journey System

> AI-Powered Multi-Agent Hospital Intelligence Platform

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange.svg)](https://console.groq.com)
[![Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black.svg)](https://your-app.vercel.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌐 Live Demo

**https://multiagentic-hospital-patient-journ.vercel.app/**

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq — `llama-3.3-70b-versatile` |
| **Agent Orchestration** | LangGraph |
| **LLM Framework** | LangChain + langchain-groq |
| **Backend** | FastAPI |
| **Database** | SQLite (4 isolated DBs) |
| **Frontend** | HTML + CSS + Vanilla JS |
| **Deployment** | Vercel |
| **Environment** | python-dotenv |


##  Features

-  **Super Agent Orchestrator** — reads the query, routes to the right agents, synthesizes the final answer
-  **Natural Language Queries** — no forms, just ask in plain English
-  **4 Isolated Databases** — AdmitCore, LabTrack, PharmaFlow, BillDesk
-  **Text-to-SQL** — each agent generates its own SQL query on the fly
-  **Critical Flag Detection** — highlights abnormal or critical lab findings
-  **Conversation Memory** — follow-up questions work without repeating patient ID
-  **Web UI** — clean chat interface with patient sidebar and quick queries
-  **Vercel Deployment** — single FastAPI app serving both UI and API

---

##  Database Structure

| Database | Tables | Contains |
|---|---|---|
| **AdmitCore** | Patients, Doctors, Admissions | Patient info, ward, doctor, admission/discharge dates |
| **LabTrack** | LabTests, TestOrders, TestResults | Tests ordered, results, normal/abnormal/critical flags |
| **PharmaFlow** | Medicines, Prescriptions, PrescriptionItems | Medicines prescribed, dosage, duration |
| **BillDesk** | Bills, BillItems, Insurance | Total bill, payments, insurance coverage |

All 4 databases are auto-created with synthetic patient data on first run.

---

##  Project Structure

```
hospital-patient-journey/
├── hospital_multiagent.py   # All agents + LangGraph workflow + DB setup
├── app.py                   # FastAPI backend 
├── index.html               # Frontend chat UI
├── requirements.txt         # Python dependencies
├── vercel.json              # Vercel deployment config
├── .env                     # API keys 
├── .gitignore               # Excludes .env
└── README.md
```

---

##  API Key Required

**Groq API** — [console.groq.com](https://console.groq.com)
- Sign up for a free account
- Generate an API key
- Model used: `llama-3.3-70b-versatile`

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_groq_key_here
```

> ⚠️ `.gitignore` already excludes `.env` — never push your key to GitHub.

---

##  Run Locally

**Prerequisites:** Python 3.9+, pip

1. **Clone the repository**
   ```bash
   git clone https://github.com/sonusaini209/hospital-patient-journey
   cd hospital-patient-journey
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv

   # macOS / Linux
   source venv/bin/activate

   # Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your API key**
   ```env
   GROQ_API_KEY=gsk_your_groq_key_here
   ```

5. **Start the server**
   ```bash
   uvicorn app:app --reload
   ```

   Open [http://localhost:8000](http://localhost:8000) — databases are auto-created on first run.
---

## 💬 Example Queries

```
"I am patient P001. What is my ward, doctor, and admission reason?"
"What are my latest lab test results?"
"What medicines have been prescribed to me and what are the dosages?"
"What is my total bill and how much is still pending?"
"Does my insurance cover my current bill?"
"Give me a complete summary of my patient journey."
```

Conversation memory means follow-up questions work naturally:
```
Q: "I am patient P005. What happened to me?"
Q: "What medicines am I on?"        ← no need to repeat P005
Q: "Is my insurance still active?"  ← still remembers P005
```

---

##  License

MIT — see [LICENSE](./LICENSE) for details.

