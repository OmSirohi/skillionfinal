import os
import io
import json
import time
import hashlib
import secrets
import zipfile
import re
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, HTTPException, Depends, Request, Header, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Third‑party libraries available in the environment
import fitz  # PyMuPDF for PDF parsing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------------------------------------------------------
# Global in‑memory stores
#
# In a production application you would persist all of this data to a proper
# database.  For the sake of this exercise everything lives in memory.  When
# the process restarts all uploaded resumés and jobs disappear.

users: Dict[int, Dict[str, Any]] = {}
resumes: Dict[int, Dict[str, Any]] = {}
jobs: Dict[int, Dict[str, Any]] = {}

resumes_by_user: Dict[int, List[int]] = {}
tokens: Dict[str, Dict[str, Any]] = {}
rate_limits: Dict[int, Dict[str, Any]] = {}
idempotency_store: Dict[int, Dict[str, Any]] = {}

paragraphs: List[Dict[str, Any]] = []  # Each entry: {resume_id, candidate_id, text}
paragraph_vectorizer: Optional[TfidfVectorizer] = None
paragraph_vectors = None  # type: ignore

resume_vectorizer: Optional[TfidfVectorizer] = None
resume_vectors = None  # type: ignore
resume_id_order: List[int] = []

# Counters for generating IDs
next_user_id = 1
next_resume_id = 1
next_job_id = 1

# ----------------------------------------------------------------------------
# Utility functions


def hash_password(password: str) -> str:
    """Return a SHA256 hash of the given password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """Check if the provided password matches the stored hash."""
    return hash_password(password) == hashed


def create_token(user_id: int) -> str:
    """Generate a new bearer token for the given user and store it."""
    token = secrets.token_hex(16)
    # Tokens expire after 24 hours
    tokens[token] = {"user_id": user_id, "expiry": time.time() + 86400}
    return token


async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Dependency to extract and verify the current user from the Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail={"error": {"code": "UNAUTHORIZED", "message": "Missing or invalid token"}})
    token = authorization[len("Bearer "):]
    rec = tokens.get(token)
    if not rec or rec["expiry"] < time.time():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail={"error": {"code": "UNAUTHORIZED", "message": "Token expired or invalid"}})
    user = users.get(rec["user_id"])
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail={"error": {"code": "UNAUTHORIZED", "message": "User not found"}})
    return user


async def enforce_rate_limit(user: Dict[str, Any] = Depends(get_current_user)) -> None:
    """Dependency to enforce 60 requests per minute per user."""
    now = time.time()
    rec = rate_limits.get(user["id"])
    if not rec or now - rec["window_start"] >= 60:
        # Reset the window
        rate_limits[user["id"]] = {"window_start": now, "count": 1}
        return
    # Increment count
    rec["count"] += 1
    if rec["count"] > 60:
        raise HTTPException(status_code=429, detail={"error": {"code": "RATE_LIMIT"}})


def store_idempotent_response(user_id: int, key: str, response: Any) -> None:
    """Store the response for the given user and idempotency key."""
    if user_id not in idempotency_store:
        idempotency_store[user_id] = {}
    idempotency_store[user_id][key] = response


def get_idempotent_response(user_id: int, key: str) -> Optional[Any]:
    """Retrieve a previously stored response for the given user and key."""
    return idempotency_store.get(user_id, {}).get(key)


def redact_pii(text: str) -> str:
    """Redact email addresses and phone numbers from the text."""
    # Redact email addresses
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED]", text)
    # Redact sequences of 10–15 digits (common phone number formats)
    text = re.sub(r"\b\d{10,15}\b", "[REDACTED]", text)
    return text


def parse_pdf(data: bytes) -> str:
    """Extract text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        return ""
    full_text = []
    for page in doc:
        try:
            full_text.append(page.get_text())
        except Exception:
            continue
    doc.close()
    return "\n".join(full_text)


def parse_txt(data: bytes) -> str:
    """Decode bytes as UTF‑8 text, ignoring errors."""
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_paragraphs(text: str) -> List[str]:
    """Split a document into paragraphs.  Filters out very short fragments."""
    # Normalize line endings and split on double newlines
    blocks = re.split(r"\n{2,}", text)
    paragraphs_list = []
    for block in blocks:
        # Clean up whitespace and ensure it has at least three words
        cleaned = block.strip()
        if len(cleaned.split()) >= 3:
            paragraphs_list.append(cleaned)
    return paragraphs_list


def update_vectorizers() -> None:
    """Recompute TF‑IDF vectorizers for paragraphs and resumés."""
    global paragraph_vectorizer, paragraph_vectors, resume_vectorizer, resume_vectors, resume_id_order
    # Paragraph vectors
    if paragraphs:
        documents = [p["text"] for p in paragraphs]
        paragraph_vectorizer = TfidfVectorizer(stop_words="english")
        paragraph_vectors = paragraph_vectorizer.fit_transform(documents)
    else:
        paragraph_vectorizer = None
        paragraph_vectors = None
    # Resume vectors
    if resumes:
        resume_id_order = list(resumes.keys())
        docs = [resumes[rid]["text"] for rid in resume_id_order]
        resume_vectorizer = TfidfVectorizer(stop_words="english")
        resume_vectors = resume_vectorizer.fit_transform(docs)
    else:
        resume_vectorizer = None
        resume_vectors = None
        resume_id_order = []


def compute_snippet_for_resume(resume_id: int, query: str) -> str:
    """Return the most relevant paragraph snippet from a resume for a query."""
    if paragraph_vectorizer is None or paragraph_vectors is None:
        return ""
    # Collect indices of paragraphs belonging to this resume
    indices = [i for i, p in enumerate(paragraphs) if p["resume_id"] == resume_id]
    if not indices:
        return ""
    # Transform query using the paragraph vectorizer
    q_vec = paragraph_vectorizer.transform([query])
    # Compute similarities only for the selected indices
    subset = paragraph_vectors[indices]
    sims = cosine_similarity(q_vec, subset)[0]
    # Find the index with highest similarity
    if len(sims) == 0:
        return ""
    best_local_idx = sims.argmax()
    best_para = paragraphs[indices[best_local_idx]]
    # Return first 200 characters of the paragraph as snippet
    snippet = best_para["text"]
    return snippet[:200] + ("..." if len(snippet) > 200 else "")


# ----------------------------------------------------------------------------
# FastAPI application setup

app = FastAPI(title="ResumeRAG API")

# Enable CORS for all origins (open during judging)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and configure templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


# ----------------------------------------------------------------------------
# Seed data loading

def load_seed_data() -> None:
    """Load seed users, resumés and jobs if available."""
    global next_user_id, next_resume_id, next_job_id
    # Create default users
    for username, password, role in [("recruiter", "recruiter123", "recruiter"), ("candidate", "candidate123", "candidate")]:
        uid = next_user_id
        next_user_id += 1
        users[uid] = {
            "id": uid,
            "username": username,
            "password_hash": hash_password(password),
            "role": role,
        }
        resumes_by_user[uid] = []
    # Optionally load seed resumés from seed_resumes directory
    seed_dir = os.path.join(BASE_DIR, "seed_resumes")
    if os.path.isdir(seed_dir):
        candidate_user_id = 2  # assign to the default candidate for demo
        for fname in os.listdir(seed_dir):
            path = os.path.join(seed_dir, fname)
            if not os.path.isfile(path):
                continue
            with open(path, "rb") as f:
                data = f.read()
            text = ""
            if fname.lower().endswith(".pdf"):
                text = parse_pdf(data)
            elif fname.lower().endswith(".txt"):
                text = parse_txt(data)
            else:
                continue
            if not text:
                continue
            rid = next_resume_id
            next_resume_id += 1
            paragraphs_list = extract_paragraphs(text)
            resumes[rid] = {
                "id": rid,
                "candidate_id": candidate_user_id,
                "filename": fname,
                "text": text,
                "paragraphs": paragraphs_list,
                "created_at": time.time(),
            }
            resumes_by_user[candidate_user_id].append(rid)
            for para in paragraphs_list:
                paragraphs.append({"resume_id": rid, "candidate_id": candidate_user_id, "text": para})
    # Optionally load seed jobs from seed_jobs.json
    jobs_file = os.path.join(BASE_DIR, "seed_jobs.json")
    if os.path.isfile(jobs_file):
        try:
            with open(jobs_file, "r", encoding="utf-8") as f:
                job_list = json.load(f)
            recruiter_id = 1  # assign jobs to default recruiter
            for job in job_list:
                jid = next_job_id
                next_job_id += 1
                jobs[jid] = {
                    "id": jid,
                    "recruiter_id": recruiter_id,
                    "title": job.get("title", f"Job {jid}"),
                    "description": job.get("description", ""),
                    "created_at": time.time(),
                }
        except Exception:
            pass
    # After loading seed data, compute vectorizers
    update_vectorizers()


@app.on_event("startup")
def startup_event() -> None:
    """Initialize seed data when the application starts."""
    load_seed_data()


# ----------------------------------------------------------------------------
# API endpoints


@app.post("/api/register")
async def register(payload: Dict[str, Any]):
    """Register a new user using a JSON body."""
    global next_user_id
    username = payload.get("username")
    password = payload.get("password")
    role = payload.get("role")
    if not username:
        raise HTTPException(status_code=400, detail={"error": {"code": "FIELD_REQUIRED", "field": "username", "message": "Username is required"}})
    if not password:
        raise HTTPException(status_code=400, detail={"error": {"code": "FIELD_REQUIRED", "field": "password", "message": "Password is required"}})
    if role not in ("candidate", "recruiter"):
        raise HTTPException(status_code=400, detail={"error": {"code": "INVALID_ROLE", "field": "role", "message": "Role must be 'candidate' or 'recruiter'"}})
    # Check username uniqueness
    if any(u["username"] == username for u in users.values()):
        raise HTTPException(status_code=400, detail={"error": {"code": "USERNAME_TAKEN", "field": "username", "message": "Username already exists"}})
    uid = next_user_id
    next_user_id += 1
    users[uid] = {
        "id": uid,
        "username": username,
        "password_hash": hash_password(password),
        "role": role,
    }
    resumes_by_user[uid] = []
    return {"id": uid, "username": username, "role": role}


@app.post("/api/login")
async def login(payload: Dict[str, Any]):
    """Authenticate a user and return a bearer token using JSON body."""
    username = payload.get("username")
    password = payload.get("password")
    if not username or not password:
        raise HTTPException(status_code=400, detail={"error": {"code": "FIELD_REQUIRED", "message": "Username and password are required"}})
    for user in users.values():
        if user["username"] == username and verify_password(password, user["password_hash"]):
            token = create_token(user["id"])
            return {"token": token}
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail={"error": {"code": "UNAUTHORIZED", "message": "Invalid credentials"}})


@app.post("/api/resumes")
async def upload_resumes(request: Request, idempotency_key: Optional[str] = Header(None), user: Dict[str, Any] = Depends(get_current_user), _rate=Depends(enforce_rate_limit)):
    """Upload one or more resumés. Accepts a JSON body: { "files": [ {"filename": ..., "content": ...}, ... ] }.
    The content field must contain a base64 encoded string (without the data URL prefix)."""
    global next_resume_id
    if user["role"] != "candidate":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail={"error": {"code": "FORBIDDEN", "message": "Only candidates may upload resumes"}})
    # Check idempotency key
    if idempotency_key:
        prev = get_idempotent_response(user["id"], idempotency_key)
        if prev is not None:
            return prev
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"error": {"code": "INVALID_BODY", "message": "Invalid JSON body"}})
    files = body.get("files")
    if not files or not isinstance(files, list):
        raise HTTPException(status_code=400, detail={"error": {"code": "FIELD_REQUIRED", "field": "files", "message": "Files field must be a list"}})
    created_records = []
    for f in files:
        filename = f.get("filename") or ""
        content_b64 = f.get("content")
        if not filename or not content_b64:
            continue
        # decode base64
        import base64
        try:
            data = base64.b64decode(content_b64)
        except Exception:
            continue
        lower = filename.lower()
        # If zip, extract
        if lower.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for zip_info in zf.infolist():
                        if zip_info.is_dir():
                            continue
                        inner_name = zip_info.filename
                        with zf.open(zip_info) as file_obj:
                            inner_data = file_obj.read()
                        ext = inner_name.lower().split(".")[-1]
                        text = ""
                        if ext == "pdf":
                            text = parse_pdf(inner_data)
                        elif ext == "txt":
                            text = parse_txt(inner_data)
                        else:
                            continue
                        if not text:
                            continue
                        rid = next_resume_id
                        next_resume_id += 1
                        paragraphs_list = extract_paragraphs(text)
                        resumes[rid] = {
                            "id": rid,
                            "candidate_id": user["id"],
                            "filename": inner_name,
                            "text": text,
                            "paragraphs": paragraphs_list,
                            "created_at": time.time(),
                        }
                        resumes_by_user[user["id"]].append(rid)
                        for para in paragraphs_list:
                            paragraphs.append({"resume_id": rid, "candidate_id": user["id"], "text": para})
                        created_records.append(resumes[rid])
            except zipfile.BadZipFile:
                continue
        else:
            text = ""
            if lower.endswith(".pdf"):
                text = parse_pdf(data)
            elif lower.endswith(".txt"):
                text = parse_txt(data)
            else:
                continue
            if not text:
                continue
            rid = next_resume_id
            next_resume_id += 1
            paragraphs_list = extract_paragraphs(text)
            resumes[rid] = {
                "id": rid,
                "candidate_id": user["id"],
                "filename": filename,
                "text": text,
                "paragraphs": paragraphs_list,
                "created_at": time.time(),
            }
            resumes_by_user[user["id"]].append(rid)
            for para in paragraphs_list:
                paragraphs.append({"resume_id": rid, "candidate_id": user["id"], "text": para})
            created_records.append(resumes[rid])
    update_vectorizers()
    response_data = [{"id": r["id"], "filename": r["filename"]} for r in created_records]
    if idempotency_key:
        store_idempotent_response(user["id"], idempotency_key, response_data)
    return response_data


@app.get("/api/resumes")
async def list_resumes(request: Request, limit: Optional[int] = 10, offset: Optional[int] = 0, q: Optional[str] = None, user: Dict[str, Any] = Depends(get_current_user), _rate=Depends(enforce_rate_limit)):
    """List resumés with optional search and pagination."""
    if limit is None or limit <= 0:
        limit = 10
    if offset is None or offset < 0:
        offset = 0
    # Determine which resumes this user can see
    visible_ids: List[int]
    if user["role"] == "recruiter":
        visible_ids = list(resumes.keys())
    else:
        visible_ids = resumes_by_user.get(user["id"], [])
    # If a query is provided and we have a vectorizer, compute similarities
    ranked_ids: List[int] = []
    if q and resume_vectorizer and resume_vectors is not None and visible_ids:
        try:
            q_vec = resume_vectorizer.transform([q])
            sims = cosine_similarity(q_vec, resume_vectors)[0]
            # Pair each resume id with its similarity score
            pairs = []
            for idx, rid in enumerate(resume_id_order):
                if rid in visible_ids:
                    pairs.append((rid, sims[idx]))
            # Sort descending by similarity then by id to be deterministic
            pairs.sort(key=lambda x: (-x[1], x[0]))
            ranked_ids = [rid for rid, _ in pairs]
        except Exception:
            ranked_ids = visible_ids
    else:
        # Default ordering: newest first (descending created_at)
        ranked_ids = sorted(visible_ids, key=lambda rid: resumes[rid]["created_at"], reverse=True)
    # Apply pagination
    sliced = ranked_ids[offset:offset + limit]
    next_offset = offset + limit if offset + limit < len(ranked_ids) else None
    items = []
    for rid in sliced:
        res = resumes[rid]
        candidate = users[res["candidate_id"]]
        candidate_name = candidate["username"]
        snippet = ""
        if q:
            snippet = compute_snippet_for_resume(rid, q)
        # Redact PII for non‑recruiters viewing others' resumés
        if user["role"] != "recruiter" and res["candidate_id"] != user["id"]:
            snippet = redact_pii(snippet)
        items.append({
            "id": rid,
            "filename": res["filename"],
            "candidate": candidate_name,
            "snippet": snippet,
        })
    return {"items": items, "next_offset": next_offset}


@app.get("/api/resumes/{resume_id}")
async def get_resume(resume_id: int, user: Dict[str, Any] = Depends(get_current_user), _rate=Depends(enforce_rate_limit)):
    """Retrieve a single resumé by ID.  Applies PII redaction for non‑recruiters."""
    res = resumes.get(resume_id)
    if not res:
        raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "Resume not found"}})
    if user["role"] != "recruiter" and res["candidate_id"] != user["id"]:
        # Non‑recruiters can only see their own resumés
        raise HTTPException(status_code=403, detail={"error": {"code": "FORBIDDEN", "message": "Access denied"}})
    # If viewer is not recruiter and not owner, redaction occurs (should not happen due to above check)
    redacted_text = res["text"]
    if user["role"] != "recruiter" and res["candidate_id"] == user["id"]:
        # Candidate can view their own without redaction
        redacted_text = res["text"]
    elif user["role"] != "recruiter":
        redacted_text = redact_pii(res["text"])
    return {
        "id": res["id"],
        "filename": res["filename"],
        "candidate": users[res["candidate_id"]]["username"],
        "text": redacted_text,
    }


@app.post("/api/ask")
async def ask_query(payload: Dict[str, Any], user: Dict[str, Any] = Depends(get_current_user), _rate=Depends(enforce_rate_limit)):
    """Ask a free‑form question across all uploaded resumés using JSON body.  Body must contain 'query' and 'k'."""
    query = payload.get("query")
    k = payload.get("k")
    if not query:
        raise HTTPException(status_code=400, detail={"error": {"code": "FIELD_REQUIRED", "field": "query", "message": "Query is required"}})
    try:
        k = int(k)
    except Exception:
        k = None
    if not k or k <= 0:
        raise HTTPException(status_code=400, detail={"error": {"code": "INVALID_FIELD", "field": "k", "message": "k must be positive"}})
    if paragraph_vectorizer is None or paragraph_vectors is None:
        return []
    try:
        q_vec = paragraph_vectorizer.transform([query])
        sims = cosine_similarity(q_vec, paragraph_vectors)[0]
        pairs = [(idx, score) for idx, score in enumerate(sims)]
        pairs.sort(key=lambda x: (-x[1], x[0]))
        results = []
        seen = set()
        for idx, score in pairs:
            para = paragraphs[idx]
            rid = para["resume_id"]
            # Check access
            if user["role"] != "recruiter" and rid not in resumes_by_user.get(user["id"], []):
                continue
            key_seen = rid if user["role"] != "recruiter" else (rid, para["text"])
            if key_seen in seen:
                continue
            text = para["text"]
            # Redact for non‑recruiters viewing others
            if user["role"] != "recruiter" and resumes[rid]["candidate_id"] != user["id"]:
                text = redact_pii(text)
            results.append({
                "resume_id": rid,
                "candidate": users[resumes[rid]["candidate_id"]]["username"],
                "snippet": text[:200] + ("..." if len(text) > 200 else "")
            })
            seen.add(key_seen)
            if len(results) >= k:
                break
        return results
    except Exception:
        return []


@app.post("/api/jobs")
async def create_job(payload: Dict[str, Any], idempotency_key: Optional[str] = Header(None), user: Dict[str, Any] = Depends(get_current_user), _rate=Depends(enforce_rate_limit)):
    """Create a new job posting using a JSON body. Recruiter only."""
    global next_job_id
    if user["role"] != "recruiter":
        raise HTTPException(status_code=403, detail={"error": {"code": "FORBIDDEN", "message": "Only recruiters may create jobs"}})
    title = payload.get("title")
    description = payload.get("description")
    if not title:
        raise HTTPException(status_code=400, detail={"error": {"code": "FIELD_REQUIRED", "field": "title", "message": "Title is required"}})
    if not description:
        raise HTTPException(status_code=400, detail={"error": {"code": "FIELD_REQUIRED", "field": "description", "message": "Description is required"}})
    if idempotency_key:
        prev = get_idempotent_response(user["id"], idempotency_key)
        if prev is not None:
            return prev
    jid = next_job_id
    next_job_id += 1
    jobs[jid] = {
        "id": jid,
        "recruiter_id": user["id"],
        "title": title,
        "description": description,
        "created_at": time.time(),
    }
    response = jobs[jid]
    if idempotency_key:
        store_idempotent_response(user["id"], idempotency_key, response)
    return response


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: int, user: Dict[str, Any] = Depends(get_current_user), _rate=Depends(enforce_rate_limit)):
    """Retrieve a job by ID.  Recruiter only."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "Job not found"}})
    if user["role"] != "recruiter" or job["recruiter_id"] != user["id"]:
        raise HTTPException(status_code=403, detail={"error": {"code": "FORBIDDEN", "message": "Access denied"}})
    return job


@app.get("/api/jobs")
async def list_jobs(user: Dict[str, Any] = Depends(get_current_user), _rate=Depends(enforce_rate_limit)):
    """List jobs belonging to the recruiter."""
    if user["role"] != "recruiter":
        raise HTTPException(status_code=403, detail={"error": {"code": "FORBIDDEN", "message": "Only recruiters may list jobs"}})
    # Filter jobs created by this recruiter
    user_jobs = [job for job in jobs.values() if job["recruiter_id"] == user["id"]]
    # Sort by newest
    user_jobs.sort(key=lambda j: j["created_at"], reverse=True)
    return user_jobs


@app.post("/api/jobs/{job_id}/match")
async def match_job(job_id: int, payload: Dict[str, Any], user: Dict[str, Any] = Depends(get_current_user), _rate=Depends(enforce_rate_limit)):
    """Match candidates against a job description. Accepts JSON body with 'top_n'."""
    if user["role"] != "recruiter":
        raise HTTPException(status_code=403, detail={"error": {"code": "FORBIDDEN", "message": "Only recruiters may match jobs"}})
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "Job not found"}})
    if job["recruiter_id"] != user["id"]:
        raise HTTPException(status_code=403, detail={"error": {"code": "FORBIDDEN", "message": "Access denied"}})
    top_n = payload.get("top_n")
    try:
        top_n = int(top_n)
    except Exception:
        top_n = None
    if not top_n or top_n <= 0:
        raise HTTPException(status_code=400, detail={"error": {"code": "INVALID_FIELD", "field": "top_n", "message": "top_n must be positive"}})
    if not resumes:
        return []
    if resume_vectorizer is None or resume_vectors is None:
        return []
    try:
        q_vec = resume_vectorizer.transform([job["description"]])
        sims = cosine_similarity(q_vec, resume_vectors)[0]
    except Exception:
        return []
    pairs = []
    for idx, rid in enumerate(resume_id_order):
        pairs.append((rid, sims[idx]))
    pairs.sort(key=lambda x: (-x[1], x[0]))
    job_tokens = set(re.findall(r"\b\w+\b", job["description"].lower()))
    stop_words = set()
    if resume_vectorizer and hasattr(resume_vectorizer, "stop_words_"):
        stop_words = resume_vectorizer.stop_words_
    job_keywords = {tok for tok in job_tokens if tok not in stop_words and len(tok) > 2}
    results = []
    count = 0
    for rid, score in pairs:
        if score <= 0:
            continue
        res = resumes[rid]
        evidence_snippet = compute_snippet_for_resume(rid, job["description"])
        missing = []
        resume_words = set(re.findall(r"\b\w+\b", res["text"].lower()))
        for kw in job_keywords:
            if kw not in resume_words:
                missing.append(kw)
        results.append({
            "resume_id": rid,
            "candidate": users[res["candidate_id"]]["username"],
            "score": float(score),
            "evidence": evidence_snippet,
            "missing_requirements": missing,
        })
        count += 1
        if count >= top_n:
            break
    return results


# ----------------------------------------------------------------------------
# UI routes


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})


@app.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request):
    return templates.TemplateResponse("jobs.html", {"request": request})


@app.get("/candidates/{resume_id}", response_class=HTMLResponse)
async def candidate_page(request: Request, resume_id: int):
    return templates.TemplateResponse("candidate.html", {"request": request, "resume_id": resume_id})


# ----------------------------------------------------------------------------
# Main entry point when running directly

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)