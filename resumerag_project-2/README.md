# ResumeRAG Hackathon Project

This repository contains a complete implementation for the **ResumeRAG** challenge. The goal is to upload and search resumés, ask free‑form questions across all uploaded documents, create job postings, and match candidates against them. Everything – from API design to a small demo UI – lives in this repository so you can run the project locally without any additional dependencies.

## Features

- **FastAPI back‑end** providing a clean REST API
  - Authentication via token (register & login)
  - Resumé upload with support for multiple files or ZIP archives
  - Full‑text search across resumés with deterministic ranking and pagination
  - Ask a question across all resumés and return the best matching snippets
  - Job posting creation and retrieval
  - Job–candidate matching that returns evidence and missing requirements
  - Built‑in rate limiting (60 requests per minute per user)
  - Idempotent create operations via the `Idempotency-Key` header
  - Uniform error format as required by the specification
  - CORS is enabled for all origins during judging

- **Minimal front‑end** served with the API to exercise all endpoints
  - Login and registration pages
  - Candidate resumé upload page
  - Recruiter search page to filter resumés
  - Jobs dashboard to create jobs and trigger matching
  - Candidate detail page with resumé text (PII redacted for non‑recruiters)

- **Seed data** is loaded on start (see `app.py`) including a recruiter and a candidate account for convenience. You can adjust or extend the seed data as needed.

## Running the Server

1. **Install prerequisites** – no external packages need to be downloaded. The code uses only libraries already present in the environment such as FastAPI, uvicorn, scikit‑learn and Jinja2.

2. **Start the server**:

   ```bash
   python3 app.py
   ```

   By default the server listens on `http://localhost:8000`. When developing you can enable live reload via `uvicorn`:

   ```bash
   uvicorn app:app --reload
   ```

3. **Visit the UI** using your browser at `http://localhost:8000/`. The landing page links to login, register, upload, search and jobs. You can interact with the API directly as well. FastAPI automatically exposes interactive API documentation at `/docs` and `/redoc`.

## API Summary

Below is a condensed summary of the provided API. Each response body is shortened for brevity – consult the code or the interactive Swagger documentation for full schemas.

| Endpoint | Method | Description |
|---|---|---|
| `/api/register` | `POST` | Register a new user. Body: `{ "username", "password", "role" }` (role is `candidate` or `recruiter`). Returns the created user object. |
| `/api/login` | `POST` | Authenticate a user. Body: `{ "username", "password" }`. Returns `{ "token" }`. |
| `/api/resumes` | `POST` | Upload one or more resumés. Accepts multipart form data; files may be PDF, TXT or ZIP archives containing either. Requires authentication as a candidate. Returns a list of created resumé records. |
| `/api/resumes` | `GET` | List resumés. Supports query parameters `limit`, `offset` and `q` (search query). Returns `{ "items", "next_offset" }`. Recruiters see all resumés; candidates see only their own. |
| `/api/resumes/{id}` | `GET` | Retrieve a single resumé by id. Requires appropriate permissions. |
| `/api/ask` | `POST` | Ask a free‑form question across all uploaded resumés. Body: `{ "query", "k" }` where `k` is the number of snippets to return. Returns a list of { resumé id, candidate id, snippet text }. |
| `/api/jobs` | `POST` | Create a job posting. Body: `{ "title", "description" }`. Recruiter only. Returns the created job. |
| `/api/jobs/{id}` | `GET` | Retrieve a job by id. Recruiter only. |
| `/api/jobs/{id}/match` | `POST` | Match candidates against a job. Body: `{ "top_n" }`. Recruiter only. Returns the top `n` matches with evidence and missing keywords. |

### Pagination

`GET /api/resumes` accepts `limit` and `offset` query parameters. If omitted, a default of 10 is used. The response includes an `items` array and a `next_offset` field – if `next_offset` is `null` there are no more pages. Use `?limit=&offset=` to fetch subsequent pages.

### Idempotency

All POST operations that create resources (`/api/resumes` and `/api/jobs`) accept an `Idempotency-Key` header. If the same user repeats a call with the same key the server returns the original result instead of creating duplicates. You can safely retry failed requests with the same header.

### Rate Limiting

Each authenticated user is limited to 60 API requests per rolling minute. When the limit is exceeded the server responds with HTTP status 429 and a body: `{ "error": { "code": "RATE_LIMIT" } }`.

### Error Format

All errors returned by the API follow this structure:

```json
{
  "error": {
    "code": "FIELD_REQUIRED",
    "field": "username",
    "message": "Username is required"
  }
}
```

The `code` field identifies the type of error (e.g. `FIELD_REQUIRED`, `UNAUTHORIZED`, `NOT_FOUND`, etc.), `field` points to the offending parameter when applicable, and `message` provides a human readable explanation.

## Test Users and Seed Data

Two users are preloaded when the application starts:

- **Recruiter** – username: `recruiter`, password: `recruiter123`, role: `recruiter`
- **Candidate** – username: `candidate`, password: `candidate123`, role: `candidate`

You can register more users via the `/api/register` endpoint or the registration page.

## Seed Data for Demos

If you would like to preload some resumés or jobs, you can place PDF or TXT files into the `seed_resumes/` directory and job descriptions into `seed_jobs.json` before starting the server. On launch the application will load this data automatically. See the comments in `app.py` for details.

## Deployment Notes

The application is completely self‑contained and uses only standard Python modules plus a handful of common libraries already available in the environment (FastAPI, Jinja2, scikit‑learn and PyMuPDF). To deploy in another environment:

1. Ensure Python 3.9+ is installed.
2. Install FastAPI, uvicorn, scikit‑learn, jinja2 and PyMuPDF (fitz). For example:

   ```bash
   pip install fastapi uvicorn scikit-learn jinja2 pymupdf
   ```

3. Copy the contents of this repository into your project directory and run `python3 app.py`.

## License

This project is provided for the hackathon and does not carry any specific software license. Feel free to reuse or adapt it for educational or evaluation purposes.