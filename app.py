import os
import networkx as nx
import re
import json
import base64
import tempfile
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # retained
import io
import os
import re
import json
import base64
import tempfile
import subprocess
import logging
from io import BytesIO
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi import FastAPI
from dotenv import load_dotenv

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Optional image conversion
try:
    from PIL import Image
    PIL_READY = True
except Exception:
    PIL_READY = False

# Optional PDF libs
try:
    import pdfplumber
    PDFPLUMBER_READY = True
except Exception:
    PDFPLUMBER_READY = False

try:
    import tabula  # requires Java runtime on the host
    TABULA_READY = True
except Exception:
    TABULA_READY = False

# LangChain / LLM imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="TDS Data Analyst Agent")

# ---- CORS (needed if prof site calls from browser) ----
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- AI Pipe / OpenAI-compatible setup ----
def get_ai_pipe_token() -> str:
    """
    Pick the first defined OPENAI_API_KEY_* from env.
    Falls back to OPENAI_API_KEY if present.
    """
    for i in range(1, 6):
        k = os.getenv(f"OPENAI_API_KEY_{i}")
        if k:
            return k
    return os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = get_ai_pipe_token()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openai/v1")
if not OPENAI_API_KEY:
    raise RuntimeError("No OPENAI_API_KEY or OPENAI_API_KEY_{i} found in environment.")

# LangChain OpenAI chat model pointing at AI Pipe
chat_model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),  # <- default to gpt-5-mini
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    timeout=int(os.getenv("LLM_TIMEOUT_SECONDS", "240")),
)

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 240))


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML interface"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1><p>Please ensure index.html is in the same directory as app.py</p>", status_code=404)


def parse_keys_and_types(raw_qs: str):
    """
    Parses the key/type section from the questions file.
    Returns:
        ordered_keys: list of keys in order
        cast_lookup: dict key -> casting function
    """
    import re
    keytype_pattern = r"-\s*`([^`]+)`\s*:\s*(\w+)"
    keytype_matches = re.findall(keytype_pattern, raw_qs)
    cast_map = {
        "number": float,
        "string": str,
        "integer": int,
        "int": int,
        "float": float
    }
    cast_lookup = {key: cast_map.get(t.lower(), str) for key, t in keytype_matches}
    ordered_keys = [k for k, _ in keytype_matches]
    return ordered_keys, cast_lookup


# -----------------------------
# PDF helpers (minimal, layered)
# -----------------------------
def _combine_tables_with_index(tables: List[pd.DataFrame]) -> pd.DataFrame:
    clean_tables = []
    for i, t in enumerate(tables):
        if t is None or len(t) == 0:
            continue
        df = t.copy()
        df.columns = pd.Index([str(c) for c in df.columns])
        df.insert(0, "table_index", i)
        clean_tables.append(df)
    if not clean_tables:
        return pd.DataFrame()
    return pd.concat(clean_tables, ignore_index=True, sort=False)


def pdf_bytes_to_dataframe(pdf_bytes: bytes) -> pd.DataFrame:
    """
    Best-effort PDF to DataFrame:
      1) Try tabula (lattice/stream) to extract tables (Java required).
      2) Fallback to pdfplumber to get tables & text.
      3) Final fallback: a single-row DataFrame with a note if nothing is extractable.
    """
    # Try Tabula
    if TABULA_READY:
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        try:
            tmp.write(pdf_bytes)
            tmp.flush()
            tmp.close()
            tabs = []
            try:
                tabs = tabula.read_pdf(tmp.name, pages="all", multiple_tables=True, lattice=True)
            except Exception:
                pass
            if not tabs:
                try:
                    tabs = tabula.read_pdf(tmp.name, pages="all", multiple_tables=True, stream=True)
                except Exception:
                    pass
            if tabs:
                df_tabs = _combine_tables_with_index(tabs)
                if not df_tabs.empty:
                    return df_tabs
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    # Fallback to pdfplumber
    if PDFPLUMBER_READY:
        try:
            tables_collected = []
            texts = []
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    try:
                        t = page.extract_text() or ""
                        if t:
                            texts.append(t)
                    except Exception:
                        pass
                    try:
                        raw_tables = page.extract_tables()
                        for rt in raw_tables or []:
                            if not rt:
                                continue
                            header = rt[0]
                            body = rt[1:] if all(len(r) == len(header) for r in rt[1:]) else rt
                            df_t = pd.DataFrame(body, columns=[str(c) for c in header] if body is rt[1:] else None)
                            tables_collected.append(df_t)
                    except Exception:
                        pass
            if tables_collected:
                df_all = _combine_tables_with_index(tables_collected)
                if not df_all.empty:
                    return df_all
            full_text = "\n".join(texts).strip()
            if full_text:
                return pd.DataFrame({"text": [full_text]})
        except Exception:
            pass

    return pd.DataFrame({
        "text": [""],
        "note": ["PDF parsing libraries not available or no extractable content found."],
    })


# -----------------------------
# DB helpers (SQLite / DuckDB)
# -----------------------------
def _detect_db_backend(tmp_path: str) -> str:
    """
    Returns 'sqlite', 'duckdb', or '' if neither opens.
    """
    # Try SQLite
    try:
        import sqlite3
        con = sqlite3.connect(tmp_path)
        con.execute("SELECT 1")
        con.close()
        return "sqlite"
    except Exception:
        pass

    # Try DuckDB
    try:
        import duckdb
        con = duckdb.connect(tmp_path)
        con.execute("SELECT 1")
        con.close()
        return "duckdb"
    except Exception:
        pass

    return ""


def _sqlite_list_tables(path: str) -> List[str]:
    import sqlite3
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    rows = cur.fetchall()
    con.close()
    return [r[0] for r in rows]


def _sqlite_sample_table(path: str, table: str, limit: int = 5) -> pd.DataFrame:
    import sqlite3
    con = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT {int(limit)}", con)
    finally:
        con.close()
    return df


def _duckdb_list_tables(path: str) -> List[str]:
    import duckdb
    con = duckdb.connect(path)
    try:
        df = con.execute("SELECT table_schema, table_name FROM information_schema.tables WHERE table_type='BASE TABLE'").df()
        # Prefer 'main' schema if exists; otherwise include all
        if "table_schema" in df and "table_name" in df:
            return [f"{r['table_schema']}.{r['table_name']}" if r["table_schema"] not in ("main", "") else r["table_name"] for _, r in df.iterrows()]
        return []
    finally:
        con.close()


def _duckdb_sample_table(path: str, table: str, limit: int = 5) -> pd.DataFrame:
    import duckdb
    con = duckdb.connect(path)
    try:
        df = con.execute(f"SELECT * FROM {table} LIMIT {int(limit)}").df()
    finally:
        con.close()
    return df


def db_bytes_preview_dataframe(db_bytes: bytes) -> Dict[str, Any]:
    """
    Write bytes to a temp .db file, detect backend, list tables, and return a small preview DataFrame
    plus metadata (backend, tables). The DataFrame is from the first discovered table (up to 500 rows).
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.write(db_bytes)
    tmp.flush()
    tmp.close()
    path = tmp.name
    backend = _detect_db_backend(path)
    meta = {"backend": backend, "tables": []}
    try:
        if backend == "sqlite":
            tables = _sqlite_list_tables(path)
            meta["tables"] = tables
            if tables:
                df = _sqlite_sample_table(path, tables[0], limit=500)
                return {"df": df, "meta": meta, "path": path}
            return {"df": pd.DataFrame(), "meta": meta, "path": path}
        elif backend == "duckdb":
            tables = _duckdb_list_tables(path)
            meta["tables"] = tables
            if tables:
                df = _duckdb_sample_table(path, tables[0], limit=500)
                return {"df": df, "meta": meta, "path": path}
            return {"df": pd.DataFrame(), "meta": meta, "path": path}
        else:
            # Unknown DB, still return a path so the sandbox can try queries if possible
            return {"df": pd.DataFrame({"note": ["Unknown .db format"]}), "meta": meta, "path": path}
    except Exception as e:
        return {"df": pd.DataFrame({"error": [str(e)]}), "meta": meta, "path": path}


# -----------------------------
# Tools
# -----------------------------
@tool
def scrape_url_to_dataframe(target_url: str) -> Dict[str, Any]:
    """
    Fetch a URL and return data as a DataFrame (supports HTML tables, CSV, Excel, Parquet, JSON, PDF, .db (best-effort), and plain text).
    Always returns {"status": "success", "data": [...], "columns": [...]} if fetch works.
    """
    print(f"Scraping URL: {target_url}")
    try:
        from io import BytesIO, StringIO
        from bs4 import BeautifulSoup

        req_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.google.com/",
        }

        r = requests.get(target_url, headers=req_headers, timeout=20)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "").lower()

        frame = None

        # --- DB from URL ---
        if target_url.lower().endswith((".db", ".sqlite", ".duckdb")) or ("application/octet-stream" in content_type and target_url.lower().endswith(".db")):
            preview = db_bytes_preview_dataframe(r.content)
            frame = preview["df"]

        # --- PDF ---
        elif "application/pdf" in content_type or target_url.lower().endswith(".pdf"):
            pdf_bytes = r.content
            frame = pdf_bytes_to_dataframe(pdf_bytes)

        # --- CSV ---
        elif "text/csv" in content_type or target_url.lower().endswith(".csv"):
            frame = pd.read_csv(BytesIO(r.content))

        # --- Excel ---
        elif any(target_url.lower().endswith(ext) for ext in (".xls", ".xlsx")) or "spreadsheetml" in content_type:
            frame = pd.read_excel(BytesIO(r.content))

        # --- Parquet ---
        elif target_url.lower().endswith(".parquet"):
            frame = pd.read_parquet(BytesIO(r.content))

        # --- JSON ---
        elif "application/json" in content_type or target_url.lower().endswith(".json"):
            try:
                data = r.json()
                frame = pd.json_normalize(data)
            except Exception:
                frame = pd.DataFrame([{"text": r.text}])

        # --- HTML / Fallback ---
        elif "text/html" in content_type or re.search(r'/wiki/|\.org|\.com', target_url, re.IGNORECASE):
            html_text = r.text
            try:
                html_tables = pd.read_html(StringIO(html_text), flavor="bs4")
                if html_tables:
                    frame = html_tables[0]
            except ValueError:
                pass

            if frame is None:
                soup_obj = BeautifulSoup(html_text, "html.parser")
                page_text = soup_obj.get_text(separator="\n", strip=True)
                frame = pd.DataFrame({"text": [page_text]})

        else:
            frame = pd.DataFrame({"text": [r.text]})

        frame.columns = frame.columns.map(str).str.replace(r'\[.*\]', '', regex=True).str.strip()

        return {
            "status": "success",
            "data": frame.to_dict(orient="records"),
            "columns": frame.columns.tolist()
        }

    except Exception as err:
        return {"status": "error", "message": str(err)}


# -----------------------------
# Utilities for executing code safely
# -----------------------------
def clean_llm_output(llm_text: str) -> Dict:
    """
    Extract JSON object from LLM output robustly.
    Returns dict or {"error": "..."}
    """
    try:
        if not llm_text:
            return {"error": "Empty LLM output"}
        content = re.sub(r"^```(?:json)?\s*", "", llm_text.strip())
        content = re.sub(r"\s*```$", "", content)
        lbrace = content.find("{")
        rbrace = content.rfind("}")
        if lbrace == -1 or rbrace == -1 or rbrace <= lbrace:
            return {"error": "No JSON object found in LLM output", "raw": content}
        json_candidate = content[lbrace:rbrace+1]
        try:
            return json.loads(json_candidate)
        except Exception as e:
            for i in range(rbrace, lbrace, -1):
                cand = content[lbrace:i+1]
                try:
                    return json.loads(cand)
                except Exception:
                    continue
            return {"error": f"JSON parsing failed: {str(e)}", "raw": json_candidate}
    except Exception as e:
        return {"error": str(e)}

# --- helper: convert bytes -> data URI for multimodal LLM ---
def _bytes_to_data_uri(img_bytes: bytes, mime="image/png") -> str:
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"

def run_vision_extraction(questions_text: str, image_uris: list[str]) -> Dict:
    """
    Send images + instructions to a vision-capable LLM and expect JSON-only output.
    No Python OCR: the model must visually read the images.
    """
    system_msg = (
        "You are a precise data extraction assistant. "
        "You will be given one or more images (e.g., scanned bank passbooks) and a set of questions. "
        "Answer ONLY by visually inspecting the images. "
        "DO NOT suggest Python OCR libraries like pytesseract; they are unavailable. "
        "Return strictly valid JSON as the final answer. No extra text."
    )

    msg_content = [{"type": "text", "text": questions_text}]
    for uri in image_uris:
        msg_content.append({"type": "image_url", "image_url": {"url": uri}})

    resp = chat_model.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=msg_content),
    ])
    text = resp.content if hasattr(resp, "content") else str(resp)
    parsed = clean_llm_output(text)
    return parsed


SCRAPE_FUNC = r'''
from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5
        )
        response.raise_for_status()
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": [],
            "columns": []
        }

    soup = BeautifulSoup(response.text, "html.parser")
    tables = pd.read_html(response.text)

    if tables:
        df = tables[0]  # Take first table
        df.columns = [str(c).strip() for c in df.columns]
        df.columns = [str(col) for col in df.columns]
        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns)
        }
    else:
        text_data = soup.get_text(separator="\n", strip=True)
        detected_cols = set(re.findall(r"\b[A-Z][a-zA-Z ]{2,15}\b", text_data))
        df = pd.DataFrame([{}])
        for col in detected_cols:
            df[col] = None
        if df.empty:
            df["text"] = [text_data]
        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns)
        }
'''


def write_and_run_temp_python(code: str, injected_pickle: str = None, timeout: int = 60,
                              injected_db_path: str = None, injected_db_backend: str = None) -> Dict[str, Any]:
    """
    Write a temp python file which:
      - provides a safe environment (imports)
      - loads df/from pickle if provided into df and data variables
      - defines helpers:
          plot_to_base64(), list_tables(), read_table(), sql_to_dataframe()
        (DB helpers active when a DB path is provided)
      - executes the user code (which should populate `results` dict)
      - prints json.dumps({"status":"success","result":results})
    """
    bootstrap_lines = [
        "import json, sys, gc",
        "import pandas as pd, numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "from io import BytesIO",
        "import base64",
    ]
    if PIL_READY:
        bootstrap_lines.append("from PIL import Image")
    if injected_db_path:
        bootstrap_lines.append(f"DB_PATH = r'''{injected_db_path}'''")
        bootstrap_lines.append(f"DB_BACKEND = r'''{injected_db_backend or ''}'''")
    else:
        bootstrap_lines.append("DB_PATH = ''")
        bootstrap_lines.append("DB_BACKEND = ''")

    if injected_pickle:
        bootstrap_lines.append(f"df = pd.read_pickle(r'''{injected_pickle}''')\n")
        bootstrap_lines.append("data = df.to_dict(orient='records')\n")
    else:
        bootstrap_lines.append("data = globals().get('data', {})\n")

    plot_helper = r'''
def plot_to_base64(max_bytes=100000):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_bytes = buf.getvalue()
    if len(img_bytes) <= max_bytes:
        return base64.b64encode(img_bytes).decode('ascii')
    for dpi in [80, 60, 50, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return base64.b64encode(b).decode('ascii')
    try:
        from PIL import Image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=40)
        buf.seek(0)
        im = Image.open(buf)
        out_buf = BytesIO()
        im.save(out_buf, format='WEBP', quality=80, method=6)
        out_buf.seek(0)
        ob = out_buf.getvalue()
        if len(ob) <= max_bytes:
            return base64.b64encode(ob).decode('ascii')
        out_buf = BytesIO()
        im.save(out_buf, format='WEBP', quality=60, method=6)
        out_buf.seek(0)
        ob = out_buf.getvalue()
        if len(ob) <= max_bytes:
            return base64.b64encode(ob).decode('ascii')
    except Exception:
        pass
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')
'''

    # DB helpers exposed to the agent code
    db_helpers = r'''
def list_tables():
    """
    Returns a list of table names for the uploaded DB (SQLite or DuckDB).
    """
    if not DB_PATH or not DB_BACKEND:
        return []
    if DB_BACKEND.lower() == 'sqlite':
        import sqlite3
        con = sqlite3.connect(DB_PATH)
        try:
            cur = con.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            return [r[0] for r in cur.fetchall()]
        finally:
            con.close()
    elif DB_BACKEND.lower() == 'duckdb':
        import duckdb
        con = duckdb.connect(DB_PATH)
        try:
            df = con.execute("SELECT table_schema, table_name FROM information_schema.tables WHERE table_type='BASE TABLE'").df()
            out = []
            for _, r in df.iterrows():
                schema = str(r.get('table_schema', '') or '')
                name = str(r.get('table_name', '') or '')
                out.append(f"{schema}.{name}" if schema not in ('main','') else name)
            return out
        finally:
            con.close()
    return []

def read_table(table_name, limit=None):
    """
    Returns a pandas DataFrame for `table_name` with optional LIMIT.
    """
    if not DB_PATH or not DB_BACKEND:
        return pd.DataFrame()
    if DB_BACKEND.lower() == 'sqlite':
        import sqlite3
        con = sqlite3.connect(DB_PATH)
        try:
            q = f"SELECT * FROM {table_name}" + (f" LIMIT {int(limit)}" if limit else "")
            return pd.read_sql_query(q, con)
        finally:
            con.close()
    elif DB_BACKEND.lower() == 'duckdb':
        import duckdb
        con = duckdb.connect(DB_PATH)
        try:
            q = f"SELECT * FROM {table_name}" + (f" LIMIT {int(limit)}" if limit else "")
            return con.execute(q).df()
        finally:
            con.close()
    return pd.DataFrame()

def sql_to_dataframe(sql):
    """
    Executes a SQL query against the uploaded DB and returns a pandas DataFrame.
    """
    if not DB_PATH or not DB_BACKEND:
        return pd.DataFrame()
    if DB_BACKEND.lower() == 'sqlite':
        import sqlite3
        con = sqlite3.connect(DB_PATH)
        try:
            return pd.read_sql_query(sql, con)
        finally:
            con.close()
    elif DB_BACKEND.lower() == 'duckdb':
        import duckdb
        con = duckdb.connect(DB_PATH)
        try:
            return con.execute(sql).df()
        finally:
            con.close()
    return pd.DataFrame()
'''

    script_buf = []
    script_buf.extend(bootstrap_lines)
    script_buf.append(plot_helper)
    if injected_db_path:
        script_buf.append(db_helpers)
    script_buf.append(SCRAPE_FUNC)
    script_buf.append("\nresults = {}\n")
    script_buf.append(code)
    script_buf.append("\nprint(json.dumps({'status':'success','result':results}, default=str), flush=True)\n")

    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
    tmpfile.write("\n".join(script_buf))
    tmpfile.flush()
    tmp_pathname = tmpfile.name
    tmpfile.close()

    try:
        proc = subprocess.run([sys.executable, tmp_pathname],
                              capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return {"status": "error", "message": proc.stderr.strip() or proc.stdout.strip()}
        stdout_text = proc.stdout.strip()
        try:
            parsed_json = json.loads(stdout_text)
            return parsed_json
        except Exception as e:
            return {"status": "error", "message": f"Could not parse JSON output: {str(e)}", "raw": stdout_text}
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Execution timed out"}
    finally:
        try:
            os.unlink(tmp_pathname)
            if injected_pickle and os.path.exists(injected_pickle):
                os.unlink(injected_pickle)
        except Exception:
            pass


# -----------------------------
# LLM agent setup
# -----------------------------
tools = [scrape_url_to_dataframe]

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a full-stack autonomous data analyst agent.

You will receive:
- A set of **rules** for this request (these rules may differ depending on whether a dataset is uploaded or not)
- One or more **questions**
- An optional **dataset preview**

You must:
1. Follow the provided rules exactly.
2. Return only a valid JSON object — no extra commentary or formatting.
3. The JSON must contain:
   - "questions": [ list of original question strings exactly as provided ]
   - "code": "..." (Python code that creates a dict called `results` with each question string as a key and its computed answer as the value)
4. Your Python code will run in a sandbox with:
   - pandas, numpy, matplotlib available
   - A helper function `plot_to_base64(max_bytes=100000)` for generating base64-encoded images under 100KB.
5. When returning plots, always use `plot_to_base64()` to keep image sizes small.
6. Make sure all variables are defined before use, and the code can run without any undefined references.
7) If image files are provided, DO NOT call any OCR libraries in Python. Rely on the model’s visual inspection to extract text from images.
8) If a database file is provided, you may call:
   - list_tables() -> list of table names
   - read_table(table_name, limit=None) -> pandas.DataFrame
   - sql_to_dataframe(sql) -> pandas.DataFrame
   Use SQL where appropriate and cast numeric columns before stats/plots.
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

tool_agent = create_tool_calling_agent(
    llm=chat_model,
    tools=[scrape_url_to_dataframe],
    prompt=agent_prompt
)

agent_runner = AgentExecutor(
    agent=tool_agent,
    tools=[scrape_url_to_dataframe],
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    handle_parsing_errors=True,
    return_intermediate_steps=False
)


# -----------------------------
# Runner: orchestrates agent -> pre-scrape inject -> execute
# -----------------------------
def run_agent_safely(llm_input: str) -> Dict:
    try:
        log.info("Invoking agent with input length=%d", len(llm_input or ""))
        agent_reply = agent_runner.invoke({"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS})
        agent_text = agent_reply.get("output") or agent_reply.get("final_output") or agent_reply.get("text") or ""
        log.info("Agent raw_out bytes=%d", len(agent_text.encode("utf-8")) if agent_text else 0)
        if not agent_text:
            return {"error": f"Agent returned no output. Full response: {agent_reply}"}

        parsed_out = clean_llm_output(agent_text)
        if "error" in parsed_out:
            return parsed_out

        if not isinstance(parsed_out, dict) or "code" not in parsed_out or "questions" not in parsed_out:
            return {"error": f"Invalid agent response format: {parsed_out}"}

        gen_code = parsed_out["code"]
        question_list: List[str] = parsed_out["questions"]

        url_list = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", gen_code)
        pickle_file = None
        if url_list:
            first_url = url_list[0]
            scrape_out = scrape_url_to_dataframe(first_url)
            if scrape_out.get("status") != "success":
                return {"error": f"Scrape tool failed: {scrape_out.get('message')}"}
            frame = pd.DataFrame(scrape_out["data"])
            tmp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            tmp_pkl.close()
            frame.to_pickle(tmp_pkl.name)
            pickle_file = tmp_pkl.name

        exec_out = write_and_run_temp_python(gen_code, injected_pickle=pickle_file, timeout=LLM_TIMEOUT_SECONDS)
        if exec_out.get("status") != "success":
            return {"error": f"Execution failed: {exec_out.get('message', exec_out)}", "raw": exec_out.get("raw")}
        result_map = exec_out.get("result", {})
        final_map = {}
        for qtext in question_list:
            final_map[qtext] = result_map.get(qtext, "Answer not found")
        return final_map

    except Exception as err:
        log.exception("run_agent_safely failed")
        return {"error": str(err)}


from fastapi import Request

# --- Accept POST on "/", "/api", and "/api/" (prevents 405 on prof portal) ---
@app.post("/")
@app.post("/api")
@app.post("/api/")
async def analyze_data(request: Request):
    try:
        formdata = await request.form()
        qfile = None
        dfile = None

        image_data_uris = []
        image_filenames = []

        for key, v in formdata.items():
            if hasattr(v, "filename") and v.filename:
                fname_lower = v.filename.lower()
                if fname_lower.endswith(".txt") and qfile is None:
                    qfile = v
                else:
                    dfile = v

        if not qfile:
            raise HTTPException(400, "Missing questions file (.txt)")

        qs_text = (await qfile.read()).decode("utf-8")
        ordered_keys, cast_lookup = parse_keys_and_types(qs_text)

        pickle_file = None
        df_head_text = ""
        has_dataset = False

        # DB wiring (path + backend) to pass into sandbox
        db_tmp_path = None
        db_backend = None

        if dfile:
            has_dataset = True
            data_filename = dfile.filename.lower()
            file_bytes = await dfile.read()
            from io import BytesIO

            if data_filename.endswith(".csv"):
                frame = pd.read_csv(BytesIO(file_bytes))
            elif data_filename.endswith((".xlsx", ".xls")):
                frame = pd.read_excel(BytesIO(file_bytes))
            elif data_filename.endswith(".parquet"):
                frame = pd.read_parquet(BytesIO(file_bytes))
            elif data_filename.endswith(".json"):
                try:
                    frame = pd.read_json(BytesIO(file_bytes))
                except ValueError:
                    frame = pd.DataFrame(json.loads(file_bytes.decode("utf-8")))
            elif data_filename.endswith(".pdf"):
                frame = pdf_bytes_to_dataframe(file_bytes)
            elif data_filename.endswith((".db", ".sqlite", ".duckdb")):
                # Prepare DB file on disk and detect backend
                preview = db_bytes_preview_dataframe(file_bytes)
                frame = preview["df"]
                db_tmp_path = preview["path"]
                db_backend = preview["meta"].get("backend") or ""
            elif data_filename.endswith((".png", ".jpg", ".jpeg")):
                try:
                    if PIL_READY:
                        img = Image.open(BytesIO(file_bytes))
                        img = img.convert("RGB")
                        frame = pd.DataFrame({"image": [img]})
                    else:
                        raise HTTPException(400, "PIL not available for image processing")
                except Exception as e:
                    raise HTTPException(400, f"Image processing failed: {str(e)}")
                mime = "image/png" if data_filename.endswith(".png") else "image/jpeg"
                image_data_uris.append(_bytes_to_data_uri(file_bytes, mime=mime))
                image_filenames.append(dfile.filename)
            else:
                raise HTTPException(400, f"Unsupported data file type: {data_filename}")

            # Only pickle if we actually have a frame (non-DB image preview frame also allowed)
            if 'frame' in locals() and isinstance(frame, pd.DataFrame):
                tmp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                tmp_pkl.close()
                frame.to_pickle(tmp_pkl.name)
                pickle_file = tmp_pkl.name

                df_head_text = (
                    f"\n\nThe uploaded dataset has {len(frame)} rows and {len(frame.columns)} columns.\n"
                    f"Columns: {', '.join(frame.columns.astype(str))}\n"
                    f"First rows:\n{frame.head(5).to_markdown(index=False)}\n"
                )

        # Build rules
        if has_dataset:
            rules_text = (
                "Rules:\n"
                "1) You have access to a pandas DataFrame called `df` (if present via preview) and its dictionary form `data`.\n"
                "2) DO NOT call scrape_url_to_dataframe() or fetch any external data when an uploaded dataset/DB is provided.\n"
                "3) Use only the uploaded content for answering questions.\n"
                "4) Produce a final JSON object with keys:\n"
                '   - "questions": [ ... original question strings ... ]\n'
                '   - "code": "..."  (Python code that fills `results` with exact question strings as keys)\n'
                "5) For plots: use plot_to_base64() helper to return base64 image data under 100kB.\n"
            )
        else:
            rules_text = (
                "Rules:\n"
                "1) If you need web data, CALL scrape_url_to_dataframe(url).\n"
                "2) Produce a final JSON object with keys:\n"
                '   - "questions": [ ... original question strings ... ]\n'
                '   - "code": "..."  (Python code that fills `results` with exact question strings as keys)\n'
                "3) For plots: use plot_to_base64() helper to return base64 image data under 100kB.\n"
            )

        if db_tmp_path and db_backend:
            rules_text += (
                "6) A database is uploaded. Use these helpers:\n"
                "   - list_tables() -> list table names\n"
                "   - read_table(table_name, limit=None) -> pandas.DataFrame\n"
                "   - sql_to_dataframe(sql) -> pandas.DataFrame\n"
                "   Prefer SQL for filtering/aggregation; cast numerics before math.\n"
            )

        if image_data_uris:
            rules_text += (
                "7) If image files are provided, DO NOT call any OCR libraries in Python. "
                "Rely on the model’s visual inspection to extract text from images.\n"
            )

        prompt_text = (
            f"{rules_text}\nQuestions:\n{qs_text}\n"
            f"{df_head_text if df_head_text else ''}"
            "Respond with the JSON object only."
        )

        if image_data_uris:
            prompt_text += (
                "\nAdditional rules for images:\n"
                "- You have access to the uploaded images directly in this message.\n"
                "- Do NOT attempt to use Python OCR libraries (not available).\n"
                "- Extract text by visually inspecting the images.\n"
                "- Return only JSON in the requested shape."
            )
            vision_result = run_vision_extraction(prompt_text, image_data_uris)
            if "error" in vision_result:
                raise HTTPException(500, detail=f"Vision extraction failed: {vision_result['error']}")
            return JSONResponse(content=vision_result)

        # Run agent (with DB helpers if present)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                run_agent_safely_unified,
                prompt_text,
                pickle_file,
                db_tmp_path,
                db_backend
            )
            try:
                result_payload = future.result(timeout=LLM_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                raise HTTPException(408, "Processing timeout")

        if "error" in result_payload:
            raise HTTPException(500, detail=result_payload["error"])

        # Post-process key mapping & type casting
        if ordered_keys and cast_lookup:
            projected = {}
            for i, qtxt in enumerate(result_payload.keys()):
                if i < len(ordered_keys):
                    k = ordered_keys[i]
                    to_type = cast_lookup.get(k, str)
                    try:
                        v = result_payload[qtxt]
                        if isinstance(v, str) and v.startswith("data:image/"):
                            v = v.split(",", 1)[1] if "," in v else v
                        projected[k] = to_type(v) if v not in (None, "") else v
                    except Exception:
                        projected[k] = result_payload[qtxt]
            result_payload = projected

        return JSONResponse(content=result_payload)

    except HTTPException as he:
        raise he
    except Exception as e:
        log.exception("analyze_data failed")
        raise HTTPException(500, detail=str(e))


def run_agent_safely_unified(llm_input: str, pickle_path: str = None,
                             db_path: str = None, db_backend: str = None) -> Dict:
    """
    Runs the LLM agent and executes code.
    - Retries up to 3 times if agent returns no output.
    - If pickle_path is provided, injects that DataFrame directly.
    - If no pickle_path, falls back to scraping when needed.
    - If db_path is provided, exposes DB helpers to the sandbox.
    """
    try:
        attempt_limit = 3
        agent_text = ""
        for attempt in range(1, attempt_limit + 1):
            log.info("Agent attempt %d/%d", attempt, attempt_limit)
            agent_reply = agent_runner.invoke({"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS})
            agent_text = agent_reply.get("output") or agent_reply.get("final_output") or agent_reply.get("text") or ""
            if agent_text:
                break
        if not agent_text:
            return {"error": f"Agent returned no output after {attempt_limit} attempts"}

        parsed_out = clean_llm_output(agent_text)
        if "error" in parsed_out:
            return parsed_out

        if "code" not in parsed_out or "questions" not in parsed_out:
            return {"error": f"Invalid agent response: {parsed_out}"}

        gen_code = parsed_out["code"]
        question_list = parsed_out["questions"]

        if pickle_path is None:
            url_list = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", gen_code)
            if url_list:
                target = url_list[0]
                scrape_out = scrape_url_to_dataframe(target)
                if scrape_out.get("status") != "success":
                    return {"error": f"Scrape tool failed: {scrape_out.get('message')}"}
                frame = pd.DataFrame(scrape_out["data"])
                tmp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                tmp_pkl.close()
                frame.to_pickle(tmp_pkl.name)
                pickle_path = tmp_pkl.name

        exec_out = write_and_run_temp_python(
            gen_code,
            injected_pickle=pickle_path,
            timeout=LLM_TIMEOUT_SECONDS,
            injected_db_path=db_path,
            injected_db_backend=db_backend
        )
        if exec_out.get("status") != "success":
            return {"error": f"Execution failed: {exec_out.get('message')}", "raw": exec_out.get("raw")}

        result_map = exec_out.get("result", {})
        return {q: result_map.get(q, "Answer not found") for q in question_list}

    except Exception as e:
        log.exception("run_agent_safely_unified failed")
        return {"error": str(e)}


from fastapi.responses import FileResponse, Response
import base64, os

# 1×1 transparent PNG fallback (if favicon.ico file not present)
_FAVICON_FALLBACK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII="
)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Serve favicon.ico if present in the working directory.
    Otherwise return a tiny transparent PNG to avoid 404s.
    """
    path = "favicon.ico"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/x-icon")
    return Response(content=_FAVICON_FALLBACK_PNG, media_type="image/png")

# Health/info — allow both /api and /api/ to avoid redirects
@app.get("/api", include_in_schema=False)
@app.get("/api/", include_in_schema=False)
async def analyze_get_info():
    """Health/info endpoint. Use POST / (or /api) for actual analysis."""
    return JSONResponse({
        "ok": True,
        "message": "Server is running. Use POST / or /api with 'questions_file' and optional 'data_file'.",
    })


# -----------------------------
# System Diagnostics
# -----------------------------
import asyncio
import httpx
import importlib.metadata
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from datetime import datetime, timedelta
import socket
import platform
import psutil
import shutil
import tempfile
import os
import time 
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse    

DIAG_NETWORK_TARGETS = {
    "Google AI": "https://generativelanguage.googleapis.com",
    "AISTUDIO": "https://aistudio.google.com/",
    "OpenAI": "https://api.openai.com",
    "AI Pipe (OpenAI)": "https://aipipe.org/openai/v1/models",
    "GitHub": "https://api.github.com",
}
DIAG_LLM_KEY_TIMEOUT = 30
DIAG_PARALLELISM = 6
RUN_LONGER_CHECKS = False

def _now_iso():
    return datetime.utcnow().isoformat() + "Z"

_executor = ThreadPoolExecutor(max_workers=DIAG_PARALLELISM)
async def run_in_thread(fn, *a, timeout=30, **kw):
    loop = asyncio.get_running_loop()
    try:
        task = loop.run_in_executor(_executor, partial(fn, *a, **kw))
        return await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError("timeout")
    except Exception as e:
        raise

def _env_check(required=None):
    required = required or []
    state = {}
    for k in required:
        state[k] = {"present": bool(os.getenv(k)), "masked": (os.getenv(k)[:4] + "..." + os.getenv(k)[-4:]) if os.getenv(k) else None}
    state["GOOGLE_MODEL"] = os.getenv("GOOGLE_MODEL")
    state["LLM_TIMEOUT_SECONDS"] = os.getenv("LLM_TIMEOUT_SECONDS")
    state["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
    return state

def _system_info():
    info = {
        "host": socket.gethostname(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
    }
    try:
        _cwd = os.getcwd()
        info["cwd_free_gb"] = round(shutil.disk_usage(_cwd).free / 1024**3, 2)
    except Exception:
        info["cwd_free_gb"] = None
    try:
        info["tmp_free_gb"] = round(shutil.disk_usage(tempfile.gettempdir()).free / 1024**3, 2)
    except Exception:
        info["tmp_free_gb"] = None
    try:
        import torch
        info["torch_installed"] = True
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch_installed"] = False
        info["cuda_available"] = False
    return info

def _temp_write_test():
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, f"diag_test_{int(time.time())}.tmp")
    with open(path, "w") as f:
        f.write("ok")
    ok = os.path.exists(path)
    os.remove(path)
    return {"tmp_dir": tmp, "write_ok": ok}

def _app_write_test():
    cwd = os.getcwd()
    path = os.path.join(cwd, f"diag_test_{int(time.time())}.tmp")
    with open(path, "w") as f:
        f.write("ok")
    ok = os.path.exists(path)
    os.remove(path)
    return {"cwd": cwd, "write_ok": ok}

def _pandas_pipeline_test():
    import pandas as _pd
    df = _pd.DataFrame({"x":[1,2,3], "y":[4,5,6]})
    df["z"] = df["x"] * df["y"]
    agg = df["z"].sum()
    return {"rows": df.shape[0], "cols": df.shape[1], "z_sum": int(agg)}

def _installed_packages_sample():
    try:
        out = []
        for dist in importlib.metadata.distributions():
            try:
                out.append(f"{dist.metadata['Name']}=={dist.version}")
            except Exception:
                try:
                    out.append(f"{dist.metadata['Name']}")
                except Exception:
                    continue
        return {"sample_packages": sorted(out)[:20]}
    except Exception as e:
        return {"error": str(e)}

def _network_probe_sync(url, timeout=30):
    try:
        r = requests.head(url, timeout=timeout)
        return {"ok": True, "status_code": r.status_code, "latency_ms": int(r.elapsed.total_seconds()*1000)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _test_openai_aipipe_models(api_key: str, base_url: str):
    try:
        url = base_url.rstrip("/") + "/models"
        r = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        return {"ok": r.ok, "status_code": r.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}

async def check_network():
    coros = []
    for name, url in DIAG_NETWORK_TARGETS.items():
        coros.append(run_in_thread(_network_probe_sync, url, timeout=30))
    results = await asyncio.gather(*[asyncio.create_task(c) for c in coros], return_exceptions=True)
    out = {}
    for (name, _), res in zip(DIAG_NETWORK_TARGETS.items(), results):
        if isinstance(res, Exception):
            out[name] = {"ok": False, "error": str(res)}
        else:
            out[name] = res
    return out

async def check_llm_keys_models():
    if not OPENAI_API_KEY:
        return {"warning": "no OPENAI_API_KEY configured"}
    return await run_in_thread(_test_openai_aipipe_models, OPENAI_API_KEY, OPENAI_BASE_URL, timeout=10)

async def check_duckdb():
    try:
        import duckdb
        def duck_check():
            conn = duckdb.connect(":memory:")
            conn.execute("SELECT 1")
            conn.close()
            return {"duckdb": True}
        return await run_in_thread(duck_check, timeout=30)
    except Exception as e:
        return {"duckdb_error": str(e)}

async def check_playwright():
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            b = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = await b.new_page()
            await page.goto("about:blank")
            ua = await page.evaluate("() => navigator.userAgent")
            await b.close()
            return {"playwright_ok": True, "ua": ua[:200]}
    except Exception as e:
        return {"playwright_error": str(e)}

from fastapi import Query

@app.get("/summary")
async def diagnose(full: bool = Query(False, description="If true, run extended checks (duckdb/playwright)")):
    started = datetime.utcnow()
    report = {
        "status": "ok",
        "server_time": _now_iso(),
        "summary": {},
        "checks": {},
        "elapsed_seconds": None
    }

    tasks = {
        "env": run_in_thread(_env_check, ["GOOGLE_API_KEY", "GOOGLE_MODEL", "LLM_TIMEOUT_SECONDS", "OPENAI_BASE_URL"], timeout=3),
        "system": run_in_thread(_system_info, timeout=30),
        "tmp_write": run_in_thread(_temp_write_test, timeout=30),
        "cwd_write": run_in_thread(_app_write_test, timeout=30),
        "pandas": run_in_thread(_pandas_pipeline_test, timeout=30),
        "packages": run_in_thread(_installed_packages_sample, timeout=50),
        "network": asyncio.create_task(check_network()),
        "llm_keys_models": asyncio.create_task(check_llm_keys_models())
    }

    if full or RUN_LONGER_CHECKS:
        tasks["duckdb"] = asyncio.create_task(check_duckdb())
        tasks["playwright"] = asyncio.create_task(check_playwright())

    results = {}
    for name, coro in tasks.items():
        try:
            res = await coro
            results[name] = {"status": "ok", "result": res}
        except TimeoutError:
            results[name] = {"status": "timeout", "error": "check timed out"}
        except Exception as e:
            results[name] = {"status": "error", "error": str(e), "trace": traceback.format_exc()}

    report["checks"] = results
    failed = [k for k, v in results.items() if v.get("status") != "ok"]
    report["status"] = "warning" if failed else "ok"
    report["summary"]["failed_checks"] = failed if failed else []
    report["elapsed_seconds"] = (datetime.utcnow() - started).total_seconds()
    return report


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
