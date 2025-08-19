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
import seaborn as sns
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

# LangChain / LLM imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI

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
# Tools
# -----------------------------

@tool
def scrape_url_to_dataframe(target_url: str) -> Dict[str, Any]:
    """
    Fetch a URL and return data as a DataFrame (supports HTML tables, CSV, Excel, Parquet, JSON, and plain text).
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

        # --- CSV ---
        if "text/csv" in content_type or target_url.lower().endswith(".csv"):
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
            # Try HTML tables first
            try:
                html_tables = pd.read_html(StringIO(html_text), flavor="bs4")
                if html_tables:
                    frame = html_tables[0]
            except ValueError:
                pass

            # If no table found, fallback to plain text
            if frame is None:
                soup_obj = BeautifulSoup(html_text, "html.parser")
                page_text = soup_obj.get_text(separator="\n", strip=True)
                frame = pd.DataFrame({"text": [page_text]})

        # --- Unknown type fallback ---
        else:
            frame = pd.DataFrame({"text": [r.text]})

        # --- Normalize columns ---
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
        # remove triple-fence markers if present
        content = re.sub(r"^```(?:json)?\s*", "", llm_text.strip())
        content = re.sub(r"\s*```$", "", content)
        # find outermost JSON object by scanning for balanced braces
        lbrace = content.find("{")
        rbrace = content.rfind("}")
        if lbrace == -1 or rbrace == -1 or rbrace <= lbrace:
            return {"error": "No JSON object found in LLM output", "raw": content}
        json_candidate = content[lbrace:rbrace+1]
        try:
            return json.loads(json_candidate)
        except Exception as e:
            # fallback: try last balanced pair scanning backwards
            for i in range(rbrace, lbrace, -1):
                cand = content[lbrace:i+1]
                try:
                    return json.loads(cand)
                except Exception:
                    continue
            return {"error": f"JSON parsing failed: {str(e)}", "raw": json_candidate}
    except Exception as e:
        return {"error": str(e)}

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
        
        # Ensure all columns are unique and string
        df.columns = [str(col) for col in df.columns]

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns)
        }
    else:
        # Fallback to plain text
        text_data = soup.get_text(separator="\n", strip=True)

        # Try to detect possible "keys" from text like Runtime, Genre, etc.
        detected_cols = set(re.findall(r"\b[A-Z][a-zA-Z ]{2,15}\b", text_data))
        df = pd.DataFrame([{}])  # start empty
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


def write_and_run_temp_python(code: str, injected_pickle: str = None, timeout: int = 60) -> Dict[str, Any]:
    """
    Write a temp python file which:
      - provides a safe environment (imports)
      - loads df/from pickle if provided into df and data variables
      - defines a robust plot_to_base64() helper that ensures < 100kB (attempts resizing/conversion)
      - executes the user code (which should populate `results` dict)
      - prints json.dumps({"status":"success","result":results})
    Returns dict with parsed JSON or error details.
    """
    # create file content
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
    # inject df if a pickle path provided
    if injected_pickle:
        bootstrap_lines.append(f"df = pd.read_pickle(r'''{injected_pickle}''')\n")
        bootstrap_lines.append("data = df.to_dict(orient='records')\n")
    else:
        # ensure data exists so user code that references data won't break
        bootstrap_lines.append("data = globals().get('data', {})\n")

    # plot_to_base64 helper that tries to reduce size under 100_000 bytes
    plot_helper = r'''
def plot_to_base64(max_bytes=100000):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_bytes = buf.getvalue()
    if len(img_bytes) <= max_bytes:
        return base64.b64encode(img_bytes).decode('ascii')
    # try decreasing dpi/figure size iteratively
    for dpi in [80, 60, 50, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return base64.b64encode(b).decode('ascii')
    # if Pillow available, try convert to WEBP which is typically smaller
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
        # try lower quality
        out_buf = BytesIO()
        im.save(out_buf, format='WEBP', quality=60, method=6)
        out_buf.seek(0)
        ob = out_buf.getvalue()
        if len(ob) <= max_bytes:
            return base64.b64encode(ob).decode('ascii')
    except Exception:
        pass
    # as last resort return downsized PNG even if > max_bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')
'''

    # Build the code to write
    script_buf = []
    script_buf.extend(bootstrap_lines)
    script_buf.append(plot_helper)
    script_buf.append(SCRAPE_FUNC)
    script_buf.append("\nresults = {}\n")
    script_buf.append(code)
    # ensure results printed as json
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
            # collect stderr and stdout for debugging
            return {"status": "error", "message": proc.stderr.strip() or proc.stdout.strip()}
        # parse stdout as json
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
# Use the ChatOpenAI `chat_model` defined above.

# Tools list for agent (LangChain tool decorator returns metadata for the LLM)
tools = [scrape_url_to_dataframe]  # we only expose scraping as a tool; agent will still produce code

# Prompt: instruct agent to call the tool and output JSON only
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
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

tool_agent = create_tool_calling_agent(
    llm=chat_model,
    tools=[scrape_url_to_dataframe],  # let the agent call tools if it wants; we will also pre-process scrapes
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
    """
    1. Run the agent_executor.invoke to get LLM output
    2. Extract JSON, get 'code' and 'questions'
    3. Detect scrape_url_to_dataframe("...") calls in code, run them here, pickle df and inject before exec
    4. Execute the code in a temp file and return results mapping questions -> answers
    """
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

        # Detect scrape calls; find all URLs used in scrape_url_to_dataframe("URL")
        url_list = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", gen_code)
        pickle_file = None
        if url_list:
            # For now support only the first URL (agent may code multiple scrapes; you can extend this)
            first_url = url_list[0]
            scrape_out = scrape_url_to_dataframe(first_url)
            if scrape_out.get("status") != "success":
                return {"error": f"Scrape tool failed: {scrape_out.get('message')}"}
            # create df and pickle it
            frame = pd.DataFrame(scrape_out["data"])
            tmp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            tmp_pkl.close()
            frame.to_pickle(tmp_pkl.name)
            pickle_file = tmp_pkl.name

        # Execute code in temp python script
        exec_out = write_and_run_temp_python(gen_code, injected_pickle=pickle_file, timeout=LLM_TIMEOUT_SECONDS)
        if exec_out.get("status") != "success":
            return {"error": f"Execution failed: {exec_out.get('message', exec_out)}", "raw": exec_out.get("raw")}

        # exec_out['result'] should be results dict
        result_map = exec_out.get("result", {})
        # Map to original questions (they asked to use exact question strings)
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

        for key, v in formdata.items():
            if hasattr(v, "filename") and v.filename:  # it's a file
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
            elif data_filename.endswith(".png") or data_filename.endswith(".jpg") or data_filename.endswith(".jpeg"):
                try:
                    if PIL_READY:
                        img = Image.open(BytesIO(file_bytes))
                        img = img.convert("RGB")  # ensure RGB format
                        frame = pd.DataFrame({"image": [img]})
                    else:
                        raise HTTPException(400, "PIL not available for image processing")
                except Exception as e:
                    raise HTTPException(400, f"Image processing failed: {str(e)}")  
            else:
                raise HTTPException(400, f"Unsupported data file type: {data_filename}")

            # Pickle for injection
            tmp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            tmp_pkl.close()
            frame.to_pickle(tmp_pkl.name)
            pickle_file = tmp_pkl.name

            df_head_text = (
                f"\n\nThe uploaded dataset has {len(frame)} rows and {len(frame.columns)} columns.\n"
                f"Columns: {', '.join(frame.columns.astype(str))}\n"
                f"First rows:\n{frame.head(5).to_markdown(index=False)}\n"
            )

        # Build rules based on data presence
        if has_dataset:
            rules_text = (
                "Rules:\n"
                "1) You have access to a pandas DataFrame called `df` and its dictionary form `data`.\n"
                "2) DO NOT call scrape_url_to_dataframe() or fetch any external data.\n"
                "3) Use only the uploaded dataset for answering questions.\n"
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

        prompt_text = (
            f"{rules_text}\nQuestions:\n{qs_text}\n"
            f"{df_head_text if df_head_text else ''}"
            "Respond with the JSON object only."
        )

        # Run agent
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(run_agent_safely_unified, prompt_text, pickle_file)
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
                            # Remove data URI prefix
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


def run_agent_safely_unified(llm_input: str, pickle_path: str = None) -> Dict:
    """
    Runs the LLM agent and executes code.
    - Retries up to 3 times if agent returns no output.
    - If pickle_path is provided, injects that DataFrame directly.
    - If no pickle_path, falls back to scraping when needed.
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

        exec_out = write_and_run_temp_python(gen_code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
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
# ---- Add these imports near other imports at top of app.py ----
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

# ---- Configuration for diagnostics (tweak as needed) ----
DIAG_NETWORK_TARGETS = {
    "Google AI": "https://generativelanguage.googleapis.com",
    "AISTUDIO": "https://aistudio.google.com/",
    "OpenAI": "https://api.openai.com",
    "AI Pipe (OpenAI)": "https://aipipe.org/openai/v1/models",
    "GitHub": "https://api.github.com",
}
DIAG_LLM_KEY_TIMEOUT = 30  # seconds per key/model simple ping test (sync tests run in threadpool)
DIAG_PARALLELISM = 6       # how many thread workers for sync checks
RUN_LONGER_CHECKS = False  # Playwright/duckdb tests run only if true (they can be slow)

# helper: iso timestamp
def _now_iso():
    return datetime.utcnow().isoformat() + "Z"

# helper: run sync func in threadpool and return result / exception info
_executor = ThreadPoolExecutor(max_workers=DIAG_PARALLELISM)
async def run_in_thread(fn, *a, timeout=30, **kw):
    loop = asyncio.get_running_loop()
    try:
        task = loop.run_in_executor(_executor, partial(fn, *a, **kw))
        return await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError("timeout")
    except Exception as e:
        # re-raise for caller to capture stacktrace easily
        raise

# ---- Diagnostic check functions (safely return dicts) ----
def _env_check(required=None):
    required = required or []
    state = {}
    for k in required:
        state[k] = {"present": bool(os.getenv(k)), "masked": (os.getenv(k)[:4] + "..." + os.getenv(k)[-4:]) if os.getenv(k) else None}
    # Also include simple helpful values
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
    # disk free for app dir and tmp
    try:
        _cwd = os.getcwd()
        info["cwd_free_gb"] = round(shutil.disk_usage(_cwd).free / 1024**3, 2)
    except Exception:
        info["cwd_free_gb"] = None
    try:
        info["tmp_free_gb"] = round(shutil.disk_usage(tempfile.gettempdir()).free / 1024**3, 2)
    except Exception:
        info["tmp_free_gb"] = None
    # GPU quick probe (if torch installed)
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
    # try writing into current working directory
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
    # return top 20 installed package names + versions
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
    # synchronous network probe for threadpool use
    try:
        r = requests.head(url, timeout=timeout)
        return {"ok": True, "status_code": r.status_code, "latency_ms": int(r.elapsed.total_seconds()*1000)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---- LLM key/model light test (AI Pipe/OpenAI) ----
def _test_openai_aipipe_models(api_key: str, base_url: str):
    try:
        url = base_url.rstrip("/") + "/models"
        r = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        return {"ok": r.ok, "status_code": r.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---- Async wrappers that call the sync checks in threadpool ----
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
    """Check AI Pipe/OpenAI list-models endpoint with provided token."""
    if not OPENAI_API_KEY:
        return {"warning": "no OPENAI_API_KEY configured"}
    return await run_in_thread(_test_openai_aipipe_models, OPENAI_API_KEY, OPENAI_BASE_URL, timeout=10)

# ---- Optional slow heavy checks (DuckDB, Playwright) ----
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

# ---- Final /diagnose route (concurrent) ----
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

    # prepare tasks
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

    # run all concurrently, collect results
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

    # quick summary flags
    failed = [k for k, v in results.items() if v.get("status") != "ok"]
    if failed:
        report["status"] = "warning"
        report["summary"]["failed_checks"] = failed
    else:
        report["status"] = "ok"
        report["summary"]["failed_checks"] = []

    report["elapsed_seconds"] = (datetime.utcnow() - started).total_seconds()
    return report


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
