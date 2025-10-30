# backend.py
# pip install fastapi uvicorn "pydantic<3" python-dotenv boto3
from typing import Tuple, Optional, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import boto3, json
from botocore.exceptions import ClientError
from string import Template
from dotenv import load_dotenv
import logging
import sys
from fastapi.responses import FileResponse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("backend")

# ---- In-memory session memory (per-process; swap to Redis for prod) ----
SESSION_MEM: Dict[str, Dict[str, str]] = {}  # { session_id: {"last_property": "South Bend IN"} }

# def get_last_property(session_id: Optional[str]) -> Optional[str]:
#     if not session_id:
#         return None
#     return SESSION_MEM.get(session_id, {}).get("last_property")

# def set_last_property(session_id: Optional[str], prop: Optional[str]) -> None:
#     if not session_id or not prop:
#         return
#     rec = SESSION_MEM.setdefault(session_id, {})
#     rec["last_property"] = prop

def is_none_like(val: Optional[str]) -> bool:
    v = (val or "").strip().strip('"').strip("'").lower()
    return v in {"", "none", "null"}


# ---- API setup ----
app = FastAPI()

@app.get("/")
def serve_homepage():
    # Get path to index.html in the same directory as this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(current_dir, "index.html")
    return FileResponse(index_path, media_type="text/html")

# If you truly need credentials (cookies/Authorization), list explicit origins instead of "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # change to ["http://localhost:3000"] in dev if you need credentials
    allow_credentials=False,      # browsers block "*" + credentials=True; set False for wildcard
    allow_methods=["*"],
    allow_headers=["*"],
)

class Inp(BaseModel):
    user_query: str
    session_id: Optional[str] = None
    filter_one: Optional[str] = None

class Out(BaseModel):
    result: str
    session_id: str
    filter_one: Optional[str] = None

# ---- Property extractor (Bedrock Runtime) ----
def extract_key(query: str) -> str:
    load_dotenv()
    runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

    prompt_template = Template("""
You are an intelligent assistant that extracts one property name from user queries related to real estate.

Task
Identify the property name mentioned in the user’s query and match it against the provided Allowed Property Names list.
Return the matched property name from the list. If nothing in the query matches any name from the list, return NONE.

Allowed Property Names: ["South Bend IN", "Bloomington", "Columbia"]

Rules
Perform case-insensitive matching.
A partial match is acceptable — if the user’s query contains a word or phrase similar to one of the allowed property names, return that full property name from the list.
Ignore filler words like “property”, “store”, “in”, “at”, “city”, or “of”.
Return only one property name that best fits the query.
If no property name from the list appears (even partially), return NONE.
Do not include explanations or any extra text — output only the property name or NONE.
If the user input does not mention property name then return "NONE"
Examples
User Query: What is the lease expiration for IN South Bend?
Output: South Bend IN
User Query: What is the store name in Bloomington?
Output: Bloomington
User Query: Who is columba?
Output: Columbia
User Query: Who is the landlord?
Output: NONE

Extract the property name from the User Query:
$query
""").substitute(query=query)

    MODEL_ID = "us.anthropic.claude-3-haiku-20240307-v1:0"
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt_template}]}
        ]
    }

    try:
        resp = runtime.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        data = json.loads(resp["body"].read().decode())
        text = (data.get("content", [{}])[0].get("text") or "").strip()
    except ClientError as e:
        logger.exception("extract_key Bedrock error: %s", e)
        return "NONE"
    except Exception:
        logger.exception("extract_key unexpected error")
        return "NONE"

    logger.info("extract_key -> %r", text)
    return text

# ---- Main KB call (Bedrock Agent Runtime) ----
def LLMcall(query: str, session_id: Optional[str], prop_filter: Optional[str]) -> Tuple[str, str, str]:
    load_dotenv()
    AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
    client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

    # Custom KB prompt template
    prompt_template = """You are a precise real-estate assistant.
Use only the provided search results to answer the user's question.
If the query is a Greeting, greet back and ask how you may assist today.

If the results are insufficient, say so clearly.

<context>
$search_results$
</context>

<question>
$query$
</question>

$output_format_instructions$
"""

    # 1) Extract property from this turn
    location = (extract_key(query) or "").strip()

    # 2) Fallback to session memory if none
    if is_none_like(location):
        if prop_filter:
            fallback = prop_filter
        else:
            fallback = None
        if fallback:
            logger.info("No property in this query; using session fallback: %r", fallback)
            location = fallback
        else:
            logger.info("No property in this query; no session fallback available")
    else:
        prop_filter = location

    # 3) Build KB config (put generationConfiguration INSIDE knowledgeBaseConfiguration)
    kb_cfg = {
        "knowledgeBaseId": os.getenv("KNOWLEDGE_BASE_ID"),
        "modelArn": os.getenv("MODEL_ARN"),
        "retrievalConfiguration": {
            "vectorSearchConfiguration": {
                "numberOfResults": 20
            }
        },
        "generationConfiguration": {
            "promptTemplate": {
                "textPromptTemplate": prompt_template
            }
        }
    }

    # Apply property filter only if we have a property
    if not is_none_like(location):
        kb_cfg["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"] = {
            "equals": {"key": "property", "value": location}
        }
        logger.info("Using property filter for: %r", location)
    else:
        logger.info("No location filter applied")

    request = {
        "input": {"text": query},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": kb_cfg,
        }
    }
    if session_id:
        request["sessionId"] = session_id

    # 4) Call Bedrock
    try:
        resp = client.retrieve_and_generate(**request)
    except ClientError as e:
        logger.exception("retrieve_and_generate ClientError: %s", e)
        raise
    except Exception:
        logger.exception("retrieve_and_generate unexpected error")
        raise

    generated_output = resp.get("output", {}).get("text", "")
    returned_session_id = resp.get("sessionId") or (session_id or "")


    logger.info(
        "session=%s last_property=%r resp_chars=%d",
        returned_session_id, location, len(generated_output)
    )
    return generated_output, returned_session_id, prop_filter

@app.post("/api/process", response_model=Out)
def process(inp: Inp):
    result_text, sid, filtered_prop  = LLMcall(inp.user_query, inp.session_id, inp.filter_one)
    return Out(result=result_text, session_id=sid, filter_one= filtered_prop )
