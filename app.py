# app.py
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
import datetime

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("backend")

SESSION_MEM: Dict[str, Dict[str, str]] = {}


def is_none_like(val: Optional[str]) -> bool:
    v = (val or "").strip().strip('"').strip("'").lower()
    return v in {"", "none", "null"}


app = FastAPI()

@app.get("/")
def serve_homepage():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(current_dir, "index.html")
    return FileResponse(index_path, media_type="text/html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
    # load_dotenv()  <-- REMOVED
    runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

    prompt_template = Template("""
 You are an intelligent assistant that performs three independent classifications on real-estate lease queries.

---

### Global Rules
- Decide each task independently. The result of one task must not influence the others.
- Output valid JSON only, exactly matching the schema below. Use uppercase TRUE/FALSE and strings only.
- If uncertain for a task, follow that task's default (Task 1 → FALSE).
- Do not include explanations, notes, or extra fields.

---

### Task 1 — Determine if the user query requires information from the *current (latest)* amendment document.

Default assumption: FALSE.  
Only mark TRUE when the question explicitly or clearly depends on the most recent controlling document.  
When in doubt, choose FALSE.

#### Classification Rules
1. Set "requires_latest_amendment": "TRUE" only if:
   - The query explicitly mentions “current”, “latest”, “most recent”, or “active”.
   - The query requests information whose meaning inherently depends on the *current effective state* of the lease, such as:
     - lease expiration / end date  
     - renewal status or renewal term  
     - commencement or termination date  
   - There is no mention of summaries, comparisons, historical details, or multiple amendments.
2. Set "requires_latest_amendment": "FALSE" for all other cases, including:
   - Queries about rent, landlord name, tenant name, or any data that could appear in older documents.
   - Queries that reference specific amendments, effective dates, or a historical period.
   - Queries that request summaries, comparisons, timelines, or lists of amendments.
   - Hypothetical, generic, or descriptive questions not tied to current controlling terms.
   - Meta or instructional queries.
   - Ambiguous or incomplete queries that do not clearly require the latest document.
3. Do not infer TRUE merely because the question mentions “lease” or “amendment”.

---

### Task 2 — Extract the property name from the user query.

Allowed Property Names: ["South Bend IN", "Bloomington", "Columbia", "FL Neptune Beach"]

Rules:
1. Match case-insensitively.
2. Partial matches are acceptable — return the full name from the list.
3. Ignore filler words like “property”, “store”, “in”, “at”, “city”, or “of”.
4. Return exactly one best match; if none, return "NONE".
5. No explanations.

---
### Output Format
Return the two results in this JSON structure:

{
  "requires_latest_amendment": "TRUE or FALSE",
  "property_name": "Matched property name or NONE",
}

---

User Query:
{$query}
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


def LLMcall(query: str, session_id: Optional[str], prop_filter: Optional[str]) -> Tuple[str, str, str]:
    AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
    client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

    prompt_template = """You are a precise real-estate assistant.
Use only the provided search results to answer the user's question.
If the query is a Greeting, greet back and ask how you may assist today.
If the results are insufficient, say so clearly.
If the context provided is not relevant to the query, return only "False".
<context>
$search_results$
</context>
<question>
$query$
</question>
$output_format_instructions$

"""

    r_d = (extract_key(query) or "").strip()
    retrival_directions = json.loads(r_d)
    location = retrival_directions.get("property_name", "")
    requires_latest_amendment = retrival_directions.get("requires_latest_amendment", "")

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

    if not is_none_like(location):
        prop_filter = location

    logger.info("Filter: %s",prop_filter)
    kb_cfg = {
        "knowledgeBaseId": os.getenv("KNOWLEDGE_BASE_ID"),
        "modelArn": os.getenv("MODEL_ARN"),               
        "retrievalConfiguration": {
            "vectorSearchConfiguration": {
                "numberOfResults": 28
            }
        },
        "generationConfiguration": {
            "promptTemplate": {
                "textPromptTemplate": prompt_template
            }
        }
    }

    filters = []
    if not is_none_like(location):
        filters.append({"equals": {"key": "property", "value": location}})
        logger.info("Using property filter for: %r", location)
    else:
        logger.info("No location filter applied")

    if requires_latest_amendment == "TRUE":
        if location == "South Bend IN":
            amend_filter= 1
        elif location == "Bloomington":
            amend_filter = 1
        elif location == "Columbia":
            amend_filter = 1
        elif location == "FL Neptune Beach":
            amend_filter = 4
        filters.append({"greaterThanOrEquals": {"key": "amendment_number", "value": amend_filter}})
        logger.info("Using amendment filter for: %r", amend_filter)
    else:
        logger.info("No amendment filter applied")

    if filters:
        vector_filter = filters[0] if len(filters) == 1 else {"andAll": filters}
        kb_cfg["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"] = vector_filter

    request = {
        "input": {"text": query},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": kb_cfg,
        }
    }
    if session_id:
        request["sessionId"] = session_id

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
    

def log(session_id: str, query:str, result:str ) -> Tuple[str, str, str]:
    AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
    DDB_TABLE = os.getenv("DDB_TABLE")
    now = datetime.datetime.now()
    formatted_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    final = formatted_timestamp + " / " + session_id
    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    if not DDB_TABLE:
        print("ERROR: DDB_TABLE environment variable is not set.")
    else:
        try:
            table = dynamodb.Table(DDB_TABLE)
            response = table.put_item(
                Item = {
                    "session_id": final,
                    "User_Query": query,
                    "Response": result
                }
            )
            print("✅ DynamoDB update successful.")

        except ClientError as e:
            print(f"DynamoDB Error: {e.response['Error']['Message']}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
@app.post("/api/process", response_model=Out)
def process(inp: Inp):
    result_text, sid, filtered_prop  = LLMcall(inp.user_query, inp.session_id, inp.filter_one)
    log(sid,inp.user_query, result_text)
    return Out(result=result_text, session_id=sid, filter_one= filtered_prop )
