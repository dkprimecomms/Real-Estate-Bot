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
  You are an intelligent assistant that performs two independent classifications on real-estate lease queries.

---

### Global Rules
- Decide each task independently. The result of one task must not influence the other.
- Output valid JSON only, exactly matching the schema below. Use uppercase TRUE/FALSE and strings only.
- If uncertain for a task, follow that task's default (Task 1 → FALSE).
- Do not include explanations, notes, or extra fields.

---

### Task 1 — Determine if the user query requires information from the *current (latest)* amendment document.

Default assumption: FALSE.  
Only mark TRUE when the question explicitly or clearly depends on the most recent controlling document.

**Important rule:**  
Questions involving **lease options** (renewal options, extension options, purchase options, termination options, or any reference to an "option") MUST be marked TRUE, because option rights are controlled by the latest amendment.

#### Classification Rules

Set `"requires_latest_amendment": "TRUE"` when:
1. The query explicitly mentions “current”, “latest”, “most recent”, or “active”.
2. The query asks about any clause whose validity or content can be modified by newer amendments, including:
   - lease expiration / end date  
   - renewal status or renewal term  
   - commencement or termination date  
   - **ANY type of option (renewal, extension, early termination, purchase, etc.)**
3. The query depends on the present effective state of the lease.
4. The query does NOT request summaries, comparisons, historic information, or analysis of multiple amendments.

Set `"requires_latest_amendment": "FALSE"` only when:
- The query is about information that can be answered from older documents (e.g., rent, landlord name, tenant name).
- The query explicitly references a past amendment (e.g., “in the second amendment”).
- The query requests summaries, historical comparisons, timelines, or all amendments.
- The query is generic, hypothetical, instructional, or unrelated to current controlling terms.

Do not infer TRUE merely because “lease” or “amendment” appears, unless above conditions apply.

---

### Task 2 — Extract the property name from the user query.

Allowed Property Names: ["South Bend IN", "Bloomington", "Columbia", "FL Neptune Beach"]

Rules:
1. Match case-insensitively.
2. Partial matches are acceptable — return the full name from the list.
3. Ignore filler words like “property”, “store”, “in”, “at”, “city”, “location”, or “of”.
4. Return exactly one best match; if none, return "NONE".
5. No explanations.

---

### Output Format

{
  "requires_latest_amendment": "TRUE or FALSE",
  "property_name": "Matched property name or NONE"
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
                "numberOfResults": 20
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
            amend_filter = 5
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
