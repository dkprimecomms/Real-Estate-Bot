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
def extract_key(query: str, prompt_type: Optional[str] = None) -> str:

    runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
    
    prompt_template_1 = Template("""
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
    - The query is about information that can be answered from older documents.
    - The query explicitly references a past amendment (e.g., “in the second amendment”).
    - The query requests summaries, historical comparisons, timelines, or all amendments.
    - The query is generic, hypothetical, instructional, or unrelated to current controlling terms.

    Do not infer TRUE merely because “lease” or “amendment” appears, unless above conditions apply.

    ---

    ### Task 2 — Extract the property name from the user query.

    Allowed Property Names: ["South Bend IN", "Bloomington", "Columbia", "FL Neptune Beach", "GA Dallas Commons"]

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

    prompt_template_2 = Template("""
     You are an intelligent assistant that identifies whether the user query refers to a specific
lease amendment number.

Your task:
- Look ONLY at the words in the user query.
- Detect whether the query explicitly mentions a particular amendment number
  (e.g., "amendment two", "2nd amendment", "amendment 3", "third amendment").
- A number is considered mentioned only if the query contains a numeric digit or
  a spelled-out ordinal/cardinal number (e.g., "2", "two", "second", "third", "3rd").
- If a specific amendment number is mentioned, return only that number as an integer.
- If no amendment number is explicitly mentioned, return FALSE.
- Do not guess or infer an amendment number from context, domain knowledge, or the
  name of the property.
- Do not add explanations, extra text, punctuation, or JSON.

Rules:
1. Accept both numeric and spelled-out amendment numbers
   (e.g., "2", "two", "second", "2nd", "third", "3rd", etc.).
2. Ignore the words "lease", "agreement", "document", "current", "latest",
   "new", "this", "that", "present", or other filler words when deciding.
   These words by themselves DO NOT indicate a number.
3. If multiple amendment numbers appear (rare), return the FIRST valid one.
4. If the query does not clearly contain a numeric digit or spelled-out number
   directly referring to an amendment, return FALSE.
5. Output must be either:
   - a single integer (e.g., 2), or
   - FALSE (uppercase).

Output format:
2
or
FALSE

User Query:
{$query}
    """).substitute(query=query)

    if prompt_type == "TRUE":
        prompt_template = prompt_template_2
    else:
        prompt_template = prompt_template_1

    MODEL_ID = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
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

<context>
$search_results$
</context>

<question>
$query$
</question>

$output_format_instructions$

    """

    # --- Extract property + "requires_latest_amendment" ---
    r_d = (extract_key(query) or "").strip()

    # --- Extract amendment number (or FALSE) ---
    a_n = (extract_key(query, "TRUE") or "").strip()
    amendment_number: Optional[int] = None
    if a_n and a_n.upper() != "FALSE":
        try:
            amendment_number = int(a_n)
        except ValueError:
            logger.warning("unexpected amendment output %r; treating as None", a_n)
            amendment_number = None

    # --- Parse JSON from first extractor safely ---
    try:
        retrival_directions = json.loads(r_d)
    except json.JSONDecodeError:
        logger.exception("Failed to parse JSON from extractor: %r", r_d)
        retrival_directions = {
            "property_name": "NONE",
            "requires_latest_amendment": "FALSE",
        }

    location = retrival_directions.get("property_name", "")
    requires_latest_amendment = retrival_directions.get("requires_latest_amendment", "")

    # --- Property/session filter handling ---
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

    logger.info("Filter: %s", prop_filter)

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

    # --- Vector filters ---
    filters = []
    if not is_none_like(location):
        filters.append({"equals": {"key": "property", "value": location}})
        logger.info("Using property filter for: %r", location)
    else:
        logger.info("No location filter applied")

    # If we need latest amendment and no specific amendment was mentioned
    if requires_latest_amendment == "TRUE" and amendment_number is None:
        if location == "South Bend IN":
            amend_filter = 1
        elif location == "Bloomington":
            amend_filter = 1
        elif location == "Columbia":
            amend_filter = 1
        elif location == "FL Neptune Beach":
            amend_filter = 5
        elif location == "GA Dallas Commons":
            amend_filter = 1
        else:
            amend_filter = 1  # safe default if new location added

        filters.append({
            "greaterThanOrEquals": {
                "key": "amendment_number",
                "value": amend_filter
            }
        })
        logger.info("Using amendment filter for: %r", amend_filter)

    # If a specific amendment number was extracted
    elif amendment_number is not None and amendment_number > 0:
        filters.append({
            "equals": {
                "key": "amendment_number",
                "value": amendment_number
            }
        })
        logger.info("Using amendment filter for: %r", amendment_number)
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
    

def log(session_id: str, query: str, result: str) -> None:
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
            table.put_item(
                Item={
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
    result_text, sid, filtered_prop = LLMcall(inp.user_query, inp.session_id, inp.filter_one)
    log(sid, inp.user_query, result_text)
    return Out(result=result_text, session_id=sid, filter_one=filtered_prop)
