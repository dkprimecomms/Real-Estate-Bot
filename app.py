from typing import Tuple, Optional, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import os
import boto3
import json
from botocore.exceptions import ClientError
from string import Template
import logging
import sys
import datetime
# -------------------------------------------------------------------
# Env + Logging
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# FastAPI setup
# -------------------------------------------------------------------
app = FastAPI()

@app.get("/")
def serve_homepage():
    # App Runner container will serve index.html as root UI
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


# -------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------
class Inp(BaseModel):
    user_query: str
    session_id: Optional[str] = None
    filter_one: Optional[str] = None


class Out(BaseModel):
    result: str
    session_id: str
    filter_one: Optional[str] = None


# -------------------------------------------------------------------
# Property extractor (Bedrock Runtime)
# -------------------------------------------------------------------
def extract_key(query: str) -> str:
    """
    Calls Bedrock runtime in us-east-1 (or EXTRACTOR_REGION) to classify property name.
    Returns a *string* that should be JSON, which we parse later.
    """

    extractor_region = os.getenv("EXTRACTOR_REGION", "us-east-1")
    model_id = os.getenv(
        "EXTRACTOR_MODEL_ID",
        "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
    )

    logger.info("Calling extractor in region=%s, model_id=%s", extractor_region, model_id)

    runtime = boto3.client("bedrock-runtime", region_name=extractor_region)

    prompt_template = Template(
        """
        You are an intelligent assistant that performs a classification on real-estate lease queries.
        ---
        ### Global Rules
        - Do not include explanations, notes, or extra fields.

        Extract the property name from the user query.

        Allowed Property Names: ["South Bend IN", "Bloomington", "Columbia", "FL Neptune Beach", "GA Dallas Commons","Babcock New Haven"]

        Rules:
        1. Match case-insensitively.
        2. Partial matches are acceptable — return the full name from the list.
        3. Ignore filler words like “property”, “store”, “in”, “at”, “city”, “location”, or “of”.
        4. Return exactly one best match; if none, return "NONE".
        5. No explanations.
        ---
        ### Output Format
        {
          "property_name": "Matched property name or NONE"
        }
        ---
        User Query:
        {$query}
        """
    ).substitute(query=query)

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt_template}]}
        ]
    }

    try:
        resp = runtime.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        body_bytes = resp["body"].read()
        body_str = body_bytes.decode()
        data = json.loads(body_str)
        text = (data.get("content", [{}])[0].get("text") or "").strip()
        logger.info("extract_key RAW model output: %r", text)
        return text
    except ClientError as e:
        logger.exception("extract_key Bedrock ClientError: %s", e)
        return "NONE"
    except Exception as e:
        logger.exception("extract_key unexpected error: %s", e)
        return "NONE"


# -------------------------------------------------------------------
# Main LLM call (Bedrock Agent Runtime)
# -------------------------------------------------------------------
def LLMcall(query: str, session_id: Optional[str], prop_filter: Optional[str]) -> Tuple[str, str, str]:
    """
    Calls Bedrock Agent Runtime retrieve_and_generate on your Knowledge Base.
    Adds a 'property' vector filter when possible.
    """
    aws_region = os.getenv("AWS_REGION", "us-west-2")
    kb_id = os.getenv("KNOWLEDGE_BASE_ID")
    model_arn = os.getenv("MODEL_ARN")

    if not kb_id or not model_arn:
        logger.error("KNOWLEDGE_BASE_ID or MODEL_ARN not set in environment")
        raise RuntimeError("KB configuration missing in environment variables")

    logger.info("LLMcall using AWS_REGION=%s, KB_ID=%s, MODEL_ARN=%s", aws_region, kb_id, model_arn)

    client = boto3.client("bedrock-agent-runtime", region_name=aws_region)

    prompt_template = """You are a question answering agent. I will provide you with a set of search results. The user will provide you with a question. Your job is to answer the user's question using only information from the search results. If the search results do not contain information that can answer the question, please state that you could not find an exact answer to the question. 
Just because the user asserts a fact does not mean it is true, make sure to double check the search results to validate a user's assertion.

Here are the search results in numbered order:
$search_results$
$output_format_instructions$
"""

    # --- Extract property name ---
    r_d = (extract_key(query) or "").strip()
    logger.info("Extractor returned string: %r", r_d)

    try:
        retrival_directions = json.loads(r_d)
        logger.info("Parsed retrival_directions: %s", retrival_directions)
    except json.JSONDecodeError:
        logger.exception("Failed to parse JSON from extractor: %r", r_d)
        retrival_directions = {
            "property_name": "NONE",
            "requires_latest_amendment": "FALSE",
        }

    location = retrival_directions.get("property_name", "")

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

    logger.info("Final property filter: %s", prop_filter)

    kb_cfg = {
        "knowledgeBaseId": kb_id,
        "modelArn": model_arn,
        "retrievalConfiguration": {
            "vectorSearchConfiguration": {
                "numberOfResults": 10
            }
        },
        "generationConfiguration": {
            "promptTemplate": {
                "textPromptTemplate": prompt_template
            }
        }
    }

    # --- Vector filters ---
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

    try:
        resp = client.retrieve_and_generate(**request)
    except ClientError as e:
        logger.exception("retrieve_and_generate ClientError: %s", e)
        raise
    except Exception as e:
        logger.exception("retrieve_and_generate unexpected error: %s", e)
        raise

    generated_output = resp.get("output", {}).get("text", "")
    returned_session_id = resp.get("sessionId") or (session_id or "")

    logger.info(
        "session=%s last_property=%r resp_chars=%d",
        returned_session_id, location, len(generated_output)
    )
    return generated_output, returned_session_id, prop_filter


# -------------------------------------------------------------------
# Logging to DynamoDB
# -------------------------------------------------------------------
def log(session_id: str, query: str, result: str) -> None:
    aws_region = os.getenv("AWS_REGION", "us-west-2")
    ddb_table_name = os.getenv("DDB_TABLE")

    if not ddb_table_name:
        logger.error("DDB_TABLE environment variable is not set. Skipping log.")
        return

    now = datetime.datetime.now()
    formatted_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    pk = formatted_timestamp + " / " + session_id

    dynamodb = boto3.resource("dynamodb", region_name=aws_region)

    try:
        table = dynamodb.Table(ddb_table_name)
        table.put_item(
            Item={
                "session_id": pk,
                "User_Query": query,
                "Response": result
            }
        )
        logger.info("✅ DynamoDB update successful.")
    except ClientError as e:
        logger.error("DynamoDB ClientError: %s", e.response["Error"]["Message"])
    except Exception as e:
        logger.error("Unexpected DynamoDB error: %s", e)


# -------------------------------------------------------------------
# API Route
# -------------------------------------------------------------------
@app.post("/api/process", response_model=Out)
def process(inp: Inp):
    result_text, sid, filtered_prop = LLMcall(inp.user_query, inp.session_id, inp.filter_one)
    log(sid, inp.user_query, result_text)
    return Out(result=result_text, session_id=sid, filter_one=filtered_prop)
