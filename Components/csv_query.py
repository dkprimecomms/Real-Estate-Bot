"""
still there is a work in session mamangement. 
Now it is capable of storing session in a single run.
ex: User asks a 2 question in same run then it is capable else no.

To do's
1. Store session data in a persistent storage (DynamoDB/S3)
2. Check Why the bedrock session manger is used.
"""

import boto3
import json
import re
import time
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

from dotenv import load_dotenv

load_dotenv()

ATHENA_REGION = "us-west-2"
BEDROCK_REGION = "us-west-2"

ATHENA_DATABASE = "store_master_list"
ATHENA_TABLE = "storemaster_store_master_list"
ATHENA_OUTPUT_LOCATION = "s3://storemasterlist/"

BEDROCK_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

FORBIDDEN_SQL = ["insert ", "update ", "delete ", "drop ", "alter ", "create "]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

US_STATE_ABBREV = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
    "district of columbia": "DC",
}

SCHEMA_DOC = f"""
You are working with an Amazon Athena (Presto) database.
Database: {ATHENA_DATABASE}
Table: {ATHENA_TABLE}

========================
DATA SEMANTICS / RULES
========================
1) store_status meanings:
The store_status column has two relevant values:
- 'OPEN':
  The store is currently operating and open to customers.

- 'TEMPORARY CLOSED':
  The store is currently NOT operating, but is expected to reopen.
  This represents an INACTIVE store at the moment.

2) State column format:
- The state column uses US state abbreviations (2-letter), NOT full names.
- Examples: TX, AL, AK, FL, GA, NC, VA, SC, CA, NV
- If user provides full state name (e.g., Texas), convert to abbreviation (TX) in SQL filters.

========================
SQL STYLE
========================
- Use Athena Presto SQL.
- Prefer aggregation for overview queries (COUNT/GROUP BY).
- Do not use SELECT *.

========================
SCHEMA
========================
Columns:
- region (string)
- avp (string)
- market (string)
- md (string)
- dm (string)
- rsm (string)
- rsm_contact_num (string)
- store (string)
- store_id (string)
- dealer_code (string)
- opusid (string)
- store_status (string)
- phone (string)
- address (string)
- city (string)
- state (string)
- zip (string)
- location_time_zone (string)
- store_type (string)
- email (string)
- legacystatus (string)
- cinglepoint_id (string)
- conexion_stores (string)
- store_classification (string)
- aroe (string)
- att_region (string)
- att_market (string)
- att_vpgm (string)
- att_avp_dos (string)
- att_arsm (string)
- sunday_open (string)
- sunday_close (string)
- monday_open (string)
- monday_close (string)
- tuesday_open (string)
- tuesday_close (string)
- wednesday_open (string)
- wednesday_close (string)
- thursday_open (string)
- thursday_close (string)
- friday_open (string)
- friday_close (string)
- saturday_open (string)
- saturday_close (string)
- storenotes (string)
- opendate (string)
"""

def user_requests_pii(question: str) -> bool:
    q = question.lower()
    return any(
        k in q
        for k in ["phone", "email", "address", "contact", "contact number", "rsm_contact_num"]
    )


def call_bedrock(system_prompt: str, user_prompt: str, max_tokens: int = 600, temperature: float = 0.1) -> str:
    """
    Calls Bedrock Anthropic model using messages API (bedrock-runtime).
    """
    client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": system_prompt + "\n\n" + user_prompt}],
            }
        ],
    }

    response = client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload),
    )
    body = json.loads(response["body"].read().decode("utf-8"))
    return body["content"][0]["text"].strip()


def extract_first_select_statement(text: str) -> str:
    """
    Extracts the first SELECT statement from model output.
    Removes common markdown fences and returns only the first statement (up to semicolon).
    """
    t = text.strip()
    t = re.sub(r"^```(?:sql)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t).strip()

    m = re.search(r"\bselect\b", t, flags=re.IGNORECASE)
    if not m:
        return t.strip()

    t = t[m.start() :].strip()
    t = t.split(";", 1)[0].strip()
    return t


def normalize_sql(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql


class BedrockSessionManager:
    """
    Stores + retrieves conversation history via in-memory storage.
    NOTE: AWS Bedrock Session APIs are not finalized. Using local dict for now.
    """
    def __init__(self, region_name: str):
        self.session_store: Dict[str, List[str]] = {}

    def create_session(self, metadata: Optional[Dict[str, str]] = None) -> str:
        session_id = str(uuid.uuid4())
        self.session_store[session_id] = []
        logger.info("Created session: %s", session_id)
        return session_id

    def create_invocation(self, session_id: str, description: str = "") -> str:
        invocation_id = str(uuid.uuid4())
        logger.info("Created invocation %s for session %s", invocation_id, session_id)
        return invocation_id

    def put_text_step(self, session_id: str, invocation_id: str, text: str) -> str:
        if session_id not in self.session_store:
            self.session_store[session_id] = []
        self.session_store[session_id].append(text[:10000])
        step_id = str(uuid.uuid4())
        logger.info("Stored step %s in session %s", step_id, session_id)
        return step_id

    def get_recent_text_history(self, session_id: str, max_steps: int = 10) -> str:
        """
        Returns a compact string of the most recent text steps,
        suitable to inject into prompts as "Conversation context".
        """
        if session_id not in self.session_store:
            return ""
        
        texts = self.session_store[session_id][-max_steps:]
        joined = "\n".join(texts).strip()
        
        if len(joined) > 6000:
            joined = joined[-6000:]
        
        return joined


def enforce_lower_on_single_quoted_comparisons(sql: str) -> str:
    """
    Rewrites comparisons with single-quoted literals:
      col = 'Value'        -> LOWER(col) = 'value'
      col LIKE 'Value'     -> LOWER(col) LIKE 'value'
      col IN ('A','B')     -> LOWER(col) IN ('a','b')
    Skips if column is already LOWER(col).
    """

    def _lower_literals_in_in_list(inner: str) -> str:
        parts: List[str] = []
        for p in inner.split(","):
            token = p.strip()
            m = re.match(r"^'([^']*)'$", token)
            if m:
                parts.append(f"'{(m.group(1) or '').lower()}'")
            else:
                parts.append(token)
        return ", ".join(parts)

    sql = re.sub(
        r"(?i)(?<!lower\()\b([a-z_][a-z0-9_]*)\b\s*=\s*'([^']*)'",
        lambda m: f"LOWER({m.group(1)}) = '{(m.group(2) or '').lower()}'",
        sql,
    )

    sql = re.sub(
        r"(?i)(?<!lower\()\b([a-z_][a-z0-9_]*)\b\s+like\s+'([^']*)'",
        lambda m: f"LOWER({m.group(1)}) LIKE '{(m.group(2) or '').lower()}'",
        sql,
    )

    sql = re.sub(
        r"(?i)(?<!lower\()\b([a-z_][a-z0-9_]*)\b\s+in\s*\(([^)]*)\)",
        lambda m: f"LOWER({m.group(1)}) IN ({_lower_literals_in_in_list(m.group(2))})",
        sql,
    )

    return sql


def apply_domain_fixes(sql: str) -> str:
    s = sql

    s = re.sub(
        r"(?i)\bstore_status\s*=\s*'active'\b",
        "store_status IN ('OPEN', 'TEMPORARY CLOSED')",
        s,
    )

    def _state_repl(match: re.Match) -> str:
        raw = match.group(1)
        key = raw.strip().lower()
        ab = US_STATE_ABBREV.get(key)
        return f"state = '{ab}'" if ab else match.group(0)

    s = re.sub(r"(?i)\bstate\s*=\s*'([^']+)'\b", _state_repl, s)

    s = enforce_lower_on_single_quoted_comparisons(s)

    return s


def validate_and_ensure_limit(sql_text: str, limit: int) -> str:
    sql = extract_first_select_statement(sql_text)
    sql = normalize_sql(sql)

    sql = apply_domain_fixes(sql)
    sql = normalize_sql(sql)

    sql_l = sql.lower()

    if any(f in sql_l for f in FORBIDDEN_SQL):
        raise ValueError(f"DDL/DML not allowed: {sql}")

    if not sql_l.startswith("select"):
        raise ValueError(f"Expected SELECT query, got: {sql}")

    if re.search(r"select\s+\*", sql_l):
        raise ValueError("SELECT * is not allowed. Must explicitly list columns.")

    if re.search(r"\blimit\b", sql_l) is None:
        sql = f"{sql} LIMIT {limit}"

    sql = sql.rstrip(" ;\n\t") + ";"
    return sql


def run_athena(sql: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    logger.info("Executing Athena SQL (final):\n%s", sql)

    athena = boto3.client("athena", region_name=ATHENA_REGION)
    resp = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DATABASE},
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_LOCATION},
    )
    qid = resp["QueryExecutionId"]

    while True:
        exec_resp = athena.get_query_execution(QueryExecutionId=qid)
        state = exec_resp["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            if state != "SUCCEEDED":
                reason = exec_resp["QueryExecution"]["Status"].get("StateChangeReason", "Unknown")
                raise RuntimeError(f"Athena query failed: {state} - {reason}")
            break
        time.sleep(1)

    headers: List[str] = []
    rows: List[Dict[str, Any]] = []
    token: Optional[str] = None

    while True:
        kwargs: Dict[str, Any] = {"QueryExecutionId": qid}
        if token:
            kwargs["NextToken"] = token

        result = athena.get_query_results(**kwargs)
        data = result["ResultSet"]["Rows"]

        if not headers and data:
            headers = [c.get("VarCharValue", "") for c in data[0]["Data"]]

        for r in data[1:]:
            vals = [c.get("VarCharValue", None) for c in r["Data"]]
            rows.append(dict(zip(headers, vals)))

        token = result.get("NextToken")
        if not token:
            break

    logger.info("Athena rows returned: %d", len(rows))
    return headers, rows

def generate_overview_sql(question: str, conversation_context: str = "") -> str:
    pii_rule = (
        "You MAY include phone/email/address/rsm_contact_num only if the user explicitly asked for them."
        if user_requests_pii(question)
        else "Do NOT select phone, email, address, or rsm_contact_num."
    )

    system = (
        "You write ONE SQL query for Amazon Athena (Presto).\n\n"
        "Hard rules:\n"
        "- Output ONLY SQL (no markdown, no explanation).\n"
        "- Use ONLY the provided schema/table/columns.\n"
        "- No SELECT *.\n"
        "- No DDL/DML.\n"
        "- Prefer aggregation (COUNT/GROUP BY) so output stays small.\n"
        "- Always include LIMIT 200 or smaller.\n"
    )

    context_block = ""
    if conversation_context.strip():
        context_block = f"\nConversation context (most recent snippets):\n{conversation_context}\n"

    user = f"""
{SCHEMA_DOC}
{context_block}
User question:
{question}

{pii_rule}

Write an OVERVIEW query that summarizes the answer using aggregation.
Return ONLY SQL.
"""

    raw = call_bedrock(system, user, max_tokens=450, temperature=0.1)
    logger.info("LLM OVERVIEW SQL (raw):\n%s", raw)

    final_sql = validate_and_ensure_limit(raw, limit=200)
    logger.info("OVERVIEW SQL (sanitized):\n%s", final_sql)
    return final_sql


def decide_drilldown(
    question: str,
    overview_headers: List[str],
    overview_rows: List[Dict[str, Any]],
    conversation_context: str = "",
) -> Dict[str, Any]:
    system = (
        "You decide whether an additional drill-down query is required.\n\n"
        "Return JSON ONLY, exactly:\n"
        "{\n"
        '  "need_drilldown": true/false,\n'
        '  "drilldown_sql": "",\n'
        '  "why": ""\n'
        "}\n\n"
        "Rules for drilldown_sql:\n"
        "- Athena Presto SQL\n"
        "- ONE query only\n"
        "- No SELECT *\n"
        "- No DDL/DML\n"
        "- MUST include LIMIT 100 or smaller\n"
        "- Must obey the DATA SEMANTICS / RULES in the schema.\n"
    )

    context_block = ""
    if conversation_context.strip():
        context_block = f"\nConversation context (most recent snippets):\n{conversation_context}\n"

    user = f"""
{SCHEMA_DOC}
{context_block}
User question:
{question}

Overview columns:
{overview_headers}

Overview rows (up to 50):
{json.dumps(overview_rows[:50], indent=2)}

Decide if drill-down is needed. Output JSON only.
"""

    raw = call_bedrock(system, user, max_tokens=550, temperature=0.1)
    logger.info("LLM DRILLDOWN DECISION (raw):\n%s", raw)

    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise ValueError(f"Could not parse drilldown JSON. Output:\n{raw}")

    decision: Dict[str, Any] = json.loads(m.group(0))
    logger.info("DRILLDOWN DECISION (parsed):\n%s", json.dumps(decision, indent=2))

    if decision.get("need_drilldown"):
        dd_raw = decision.get("drilldown_sql", "")
        logger.info("LLM DRILLDOWN SQL (raw):\n%s", dd_raw)

        decision["drilldown_sql"] = validate_and_ensure_limit(dd_raw, limit=100)
        logger.info("DRILLDOWN SQL (sanitized):\n%s", decision["drilldown_sql"])

    return decision


def summarize(
    question: str,
    overview_rows: List[Dict[str, Any]],
    drill_rows: List[Dict[str, Any]],
    conversation_context: str = "",
) -> str:
    system = (
        "You are a concise and professional assistant helping users find information about stores.\n\n"
        "Rules:\n"
        "- Use ONLY the provided data (overview + drill-down results).\n"
        "- Answer directly. Do NOT add follow-up questions or suggestions.\n"
        "- If drill-down results are limited/sample, explicitly mention that.\n"
        "- Avoid extra commentary.\n"
    )

    context_block = ""
    if conversation_context.strip():
        context_block = f"\nConversation context (most recent snippets):\n{conversation_context}\n"

    user = f"""
{context_block}
User question:
{question}

Overview results:
{json.dumps(overview_rows, indent=2)[:12000]}

Drill-down results (may be empty):
{json.dumps(drill_rows, indent=2)[:12000]}

Return a direct answer using only the data above.
"""

    text = call_bedrock(system, user, max_tokens=600, temperature=0.2)
    logger.info("LLM SUMMARY OUTPUT:\n%s", text)
    return text


def answer_user_question(
    question: str,
    session_mgr: Optional[BedrockSessionManager] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    If session_mgr is provided, the function will:
      - create/reuse a Bedrock Session (session_id)
      - retrieve recent stored text steps and inject into prompts
      - store this interaction back into the session (as invocation steps)
    """
    logger.info("USER QUESTION: %s", question)

    conversation_context = ""
    invocation_id: Optional[str] = None

    if session_mgr is not None:
        if not session_id:
            session_id = session_mgr.create_session(metadata={"app": "athena-sql-assistant"})
            logger.info("Created Bedrock session_id: %s", session_id)

        conversation_context = session_mgr.get_recent_text_history(session_id=session_id, max_steps=10)
        invocation_id = session_mgr.create_invocation(session_id=session_id, description=f"User question: {question}")

        # Store question as a step
        session_mgr.put_text_step(session_id=session_id, invocation_id=invocation_id, text=f"USER: {question}")

    overview_sql = generate_overview_sql(question, conversation_context=conversation_context)
    overview_headers, overview_rows = run_athena(overview_sql)

    decision = decide_drilldown(
        question,
        overview_headers,
        overview_rows,
        conversation_context=conversation_context,
    )

    drill_headers: List[str] = []
    drill_rows: List[Dict[str, Any]] = []
    if decision.get("need_drilldown") and decision.get("drilldown_sql"):
        drill_headers, drill_rows = run_athena(decision["drilldown_sql"])

    final = summarize(question, overview_rows, drill_rows, conversation_context=conversation_context)

    print("\n=== FINAL ANSWER ===\n")
    print(final)

    # Store outputs in session (SQL + answer)
    if session_mgr is not None and session_id and invocation_id:
        session_mgr.put_text_step(session_id=session_id, invocation_id=invocation_id, text=f"OVERVIEW_SQL: {overview_sql}")
        if decision.get("need_drilldown") and decision.get("drilldown_sql"):
            session_mgr.put_text_step(
                session_id=session_id,
                invocation_id=invocation_id,
                text=f"DRILLDOWN_SQL: {decision['drilldown_sql']}",
            )
        session_mgr.put_text_step(session_id=session_id, invocation_id=invocation_id, text=f"ANSWER: {final}")

    return {
        "session_id": session_id,
        "overview_sql": overview_sql,
        "overview_headers": overview_headers,
        "overview_rows": overview_rows,
        "drilldown_decision": decision,
        "drill_headers": drill_headers,
        "drill_rows": drill_rows,
        "final_answer": final,
    }


def start(user_query: str, sessionid: str):
    session_manager = BedrockSessionManager(region_name=BEDROCK_REGION)
    result = answer_user_question(
        user_query,
        session_mgr=session_manager,
        session_id=sessionid,
    )
    print("\nSession ID:", result.get("session_id"))

if __name__ == "__main__":
    test_question = "What are the stores in Sugar land ?."
    start(test_question, sessionid="8b520838-66ce-4d5f-9f86-db1b25643f5b")