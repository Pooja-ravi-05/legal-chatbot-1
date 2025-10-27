
# whatsapp_agent.py
import os
import json
import logging
import uuid
import asyncio
from datetime import datetime
from typing import Optional, Any, Dict

from fastapi import FastAPI, APIRouter, Request, Response, status, HTTPException
from fastapi.responses import StreamingResponse

import httpx

# --- LangChain / Pinecone / OpenAI (as in your agent_service) ---
# NOTE: The imports below assume you have these packages available in your deployment environment.
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from mangum import Mangum

# ---------------- logging ----------------
logger = logging.getLogger("whatsapp_agent")
logging.basicConfig(level=logging.INFO)

# ---------------- FastAPI app & router ----------------
app = FastAPI()
router = APIRouter()

# ---------------- Configuration / secrets ----------------
# Prefer environment variables. For quick tests you can set them here, but **do not** commit real keys.
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "project")
ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", os.getenv("ACCESS_TOKEN", ""))
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", os.getenv("PHONE_NUMBER_ID", ""))
GRAPH_API_VERSION = os.getenv("GRAPH_API_VERSION", "v22.0")
GRAPH_API_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{PHONE_NUMBER_ID}/messages"

# Agent / Pinecone / OpenAI config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "legal-chat-history-index")
PINECONE_ENV = os.getenv("PINECONE_ENV", None)

if not ACCESS_TOKEN:
    logger.warning("WHATSAPP access token not set. Outbound messages will fail until configured.")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. Agent LLM calls will fail unless provided.")
if not PINECONE_API_KEY:
    logger.warning("PINECONE_API_KEY not set. Pinecone RAG will not persist/retrieve history.")

# ---------------- Simple request models ----------------
class AgentRequest(BaseModel):
    input: str = Field(..., description="User input to send to the agent")

# ---------------- Pinecone client holders ----------------
_pinecone_client: Optional[Any] = None
_pinecone_index_obj: Optional[Any] = None


def init_pinecone() -> Any:
    global _pinecone_client, _pinecone_index_obj
    if _pinecone_index_obj is not None:
        return _pinecone_index_obj
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set in environment")
    try:
        _pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index_obj = _pinecone_client.Index(PINECONE_INDEX)
        logger.info(f"Pinecone index initialized: {PINECONE_INDEX}")
        return _pinecone_index_obj
    except Exception as e:
        logger.exception("Failed to initialize Pinecone client / index")
        raise


def get_pine_index() -> Any:
    global _pinecone_index_obj
    if _pinecone_index_obj is None:
        return init_pinecone()
    return _pinecone_index_obj

# ---------------- RAG tool ----------------
class RAGTool(BaseTool):
    name: str = "rag_search"
    description: str = "Retrieve context from vector store (chat history + docs) and answer the user's question."
    args_schema = AgentRequest  # compatible

    pinecone_index: Optional[Any] = None
    embeddings: Optional[Any] = None
    llm: Optional[Any] = None

    def _run(self, *args, **kwargs):
        # Normalize the query
        query = None
        if "input" in kwargs and kwargs["input"] is not None:
            query = kwargs["input"]
        elif len(args) > 0 and args[0] is not None:
            query = args[0]
        elif len(args) > 0:
            maybe = args[0]
            try:
                query = getattr(maybe, "input", None) or maybe.get("input")
            except Exception:
                query = str(maybe)

        if isinstance(query, (list, dict)):
            query = json.dumps(query)

        if query is None:
            return "No query provided to RAG tool."

        if self.pinecone_index is None or self.embeddings is None or self.llm is None:
            return "RAG tool not initialized (missing pinecone index, embeddings, or llm)."

        try:
            vect = PineconeVectorStore(index=self.pinecone_index, embedding=self.embeddings, text_key="text")
            retriever = vect.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever, chain_type="stuff")
            answer = qa_chain.run(query)
            return {"tool": "rag_search", "result": answer}
        except Exception as e:
            logger.exception("RAG tool error")
            return f"RAG tool error: {str(e)}"

    async def _arun(self, *args, **kwargs):
        return self._run(*args, **kwargs)

# ---------------- Create agent ----------------
def create_custom_agent():
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # init pinecone index (returns a Pinecone Index object) - might raise if not configured
    pine_index = None
    try:
        pine_index = init_pinecone()
    except Exception:
        logger.warning("Pinecone not available; RAG will be disabled but agent can still run without retrieval.")

    rag_tool = RAGTool(pinecone_index=pine_index, embeddings=embeddings, llm=llm)
    tools = [rag_tool]

    system_prompt = (
        "You are an intelligent assistant that answers user questions by retrieving relevant context from a "
        "document + chat-history vector store. Use the internal RAG tool to find and use context. "
        "Do not expose tool internals or raw vectors to the user. Answer concisely and include ₹ currency symbol when "
        "referring to amounts if present in the retrieved context."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# ---------------- Persistence helper ----------------
def persist_message_to_pinecone(text: str, metadata: Dict[str, Any] = None):
    if metadata is None:
        metadata = {}
    if not (PINECONE_API_KEY and OPENAI_API_KEY):
        logger.debug("Skipping persist to Pinecone because credentials are missing.")
        return None
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        emb = embeddings.embed_query(text)
        pine_index = get_pine_index()
        uid = str(uuid.uuid4())
        upsert_items = [(uid, emb, {**metadata, "text": text})]
        pine_index.upsert(vectors=upsert_items)
        logger.debug("Persisted message to Pinecone id=%s", uid)
        return uid
    except Exception as e:
        logger.exception("Failed to persist message to Pinecone: %s", e)
        return None

# ---------------- WhatsApp helper: send text message ----------------
async def send_text_message(to_phone: str, text_body: str) -> Optional[dict]:
    """
    Sends a plain text message to `to_phone` using the WhatsApp Cloud API.
    Returns Graph response JSON on success, otherwise None.
    """
    if not ACCESS_TOKEN:
        logger.error("WHATSAPP access token not configured.")
        return None
    if not PHONE_NUMBER_ID:
        logger.error("WHATSAPP phone number ID not configured.")
        return None

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone,
        "type": "text",
        "text": {"body": text_body},
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(GRAPH_API_URL, headers=headers, json=payload)
            logger.info(f"Reply HTTP status: {resp.status_code}")
            if 200 <= resp.status_code < 300:
                logger.info(f"Sent message to {to_phone}")
                return resp.json()
            else:
                logger.error(f"Failed to send message: {resp.status_code} - {resp.text}")
                return None
    except Exception as e:
        logger.exception("Exception while sending message", exc_info=e)
        return None

# ---------------- Webhook verification endpoint ----------------
@router.get("/")
async def verify_webhook(request: Request):
    qp = request.query_params
    mode = qp.get("hub.mode")
    token = qp.get("hub.verify_token")
    challenge = qp.get("hub.challenge")

    if not VERIFY_TOKEN:
        logger.error("VERIFY_TOKEN is not set.")
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content="Server misconfiguration")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logger.info("WEBHOOK VERIFIED")
        return Response(content=challenge or "", status_code=status.HTTP_200_OK)
    else:
        logger.info(f"Webhook verification failed. mode={mode} token_ok={token==VERIFY_TOKEN}")
        return Response(status_code=status.HTTP_403_FORBIDDEN)

# ---------------- Core webhook receive that routes to agent ----------------
@router.post("/")
async def receive_webhook(request: Request):
    """
    Receives webhook POSTs from Meta/WhatsApp Cloud API.
    For incoming messages: persist -> agent -> persist -> reply to same sender.
    """
    try:
        body = await request.json()
    except Exception:
        raw = await request.body()
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"\n\nWebhook received (raw) {timestamp}\n")
        logger.info(raw)
        return Response(status_code=status.HTTP_200_OK)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"\n\nWebhook received {timestamp}\n")
    logger.info(json.dumps(body, indent=2))

    # Safe navigate structure
    try:
        entries = body.get("entry", [])
        for entry in entries:
            changes = entry.get("changes", [])
            for change in changes:
                value = change.get("value", {})
                messages = value.get("messages", [])
                for msg in messages:
                    from_phone = msg.get("from")
                    message_id = msg.get("id")
                    message_type = msg.get("type")
                    logger.info(f"Incoming message from {from_phone} id={message_id} type={message_type}")

                    # Extract text content for typical text messages
                    user_text = None
                    if message_type == "text":
                        user_text = msg.get("text", {}).get("body")
                    else:
                        # If not text, try to grab caption / fallback
                        user_text = msg.get("text", {}).get("body") if msg.get("text") else None
                        if not user_text:
                            user_text = msg.get("caption") or msg.get("interactive", {}).get("button_reply", {}).get("title")

                    if not user_text:
                        logger.info(f"No usable text in message id={message_id}, skipping agent.")
                        # Could send a message saying "I can only handle text for now"
                        await send_text_message(from_phone, "Sorry — I can only process plain text messages right now.")
                        continue

                    # Persist user message into Pinecone history (best-effort)
                    try:
                        persist_message_to_pinecone(user_text, metadata={"role": "user", "created_at": datetime.utcnow().isoformat(), "whatsapp_from": from_phone})
                    except Exception:
                        logger.exception("persist to pinecone failed (continuing)")

                    # Run the agent: the agent run may be synchronous, so execute in threadpool
                    try:
                        agent_exec = create_custom_agent()
                        payload = {"input": user_text}

                        invoke_fn = getattr(agent_exec, "invoke", None)
                        if invoke_fn is None:
                            run_fn = getattr(agent_exec, "run", None)
                            if run_fn is None:
                                raise RuntimeError("AgentExecutor has neither 'invoke' nor 'run' methods.")
                            loop = asyncio.get_running_loop()
                            # run blocking run_fn in executor
                            result = await loop.run_in_executor(None, lambda: run_fn(payload))
                        else:
                            res = invoke_fn(payload)
                            if hasattr(res, "__await__"):
                                res = await res
                            result = res

                        # Normalize assistant text from resu
                        assistant_text = None
                        if isinstance(result, dict):
                            for key in ("output", "answer", "output_text", "result", "text"):
                                if key in result:
                                    assistant_text = result[key]
                                    break
                            if assistant_text is None:
                                if "tool" in result and "result" in result:
                                    assistant_text = result.get("result")
                                else:
                                    # fallback to stringified dict
                                    assistant_text = json.dumps(result)
                        else:
                            assistant_text = str(result)

                        # persist assistant reply
                        try:
                            persist_message_to_pinecone(assistant_text, metadata={"role": "assistant", "created_at": datetime.utcnow().isoformat(), "whatsapp_to": from_phone})
                        except Exception:
                            logger.exception("persist assistant to pinecone failed (continuing)")

                        # Send reply back over WhatsApp
                        send_result = await send_text_message(from_phone, assistant_text)
                        if send_result:
                            logger.info(f"Reply sent to {from_phone}")
                        else:
                            logger.error(f"Failed to send reply to {from_phone}")

                    except Exception as e:
                        logger.exception("Error running agent")
                        # Try to notify user about the error
                        try:
                            await send_text_message(from_phone, "Sorry — I had an internal error while processing your message.")
                        except Exception:
                            logger.exception("Failed to notify user about agent error")

    except Exception as e:
        logger.exception("Error parsing webhook payload", exc_info=e)

    # Always return 200 quickly so Meta knows we handled the webhook
    return Response(status_code=status.HTTP_200_OK)

# ---------------- Optional health check and agent endpoints ----------------
@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/chat")
async def interact_with_func(req: AgentRequest):
    """
    External HTTP endpoint to call the agent directly (not via WhatsApp).
    Useful for testing / debugging.
    """
    try:
        # persist
        persist_message_to_pinecone(req.input, metadata={"role": "user", "created_at": datetime.utcnow().isoformat()})
        agent_exec = create_custom_agent()
        payload = {"input": req.input}

        invoke_fn = getattr(agent_exec, "invoke", None)
        if invoke_fn is None:
            run_fn = getattr(agent_exec, "run", None)
            if run_fn is None:
                raise RuntimeError("AgentExecutor has neither 'invoke' nor 'run' methods.")
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: run_fn(payload))
        else:
            res = invoke_fn(payload)
            if hasattr(res, "__await__"):
                res = await res
            result = res

        assistant_text = None
        if isinstance(result, dict):
            for key in ("output", "answer", "output_text", "result", "text"):
                if key in result:
                    assistant_text = result[key]
                    break
            if assistant_text is None:
                if "tool" in result and "result" in result:
                    assistant_text = result.get("result")
                else:
                    assistant_text = json.dumps(result)
        else:
            assistant_text = str(result)

        persist_message_to_pinecone(assistant_text, metadata={"role": "assistant", "created_at": datetime.utcnow().isoformat()})
        return {"output": assistant_text}
    except Exception as e:
        logger.exception("Agent error")
        raise HTTPException(status_code=500, detail=str(e))

# mount router at root for webhook endpoints
# app.include_router(router, prefix="")
app.include_router(router)

# Mangum handler for Lambda
handler = Mangum(app)

# ---------------- Notes for Lambda deployment ----------------
# If deploying to AWS Lambda you will typically wrap the FastAPI app with Mangum:
#   from mangum import Mangum
#   handler = Mangum(app)
# Then configure API Gateway to invoke the Lambda.
#
# Requirements (example):
# fastapi, uvicorn, httpx, langchain, langchain-openai, langchain-pinecone, pinecone-client, pydantic, mangum
#
# SECURITY: set OPENAI_API_KEY, PINECONE_API_KEY, WHATSAPP_ACCESS_TOKEN, WHATSAPP_VERIFY_TOKEN, WHATSAPP_PHONE_NUMBER_ID as environment variables.
#
# Logging and observability: tune logging, set timeouts and backoffs for production use.
