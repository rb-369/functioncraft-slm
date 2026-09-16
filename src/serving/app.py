"""Production FastAPI serving application with OpenAI-compatible endpoints."""

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.common.logger import setup_logger
from src.serving.constrained import ConstrainedDecoder
from src.serving.engine import create_engine
from src.serving.telemetry import (
    INFERENCE_REQUESTS_TOTAL,
    SCHEMA_VIOLATIONS_TOTAL,
    PrometheusMiddleware,
    get_metrics_response,
)

logger = setup_logger("serving_app")

# Global state
schemas: dict[str, dict[str, Any]] = {}
engine = None
decoder = None


def init_resources():
    global schemas, engine, decoder
    schemas_dir = Path("data/schemas")
    if schemas_dir.exists():
        for p in schemas_dir.glob("*.json"):
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
                if "name" in data:
                    schemas[data["name"]] = data

    logger.info("Loaded %d tool schemas: %s", len(schemas), list(schemas.keys()))
    decoder = ConstrainedDecoder(schemas)

    engine_type = os.getenv("ENGINE_TYPE", "mock")
    model_path = os.getenv("MODEL_PATH", "models/functioncraft-merged")
    engine = create_engine(engine_type=engine_type, model_path=model_path, schemas=schemas)
    logger.info("Inference engine initialized: %s", type(engine).__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_resources()
    yield


app = FastAPI(
    title="FunctionCraft-SLM Inference Service",
    description="High-throughput, schema-compliant Small Language Model tool calling engine.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(PrometheusMiddleware)


# Pydantic Request/Response Models
class ToolInvokeRequest(BaseModel):
    prompt: str = Field(..., json_schema_extra={"example": "Show top 10 customers signed up in March by lifetime value."})
    preferred_tool: str | None = Field(None, json_schema_extra={"example": "execute_sql_query"})
    stream: bool = False


class ToolInvokeResponse(BaseModel):
    is_valid: bool
    tool_name: str | None
    arguments: dict[str, Any]
    raw_output: str
    latency_ms: float
    error: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "functioncraft-qwen-1.5b"
    messages: list[ChatMessage]
    tools: list[dict[str, Any]] | None = None
    temperature: float = 0.1
    max_tokens: int = 512


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "functioncraft-slm",
        "engine": type(engine).__name__ if engine else "uninitialized",
        "schemas_loaded": len(schemas),
    }


@app.get("/metrics")
def metrics():
    return get_metrics_response()


@app.get("/v1/schemas")
def list_schemas():
    return {"count": len(schemas), "schemas": schemas}


@app.post("/v1/tools/invoke", response_model=ToolInvokeResponse)
def invoke_tool(req: ToolInvokeRequest):
    if not engine or not decoder:
        raise HTTPException(status_code=503, detail="Engine not ready")

    t0 = time.perf_counter()
    raw_output = engine.generate(req.prompt)
    is_valid, tool_dict, err = decoder.decode_and_enforce(raw_output, preferred_tool=req.preferred_tool)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    tool_name = tool_dict.get("name") if tool_dict else None
    args = tool_dict.get("arguments", {}) if tool_dict else {}

    # Telemetry
    if INFERENCE_REQUESTS_TOTAL:
        status_label = "success" if is_valid else "schema_violation"
        INFERENCE_REQUESTS_TOTAL.labels(
            endpoint="/v1/tools/invoke",
            tool_name=tool_name or "unknown",
            status=status_label,
        ).inc()
        if not is_valid and SCHEMA_VIOLATIONS_TOTAL:
            SCHEMA_VIOLATIONS_TOTAL.labels(tool_name=tool_name or "unknown").inc()

    return ToolInvokeResponse(
        is_valid=is_valid,
        tool_name=tool_name,
        arguments=args,
        raw_output=raw_output,
        latency_ms=round(latency_ms, 2),
        error=err,
    )


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint."""
    if not engine or not decoder:
        raise HTTPException(status_code=503, detail="Engine not ready")

    user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    t0 = time.perf_counter()
    raw_output = engine.generate(user_msg)
    is_valid, tool_dict, err = decoder.decode_and_enforce(raw_output)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    tool_calls = []
    if tool_dict and tool_dict.get("name"):
        tool_calls.append({
            "id": f"call_{int(time.time()*1000)}",
            "type": "function",
            "function": {
                "name": tool_dict["name"],
                "arguments": json.dumps(tool_dict.get("arguments", {})),
            },
        })

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None if tool_calls else raw_output,
                    "tool_calls": tool_calls if tool_calls else None,
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(user_msg.split()) * 2,
            "completion_tokens": len(raw_output.split()) * 2,
            "total_tokens": (len(user_msg.split()) + len(raw_output.split())) * 2,
        },
        "latency_ms": round(latency_ms, 2),
    }
