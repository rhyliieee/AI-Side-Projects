from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import Callable, Annotated
from typing import AsyncGenerator, List, Dict
from pydantic import BaseModel

from .graph_flow import graph_builder

# API HANDLING MODULES
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# UTILITIES
import os
from dotenv import load_dotenv
import logging
import gc
import asyncio
import json

# LOAD ENVIRONMENT VARIABLES
load_dotenv()

# INITIALIZE LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# INITIALIZE FASTAPI APPLICATION
app = FastAPI()

# SECURITY HEADERS AND MIDDLEWARE
@app.middleware("http")
async def add_security_headers(request: Request, call_next: Callable) -> Response:
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# CONFIGURE CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API KEY SECURITY SCHEME
API_KEY_NAME = "CPG-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# CONFIGURE DEFAULT AND ALLOWED KEYS
API_KEYS = {
    os.getenv("RHYLE_CPG_API_KEY", "RHYLE_CPG_API_KEY"): "default-user"
}

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header in API_KEYS:
        return api_key_header
    raise HTTPException(
        status_code=403,
        detail="INVALID API KEY"
    )

# DEFINE ROOT ENDPOINT
@app.get("/")
def root(api_key: str = Depends(get_api_key)):
    return {"message": "You have reached the CPG Graph API"}

# CHAT REQUEST FORMAT
class ChatRequest(BaseModel):
    messages: Annotated[
        List[Dict[str, str]], 
        "THE CURRENT USER'S MESSAGE IN THE FORMAT [{'userId': 'dasd6451', 'content': 'content'}]"]
    sessionId: Annotated[str, "THE CURRENT USER'S SESSION ID TO STORE TEMPORARY CHAT MESSAGES"]

# DEFINE AND SECURE ENDPOINT 
@app.post("/api/v1/cpg_chat")
async def invoke_graph(request: ChatRequest, api_key: str = Depends(get_api_key)):
    try:
        cpg_graph = None
        # BUILD THE CPG GRAPH
        if cpg_graph is None and not isinstance(cpg_graph, CompiledStateGraph):
            cpg_graph = graph_builder()
        
        # ADD USER CONFIGURABLE DATA FOR REFERENCE
        config = {"configurable": {"thread_id": request.sessionId, "userId": request.messages[-1].get('userId')}}

        # USER INPUT
        input_message = HumanMessage(content=request.messages[-1].get('content'))
        
        print(f"MESSAGES: {input_message}")
                
        # INVOKE THE GRAPH
        # Stream LLM tokens for messages generated in nodes
        async def generate_stream() -> AsyncGenerator[str, None]:
            first = True
            gathered = []
            messages = {"messages": [input_message]}
            
            try:
                async for msg, metadata in cpg_graph.astream(
                    messages,
                    config=config,
                    stream_mode="messages"
                ):
                    print(f"RECEIVED MESSAGE FROM NODE: {metadata.get('langgraph_node')}")
                    
                    if metadata.get("langgraph_node") in ["general_agent_node", "cpg_agent_node"]:
                        if isinstance(msg, AIMessageChunk):
                            print(f"PROCESSING CHUNK: {msg.content}")
                            gathered.append(msg.content)
                            
                            # FORMAT THE CHUNK FOLLOWING SSE
                            chunk_data = json.dumps({"content": msg.content})
                            yield f"data: {chunk_data}\n\n"
                            
                print(f"FINAL GATHERED CONTENT: {''.join(gathered)}")
                yield f"data: [DONE]\n\n"
                    
            except Exception as e:
                logging.error(f"ERROR IN STREAMING: {str(e)}")
                raise e
                   
        # RETURN THE STREAMING RESPONSE
        return StreamingResponse(
            content=generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except HTTPException as http_exc:
        logger.error(f"HTTP EXCEPTION: {str(http_exc)}")
        raise http_exc
    except Exception as e:
        logger.error(f"GENERAL EXCEPTION: {str(e)}")
        return JSONResponse(content={"status":"error", "message": str(e)}, status_code=500)


# ERROR HANDLER FOR INVALID API KEYS
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# if __name__ == "__main__":
#     gc.collect()
#     asyncio.run(invoke_graph(cpg_graph))