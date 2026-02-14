"""
Chat API endpoint.
Manages session memory and agent invocation.
"""

from fastapi import APIRouter, HTTPException
from uuid import uuid4
from langchain_core.messages import HumanMessage
from schemas.chat import ChatRequest, ChatResponse
from agents.business_agent import build_agent

router = APIRouter()

# Compile agent once at startup
agent_app = build_agent()

# In-memory conversation store (session-based)
conversation_store = {}

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        # Create or reuse session
        session_id = request.session_id or str(uuid4())
        conversation_store.setdefault(session_id, [])

        # Append user message to history
        conversation_store[session_id].append(
            HumanMessage(content=request.message)
        )

        # Invoke agent with full conversation
        result = agent_app.invoke(
            {"messages": conversation_store[session_id]}
        )

        # Extract AI response
        ai_message = result["messages"][-1]

        # Store AI response
        conversation_store[session_id].append(ai_message)

        return ChatResponse(
            response=ai_message.content,
            session_id=session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
