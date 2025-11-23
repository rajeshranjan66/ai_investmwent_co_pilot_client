# This ensures the project root (where utils lives) is on the Python path, so from utils.pdf_export import create_pdf will work when running with Streamlit.
import json
import sys
import os
import re
from datetime import datetime
from pprint import pprint
# python
import asyncio
import functools
import logging
from langchain_core.messages import ToolMessage
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END

#from mcp_client.auth.session_manager import SessionManager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio

from langchain.schema import AIMessage
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
import logging
import uuid
import os
import streamlit as st
import io
import contextlib
import langgraph
from textwrap import dedent
from utils.pdf_export import create_pdf, render_pdf_download
from utils.latex_check import (
    is_latex,
    convert_latex_display_math,
    auto_wrap_latex_math,
    normalize_latex,
    render_latex_blocks,
    render_full_output
)

# from ui.response_renderer import render_response_content
# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import UI components
# Update imports to use relative imports
from ui.ui_components import (
    configure_page_settings,
    create_header,
    create_multi_stock_selection_ui,
    format_analysis_prompts_for_rows,
    create_analysis_buttons,
    create_chat_interface,
    display_agent_thinking,
    render_forecast_table_from_text,
    show_spinner_overlay, inject_seo_metadata

)

# Update import to include UI_STYLES
from ui.ui_constants import ANALYSIS_PROMPTS, UI_STYLES
from mcp_client.auth.auth_ui import AuthUI
from mcp_client.auth.auth_wrapper import require_auth
from mcp_client.auth.navbar_ui import NavbarUI
from mcp_client.auth.auth_manager import AuthManager
from mcp_client.auth.responsive import inject_responsive_css
from mcp_client.auth.animations import inject_animations_css
from mcp_client.auth.auth_manager import check_session_timeout
from mcp_client.auth.auth_ui import AuthUI
from streamlit_js_eval import streamlit_js_eval
import time

from mcp_client.agents.agents_prompt import research_instructions_prompt, fundamental_analyst, technical_analyst, \
    risk_analyst,news_sentiment_analyst,macroeconomic_analyst,valuation_specialist


# Add this right after your imports, before any function definitions

import json





configure_page_settings()
inject_seo_metadata()
inject_responsive_css()
inject_animations_css()

RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() == "true"
# Add this right after your imports, before any other code
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = os.getenv("RAG_ENABLED", "true").lower() == "true"

# 1. Set timeout duration (30 minutes = 1,800,000 ms)
SESSION_TIMEOUT_MS = 1800000

# 2. Track user activity (update on interaction)
activity = streamlit_js_eval(
    js_expressions="Date.now()",
    trigger_on=["mousemove", "mousedown", "keydown", "scroll", "touchstart"],
    key="activity"
)
if activity:
    st.session_state["last_activity"] = activity

# 3. Check for session expiration
now = int(time.time() * 1000)
last = st.session_state.get("last_activity", now)
if now - last > SESSION_TIMEOUT_MS:
    st.session_state["session_expired"] = True
else:
    st.session_state["session_expired"] = False

if st.session_state["session_expired"]:
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; height: 30vh;">
            <div style="background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; border-radius: 4px; padding: 20px; font-size: 1.2rem; text-align: center; max-width: 500px;">
                <strong>Session expired due to inactivity.</strong><br>
                Please refresh to log in again.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

st.markdown("""
<style>
@supports (-webkit-touch-callout: none) {
  @media screen and (max-device-width: 1024px) {
    input::placeholder,
    textarea::placeholder {
        color: #ffffff !important;
        opacity: 1 !important;
    }
    label,
    .stTextInput label,
    .stPasswordInput label {
        color: #ffffff !important;
    }
  }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
""", unsafe_allow_html=True)

st.markdown(UI_STYLES["main_container"], unsafe_allow_html=True)

# Load environment variables
load_dotenv()

DEBUG_LOGGING = os.getenv("DEBUG_LOGGING", "false").lower() == "true"
logging.basicConfig(
    level=logging.DEBUG if DEBUG_LOGGING else logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# DEBUG


# Configuration and setup code
# To avoid printing the unique ID message on every Streamlit rerun, only print it during the initial run (not on reruns triggered by Streamlit)
if "unique_id" not in st.session_state:
    unique_id = uuid.uuid4().hex[0:8]
    # print(f"Stock Analysis ReAct Agent- Unique ID for current execution session: {unique_id}")
    st.session_state["unique_id"] = unique_id
else:
    unique_id = st.session_state["unique_id"]
# print(f"Stock Analysis ReAct Agent- Unique ID for this execution session: {unique_id}")

os.environ.update({
    "LANGCHAIN_TRACING_V2": "true",
    "LANGCHAIN_PROJECT": f"AI Investment Co-Pilot- {unique_id}",
    "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
})

# Initialize the chat model
# model = init_chat_model("gpt-4o")




# Add this near your other configuration variables
# LOCAL_MODEL_ENABLED = os.getenv("LOCAL_MODEL_ENABLED", "false").lower() == "true"
# LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "phi3:mini")  # Ollama model name
#
# DEEPAGENT_ENABLED = os.getenv("DEEPAGENT_ENABLED", "false").lower() == "true"
#
# if LOCAL_MODEL_ENABLED:
#     DEEPAGENT_ENABLED = False
#
# # Replace your current model initialization
# # model = init_chat_model("o4-mini", model_provider="openai")
#
# # Use this instead:
# if LOCAL_MODEL_ENABLED:
#     from langchain_community.chat_models import ChatOllama
#     model = ChatOllama(
#         model=LOCAL_MODEL_NAME,
#         temperature=0,
#         format="json"  # ← CRITICAL for tool calling
#     )
#
# else:
#     model = init_chat_model("o4-mini", model_provider="openai")
#
# print("***********model********", model)

# Local model settings
# ----------------------------
# Local Model Settings
# ----------------------------
LOCAL_MODEL_ENABLED = os.getenv("LOCAL_MODEL_ENABLED", "false").lower() == "true"
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "llama3.2:1b")  # default local model

DEEPAGENT_ENABLED = os.getenv("DEEPAGENT_ENABLED", "false").lower() == "true"

# If local model is enabled → force disable DeepAgent (tiny models cannot handle it)
#local model like llama doesn't support deepagent hence when local model enabled,
# it will be normal run_agent method will be called which will execute traditional graph's node
if LOCAL_MODEL_ENABLED:
    DEEPAGENT_ENABLED = False


# ----------------------------
# Model Initialization
# ----------------------------
if LOCAL_MODEL_ENABLED:
    print(f"🔥 Using LOCAL model via Ollama: {LOCAL_MODEL_NAME}")

    from langchain_community.chat_models import ChatOllama

    # Local Ollama model
    model = ChatOllama(
        model=LOCAL_MODEL_NAME,
        temperature=0,         # deterministic, reliable for debugging
        streaming=False,       # safer for CPU-based local models
        num_ctx=4096           # ensure enough context for your research flow
        # ❗ DO NOT set `format="json"` → small models hallucinate JSON, break tools
    )

    # Debug print to guarantee it's using local server
    print("🔍 LLM Base URL (should be local 11434): http://localhost:11434")

else:
    print("⚡ Using Cloud model: o4-mini (OpenAI)")

    # Your OpenAI initialization
    model = init_chat_model(
        "o4-mini",
        model_provider="openai"
    )

    # Debug print to confirm Cloud usage
    print("🔍 LLM Base URL (should be OpenAI): https://api.openai.com/v1")


print("🎯 Active Model Object →", model)






# model=ChatGoogleGenerativeAI(model='gemini-1.5-flash')
# model = init_chat_model("gpt-5", model_provider="openai")
# model = init_chat_model(
#     "gpt-5",
#     model_provider="openai",
#     extra_body={"effort": "high"}
# )
# "low" → faster, cheaper, less thorough reasoning.
# "medium" → balanced.
# "high" → slower, more expensive, deeper reasoning


import os

# Get the absolute path to the mcp_server directory relative to the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
mcp_server_dir = os.path.join(os.path.dirname(current_dir), 'mcp_server')

# Define the server paths
stock_server_path = os.path.join(mcp_server_dir, 'mcp_stock_server.py')
webcrawler_path = os.path.join(mcp_server_dir, 'mcp_webcrawler_server.py')

# Define multiple MCP servers
# server_params = {
#     # below 2 block is for stdio local  and deployed  running
#     # "stocksAnalysisMCPServer": {
#     #     "command": "python",
#     #     "args": [stock_server_path],
#     #     "transport": "stdio",
#     # },
#     # "WebCrawlerMCP": {
#     #     "command": "python",
#     #     "args": [webcrawler_path],
#     #     "transport": "stdio",
#     # },
#     # for local running when MCP is running in another paycharm
#     # "stocksMCPServerHTTPYFinance": {
#     #     "url": "http://localhost:8001/mcp/",
#     #     "transport": "streamable_http",
#     # },
#
#     # for deployed MCP server running in AWS
#     "stocksMCPServerHTTPYFinance": {
#         "url": "http://stock-mcp-http-internal.gen-ai-mcp-services:8001/mcp/",
#         "transport": "streamable_http",
#     }
#     # "AlphaVantageMCPServer": {
#     #     "command": "python",
#     #     "args": [os.path.join(mcp_server_dir, 'mcp_stock_server_alpha.py')],
#     #     "transport": "stdio",
#     # }
# }


import os

# Determine if we're running locally or in production
IS_PRODUCTION = os.getenv('DEPLOYED') == 'true'

server_params = {
    "stocksMCPServerHTTPYFinance": {
        "url": "https://ai-invesment-mcp-server.fly.dev/mcp/" if IS_PRODUCTION
               else "http://localhost:8001/mcp/",
        "transport": "streamable_http",
    }
}


from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from utils.rag_db import get_context_from_chroma
from typing import TypedDict, List, Optional, Literal, Annotated
import operator


# First, update the CustomState to include user_notes
class CustomState(TypedDict):
    # Multiple nodes can contribute messages in the same step (LLM, tools) → annotate with add
    messages: Annotated[List[BaseMessage], operator.add]
    # Treat these as single-writer keys; do NOT annotate. Only the node that changes them should return them.
    rag_enabled: bool
    approval_status: Optional[Literal["pending", "approved", "rejected"]]
    review_data: Optional[dict]
    user_notes: Optional[str]  # Add this line for user notes
    # If you increment this in multiple nodes in the same step, annotate with add; else leave as plain int.
    iteration_count: Annotated[int, operator.add]




def rag_context_node(state: CustomState):
    """Node to augment user queries with RAG context"""
    print(f"DEBUG [rag_context_node]: ENTERING - iteration: {state.get('iteration_count', 0)}")
    print("*********rag_context_node********RAG Enabled from state:", state["rag_enabled"])
    if not state["rag_enabled"]:
        print(f"DEBUG [rag_context_node]: EXITING - RAG disabled")
        return state  # Skip RAG processing

    # Only process if we have at least one message and the last is from a human
    if len(state["messages"]) > 0 and isinstance(state["messages"][-1], HumanMessage):
        last_msg = state["messages"][-1]
        content = getattr(last_msg, "content", None)

        if content and content.strip():
            try:
                context = get_context_from_chroma(content)
                if context and context.strip():
                    augmented_content = f"## Relevant Research Context:\n{context}\n\n## User Query:\n{content}"

                    # Create a new list of messages, modifying only the last one
                    updated_messages = state["messages"][:-1] + [HumanMessage(content=augmented_content)]

                    return {
                        "messages": updated_messages,
                        "rag_enabled": state["rag_enabled"]
                    }
                    print(f"DEBUG [rag_context_node]: EXITING - augmented with RAG context")


            except Exception as e:
                logging.error("RAG context error: %s", str(e), exc_info=True)

    # Return original state if no augmentation was performed
    # Return original state if no augmentation was performed
    print(f"DEBUG [rag_context_node]: EXITING - no RAG augmentation performed")
    return state





async def call_model(state: CustomState, tools):
    """Invokes the model asynchronously with the current state and tools."""
    print(f"DEBUG [call_model]: ENTERING - iteration: {state.get('iteration_count', 0)}")
    try:
        # DEBUG: Check if we have user notes in the state
        user_notes = state.get("user_notes", "")
        if user_notes:
            print(f"DEBUG [call_model]: User notes available for LLM: '{user_notes}'")

        # Check if we should incorporate user notes into the messages
        if user_notes and state.get("messages"):
            # Find the last human message to append notes to
            messages_copy = state["messages"][:]  # Make a copy to avoid mutating original
            for i, msg in enumerate(reversed(state["messages"])):
                if isinstance(msg, HumanMessage):
                    # Create augmented message with user feedback
                    augmented_content = f"{msg.content}\n\n## User Feedback for Revision:\n{user_notes}"
                    # Replace the message in the copy
                    messages_copy[-(i + 1)] = HumanMessage(content=augmented_content)
                    print(f"DEBUG [call_model]: Augmented human message with user feedback")
                    break
        else:
            messages_copy = state["messages"]

        # Quick fix: Truncate messages if they're getting too long
        total_length = sum(len(str(getattr(msg, 'content', str(msg)))) for msg in messages_copy)
        print("DEBUG [call_model]: Total message length (chars):", total_length)
        if total_length > 600000:  # Rough character limit (150k tokens * 4 chars/token)
            print(f"DEBUG [call_model]: Message length {total_length} exceeds limit, truncating...")

            # Preserve important messages: system messages and last few human/AI pairs
            preserved_messages = []
            recent_messages = []

            # First pass: collect system messages and categorize others
            for msg in messages_copy:
                if hasattr(msg, 'type') and msg.type == 'system':
                    preserved_messages.append(msg)
                else:
                    recent_messages.append(msg)

            # Keep the last human message (which may have user notes) and last few AI responses
            if recent_messages:
                # Keep last 1 messages (1 human-AI pairs) to preserve recent context
                keep_count = min(1, len(recent_messages))
                recent_messages = recent_messages[-keep_count:]

            messages_copy = preserved_messages + recent_messages
            print(f"DEBUG [call_model]: Truncated to {len(messages_copy)} messages")

        # Use model.ainvoke for asynchronous execution in an async graph
        #response = await model.bind_tools(tools).ainvoke(messages_copy)
        if LOCAL_MODEL_ENABLED:
            # For local models - skip bind_tools
            response = await model.ainvoke(messages_copy)
        else:
            # For OpenAI - use bind_tools
            response = await model.bind_tools(tools).ainvoke(messages_copy)

        # The response is already a LangChain AIMessage.
        # Append it to the state and preserve other state variables.
        result = {
            "messages": state["messages"] + [response],
            "rag_enabled": state["rag_enabled"],
            "user_notes": None  # Clear user notes after processing
        }

        print(f"DEBUG [call_model]: EXITING - returning {len(result['messages'])} messages")
        return result

    except Exception as e:
        error_msg = str(e)
        logging.error(f"Agent/model error: {error_msg}", exc_info=True)
        if (
                "context_length_exceeded" in error_msg
                or "maximum context length" in error_msg
                or "token limit" in error_msg
                or "429" in error_msg
                or "rate limit" in error_msg
        ):
            st.warning(
                "⚠️ The request was too large or hit a rate limit. "
                "Please try again later."
            )
        else:
            st.error(f"An error occurred: {error_msg}")
        raise


async def custom_tool_node(state: CustomState, tools):
    """
    Invoke tools for each tool_call on the last assistant message and
    return ToolMessage(s) with matching tool_call_id fields so the LLM's
    tool_call handshake is satisfied.
    """
    print(f"DEBUG [custom_tool_node]: ENTERING - iteration: {state.get('iteration_count', 0)}")
    msgs = state.get("messages") or []
    if not msgs:
        return {}

    last = msgs[-1]
    tool_calls = getattr(last, "tool_calls", None) or getattr(last, "tool_call", None) or []
    if not tool_calls:
        print(f"DEBUG [custom_tool_node]: EXITING - no tool calls")
        return {}

    loop = asyncio.get_running_loop()
    tool_response_msgs = []

    for tc in tool_calls:
        # tc may be an object or dict depending on adapter
        call_id = getattr(tc, "id", None) or tc.get("id") if isinstance(tc, dict) else None
        tool_name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else None)
        # arguments might be named differently (arguments, input, kwargs)
        args = None
        if isinstance(tc, dict):
            args = tc.get("arguments") or tc.get("input") or tc.get("args")
        else:
            args = getattr(tc, "arguments", None) or getattr(tc, "input", None)

        if args is None:
            args = ""

        # Find tool by name - tools may be dict-like or list-like
        tool = None
        if isinstance(tools, dict):
            tool = tools.get(tool_name)
        else:
            # try to match by attribute 'name' or by index
            for t in tools:
                if getattr(t, "name", None) == tool_name or getattr(t, "tool_name", None) == tool_name:
                    tool = t
                    break

        if tool is None:
            logging.error("Tool not found: %s", tool_name)
            tm = ToolMessage(content=f"Tool not found: {tool_name}", tool_call_id=call_id)
            tool_response_msgs.append(tm)
            continue

        # Invoke tool (supporting async API, .ainvoke, .run or sync callable)
        try:
            result = None
            if hasattr(tool, "ainvoke"):
                result = await tool.ainvoke(args)
            elif asyncio.iscoroutinefunction(tool):
                result = await tool(args)
            elif hasattr(tool, "run"):
                # run may be sync -> run in threadpool
                result = await loop.run_in_executor(None, functools.partial(tool.run, args))
            else:
                # fallback to calling the callable in threadpool
                result = await loop.run_in_executor(None, functools.partial(tool, args))

            # Normalize result to string / dict as appropriate
            if isinstance(result, dict):
                # If tool returned a dict with 'output' or similar, try to stringify sensibly
                content = result.get("output") if result.get("output") is not None else str(result)
            else:
                content = str(result)
        except Exception as e:
            logging.exception("Tool invocation failed for %s", tool_name)
            content = f"Tool invocation error for {tool_name}: {e}"

        # Create ToolMessage with the original tool_call_id so assistant can match it
        try:
            tm = ToolMessage(content=content, tool_call_id=call_id)
        except TypeError:
            # Fallback if signature differs; try populating fields that exist
            tm = ToolMessage(content=content)
            # attach call id attribute if possible
            try:
                setattr(tm, "tool_call_id", call_id)
            except Exception:
                pass

        tool_response_msgs.append(tm)

    result = {"messages": tool_response_msgs}
    print(f"DEBUG [custom_tool_node]: EXITING - returning {len(result['messages'])} tool messages")
    return result


def tools_router(state: CustomState):
    """
    Route to 'tools' if the last assistant message contains tool_calls,
    otherwise go to 'human_review'.
    """
    msgs = state.get("messages") or []
    if not msgs:
        return "human_review"
    last = msgs[-1]
    # Support multiple shapes: attribute or dict-like
    tool_calls = getattr(last, "tool_calls", None) or getattr(last, "tool_call", None) or (
        getattr(last, "additional_kwargs", {}).get("tool_calls") if getattr(last, "additional_kwargs", None) else None)
    if not tool_calls:
        # some adapters embed tool call metadata in message.extra or message.metadata
        meta = getattr(last, "metadata", None) or getattr(last, "extra", None)
        if isinstance(meta, dict) and meta.get("tool_calls"):
            tool_calls = meta.get("tool_calls")

    if tool_calls:
        # non-empty -> go to tools node
        return "tools"
    return "human_review"


def human_review_node(state: CustomState):
    """
    Prepare content for human review without mutating input state.
    Return only keys we modify.
    """
    print(f"DEBUG [human_review_node]: ENTERING - iteration: {state.get('iteration_count', 0)}")
    msgs = state.get("messages") or []
    last_message = msgs[-1] if msgs else None
    content = getattr(last_message, "content", "") if last_message else ""
    display_iteration = int(state.get("iteration_count", 0)) + 1
    result= {
        "review_data": {
            "response_content": content,
            "iteration": display_iteration,
            "user_notes": state.get("user_notes", "")  # Include existing user notes if any
        },
        "approval_status": "pending",
    }
    print(f"DEBUG [human_review_node]: EXITING - setting approval_status: pending")
    return result


def rejected_node(state: CustomState):
    """
    Increment iteration and clear approval status. Return only the changed keys.
    We need to track if this is a fresh rejection vs graph loopback.
    """
    print(
        f"DEBUG [rejected_node]: ENTERING - iteration: {state.get('iteration_count', 0)}, approval_status: {state.get('approval_status')}")

    current_iteration = int(state.get("iteration_count", 0))

    # Check if we have user notes - this indicates a fresh human rejection
    user_notes = state.get("user_notes", "")
    has_user_feedback = bool(user_notes and user_notes.strip())

    if has_user_feedback:
        # This is a fresh rejection from human review
        new_iteration = current_iteration + 1
        print(
            f"DEBUG [rejected_node]: INCREMENTING iteration {current_iteration} -> {new_iteration} (fresh human rejection with notes)")
    else:
        # This is a graph loopback, don't increment
        new_iteration = current_iteration
        print(f"DEBUG [rejected_node]: SKIPPING increment - graph loopback. Current: {current_iteration}")

    if user_notes:
        print(f"DEBUG [rejected_node]: User notes received: '{user_notes}'")

    result = {
        "iteration_count": new_iteration,
        "approval_status": None
        #"user_notes": None,
    }
    print(f"DEBUG [rejected_node]: EXITING - returning iteration_count: {new_iteration}")
    return result




def approved_node(state: CustomState):
    """
    No-op terminal node. Return nothing.
    """
    return {}

import asyncio
from typing import List
from langchain_core.tools import StructuredTool, BaseTool
def wrap_mcp_tools_for_sync_and_async(tools: List[BaseTool]) -> List[BaseTool]:
    """
    Ensure MCP StructuredTools can be used in both sync and async contexts.

    DeepAgent / LangGraph sometimes calls tools synchronously (.run),
    but MCP tools from langchain_mcp_adapters are often async-only.
    This wrapper adds a sync func that simply runs the coroutine.
    """
    wrapped: List[BaseTool] = []

    for t in tools:
        # Only touch StructuredTool instances that have a coroutine but no sync func
        if isinstance(t, StructuredTool) and getattr(t, "coroutine", None) and not getattr(t, "func", None):
            async_coro = t.coroutine

            def make_sync(coro):
                def sync_fn(*args, **kwargs):
                    # This runs in a worker thread in LangChain, so asyncio.run is OK
                    return asyncio.run(coro(*args, **kwargs))
                return sync_fn

            sync_fn = make_sync(async_coro)

            # Rebuild a StructuredTool that has both func and coroutine
            wrapped_tool = StructuredTool.from_function(
                func=sync_fn,
                coroutine=async_coro,
                name=t.name,
                description=t.description,
                args_schema=t.args_schema,
            )
            wrapped.append(wrapped_tool)
        else:
            wrapped.append(t)

    return wrapped

async def get_mcp_tools():
    if "mcp_tools" in st.session_state:
        return st.session_state["mcp_tools"]

    client = MultiServerMCPClient(server_params)
    raw_tools = await client.get_tools()
    tools = wrap_mcp_tools_for_sync_and_async(raw_tools)
    st.session_state["mcp_tools"] = tools
    return tools


from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import functools
import streamlit as st

async def get_or_create_graph_app(rag_enabled: bool):
    """
    Build and cache the compiled LangGraph app for the current session.
    Cached separately for rag_enabled=True/False.
    """
    cache_key = f"graph_app_rag_{bool(rag_enabled)}"

    # If already built for this session, reuse it
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    # 1. Get tools ONCE (they're reused by partials)
    tools = await get_mcp_tools()   # your existing helper

    # 2. Build the graph from scratch
    builder = StateGraph(CustomState)

    entry_point = "rag_context" if rag_enabled else "LLM_Call_with_Tool"

    # Optional RAG node
    if rag_enabled:
        builder.add_node("rag_context", rag_context_node)

    # Model invoker
    model_invoker = functools.partial(call_model, tools=tools)
    builder.add_node("LLM_Call_with_Tool", model_invoker)

    # Tools node using custom tool node to preserve state merge
    tool_invoker = functools.partial(custom_tool_node, tools=tools)
    builder.add_node("tools", tool_invoker)

    # HIL nodes
    builder.add_node("human_review", human_review_node)
    builder.add_node("rejected", rejected_node)
    builder.add_node("approved", approved_node)

    # --- REQUIRED ENTRYPOINT ---
    builder.add_edge(START, entry_point)
    if rag_enabled:
        builder.add_edge("rag_context", "LLM_Call_with_Tool")

    # Route LLM to tools OR human_review depending on last assistant message
    builder.add_conditional_edges("LLM_Call_with_Tool", tools_router)

    # Tool loop back to LLM
    builder.add_edge("tools", "LLM_Call_with_Tool")

    # After human_review, branch to approved or rejected based on approval_status
    builder.add_conditional_edges(
        "human_review",
        lambda state: "approved" if state.get("approval_status") == "approved" else "rejected",
    )

    # Rejection router with iteration limit
    MAX_USER_REVISIONS = 4

    def debug_rejected_router(state):
        iteration = state.get("iteration_count", 0)
        result = "LLM_Call_with_Tool" if iteration < MAX_USER_REVISIONS else "approved"
        print(
            f"DEBUG [rejected_router]: iteration={iteration}, "
            f"MAX={MAX_USER_REVISIONS}, routing to: {result}"
        )
        return result

    builder.add_conditional_edges("rejected", debug_rejected_router)

    # Approved is terminal
    builder.add_edge("approved", END)

    # 3. Compile ONCE (with checkpointer + interrupt)
    memory = MemorySaver()
    app = builder.compile(
        checkpointer=memory,
        interrupt_after=["human_review"],
    )

    # 4. Cache app in session_state
    st.session_state[cache_key] = app
    return app


async def run_agent(prompt, rag_enabled=None, user_notes=None):
    """
    Run the agent graph for a single prompt with HIL support.
    Uses a cached LangGraph app per (rag_enabled) and a fresh state per run.
    """
    # Resolve RAG toggle
    if rag_enabled is None:
        rag_enabled = st.session_state.get("rag_enabled", RAG_ENABLED)

    if user_notes:
        print(f"DEBUG [run_agent]: User notes received: '{user_notes}'")
    else:
        print("DEBUG [run_agent]: No user notes provided")

    # Get compiled / cached app
    app = await get_or_create_graph_app(bool(rag_enabled))

    # Persist app/config for UI resume
    thread_id = (
        st.session_state.get("username")
        or st.session_state.get("unique_id")
        or "anon"
    )
    config = {"configurable": {"thread_id": thread_id}}
    st.session_state["graph_app"] = app
    st.session_state["graph_config"] = config

    # Initial state includes HIL fields
    initial_state = {
        "messages": [HumanMessage(content=prompt)],
        "rag_enabled": rag_enabled,
        "approval_status": None,
        "review_data": None,
        "iteration_count": 0,
        "user_notes": user_notes or "",
    }
    print(f"DEBUG [run_agent]: Initial iteration_count: {initial_state['iteration_count']}")

    # Clear previous checkpoint/state for this thread to ensure a fresh run
    app.get_state(config)
    app.update_state(config, None)

    # Run until interrupt (human_review) or completion
    agent_response = await app.ainvoke(
        initial_state,
        {"recursion_limit": 20, **config},
    )

    # If graph paused for human review, mirror to session for the UI
    if isinstance(agent_response, dict) and agent_response.get("approval_status") == "pending":
        st.session_state["approval_status"] = "pending"
        st.session_state["review_data"] = agent_response.get("review_data")

    return agent_response




#backup is below
# async def run_agent(prompt, rag_enabled=None,user_notes=None):
#     """
#     Run the agent graph for a single prompt with HIL support.
#     - Adds optional RAG node.
#     - Uses model invoker and custom tool node.
#     - Adds HIL nodes (human_review, rejected, approved).
#     - Wires conditional edges so LLM -> (tools | human_review),
#       human_review -> (approved | rejected),
#       rejected -> LLM (until iteration limit) -> approved.
#     - Compiles with interrupt_after=['human_review'] so UI can handle approval.
#     """
#     # Resolve RAG toggle
#     if rag_enabled is None:
#         rag_enabled = st.session_state.get("rag_enabled", RAG_ENABLED)
#
#     if user_notes:
#         print(f"DEBUG [run_agent]: User notes received: '{user_notes}'")
#     else:
#         print("DEBUG [run_agent]: No user notes provided")
#
#     # Initialize MCP client and tools
#     # client = MultiServerMCPClient(server_params)
#     # tools = await client.get_tools()
#
#     tools = await get_mcp_tools()
#
#     # Build graph
#     builder = StateGraph(CustomState)
#
#     entry_point = "rag_context" if rag_enabled else "LLM_Call_with_Tool"
#
#     # Optional RAG node
#     if rag_enabled:
#         builder.add_node("rag_context", rag_context_node)
#
#     # Model invoker
#     model_invoker = functools.partial(call_model, tools=tools)
#     builder.add_node("LLM_Call_with_Tool", model_invoker)
#
#     # Tools node using custom tool node to preserve state merge
#     tool_invoker = functools.partial(custom_tool_node, tools=tools)
#     builder.add_node("tools", tool_invoker)
#
#     # HIL nodes
#     builder.add_node("human_review", human_review_node)
#     builder.add_node("rejected", rejected_node)
#     builder.add_node("approved", approved_node)
#
#     # Edges and routing
#     builder.add_edge(START, entry_point)
#     if rag_enabled:
#         builder.add_edge("rag_context", "LLM_Call_with_Tool")
#
#     # Route LLM to tools OR human_review depending on last assistant message
#     builder.add_conditional_edges("LLM_Call_with_Tool", tools_router)
#
#     # Tool loop back to LLM
#     builder.add_edge("tools", "LLM_Call_with_Tool")
#
#     # After human_review, branch to approved or rejected based on approval_status
#     builder.add_conditional_edges(
#         "human_review",
#         lambda state: "approved" if state.get("approval_status") == "approved" else "rejected",
#     )
#
#     # Approved is terminal
#     builder.add_edge("approved", END)
#     # MAX_USER_REVISIONS = 3
#     #Initial run: iteration_count = 0 → goes to human review
#     #First rejection: iteration_count increments to 1 → 1 < 3 → true → goes back to LLM → then to human review again
#     #Second rejection: iteration_count increments to 2 → 2 < 3 → true → goes back to LLM → then to human review again
#     #Third rejection: iteration_count increments to 3 → 3 < 3 → false → goes to approved (no more reviews)
#     MAX_USER_REVISIONS = 4
#     #Rejected loops back to LLM until iteration limit, otherwise go to approved
#     # builder.add_conditional_edges(
#     #     "rejected",
#     #     lambda state: "LLM_Call_with_Tool" if state.get("iteration_count", 0) < MAX_USER_REVISIONS else "approved",
#     # )
#
#     def debug_rejected_router(state):
#         iteration = state.get("iteration_count", 0)
#         result = "LLM_Call_with_Tool" if iteration < MAX_USER_REVISIONS else "approved"
#         print(f"DEBUG [rejected_router]: state iteration={iteration}, MAX={MAX_USER_REVISIONS}, routing to: {result}")
#         print(f"DEBUG [rejected_router]: state keys: {state.keys()}")
#         print(f"DEBUG [rejected_router]: full state: {state}")
#         return result
#
#     builder.add_conditional_edges("rejected", debug_rejected_router)
#
#
#     # Checkpointer and compile — interrupt AFTER human_review so UI can take over
#     memory = MemorySaver()
#     app = builder.compile(checkpointer=memory, interrupt_after=["human_review"])
#
#
#
#     # Persist app/config for UI resume
#     thread_id = st.session_state.get("username") or st.session_state.get("unique_id") or "anon"
#     config = {"configurable": {"thread_id": thread_id}}
#     st.session_state["graph_app"] = app
#     st.session_state["graph_config"] = config
#
#     # Initial state includes HIL fields
#     initial_state = {
#         "messages": [HumanMessage(content=prompt)],
#         "rag_enabled": rag_enabled,
#         "approval_status": None,
#         "review_data": None,
#         "iteration_count": 0,
#         "user_notes": user_notes or "",  # Add user notes to initial state
#     }
#     print(f"DEBUG [run_agent]: Initial iteration_count: {initial_state['iteration_count']}")
#     # Clear previous checkpoint/state for this thread to ensure a fresh run
#     app.get_state(config)
#     app.update_state(config, None)
#
#     # Run until interrupt (human_review) or completion
#     agent_response = await app.ainvoke(
#         initial_state,
#         {"recursion_limit": 20, **config},
#     )
#
#     # If graph paused for human review, mirror to session for the UI
#     if isinstance(agent_response, dict) and agent_response.get("approval_status") == "pending":
#         st.session_state["approval_status"] = "pending"
#         st.session_state["review_data"] = agent_response.get("review_data")
#
#     return agent_response





def truncate_messages_for_deepagent(messages, max_tokens=150000):
    """
    Truncate messages to stay within token limits for deepagent.
    Works with dict-format messages: [{"role": "user", "content": "..."}]
    """
    import tiktoken

    try:
        # Change to:
        if not LOCAL_MODEL_ENABLED:
            encoding = tiktoken.encoding_for_model("gpt-4")
        else:
            # Local models don't need tiktoken
            encoding = None
    except:
        # Fallback estimation: ~4 chars per token
        def count_tokens(text):
            return len(str(text)) // 4
    else:
        def count_tokens(text):
            return len(encoding.encode(str(text)))

    if not messages:
        return messages

    # Always preserve system messages and last user message
    preserved_messages = []
    other_messages = []

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            if role == "system":
                preserved_messages.append(msg)
            elif role == "user" and msg == messages[-1]:  # Last user message (may have user notes)
                preserved_messages.append(msg)
            else:
                other_messages.append(msg)
        else:
            other_messages.append(msg)

    # Count tokens in preserved messages
    preserved_tokens = sum(count_tokens(msg.get("content", str(msg)) if isinstance(msg, dict) else str(msg))
                           for msg in preserved_messages)

    # Add other messages from most recent until we hit the limit
    available_tokens = max_tokens - preserved_tokens - 1000  # Buffer for tools/functions
    selected_messages = []
    current_tokens = 0

    # Add messages from the end (most recent first)
    for msg in reversed(other_messages):
        msg_content = msg.get("content", str(msg)) if isinstance(msg, dict) else str(msg)
        msg_tokens = count_tokens(msg_content)
        if current_tokens + msg_tokens <= available_tokens:
            selected_messages.insert(0, msg)  # Insert at beginning to maintain order
            current_tokens += msg_tokens
        else:
            break

    result = preserved_messages + selected_messages
    print(f"DEBUG [truncate_deepagent]: Reduced {len(messages)} to {len(result)} messages")
    return result



#DeepAgent / LangGraph, however, sometimes tries to call tools in a sync context (tool.run(...) → _run()),
# and when it does that on a StructuredTool that only has coroutine defined, you get:
#NotImplementedError: StructuredTool does not support sync invocation.
#So the fix is: wrap the tools after you fetch them from MultiServerMCPClient, and give each one a small sync shim.



async def run_deepagent(messages, rag_enabled=None, user_notes=None):
    logging.debug("^^^^^^^^^^^^^^^^^^^^Starting run_deepagent:^^^^^^^^^^^^^^^^^^^^^^")
    print("*********message received in run_deepagent********", messages)

    # DEBUG: Check if user notes are provided
    if user_notes:
        print(f"DEBUG [run_deepagent]: User notes received: '{user_notes}'")
    else:
        print("DEBUG [run_deepagent]: No user notes provided")

    # Use session state if not explicitly provided
    if rag_enabled is None:
        rag_enabled = st.session_state.get('rag_enabled', RAG_ENABLED)
        print("*********run_deepagent********RAG Enabled from session:", rag_enabled)

    from deepagents import create_deep_agent
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from mcp_client.agents.user_checkpointer import UserCheckpointer

    # Initialize client and get tools
    # client = MultiServerMCPClient(server_params)
    # tools = await client.get_tools()

    # client = MultiServerMCPClient(server_params)
    # raw_tools = await client.get_tools()
    # tools = wrap_mcp_tools_for_sync_and_async(raw_tools)
    tools = await get_mcp_tools()

    subagents = [fundamental_analyst, technical_analyst]
    #subagents = [fundamental_analyst, technical_analyst, risk_analyst, news_sentiment_analyst, macroeconomic_analyst,valuation_specialist]
    instructions = research_instructions_prompt
    checkpointer = UserCheckpointer()
    user_id = st.session_state.get("username")

    # Create the deep agent
    agent = create_deep_agent(
        tools,
        instructions=instructions,
        model=model,
        subagents=subagents
    )

    try:
        # Add user notes to the prompt if provided (for revision requests)
        if user_notes and user_notes.strip():
            print(f"DEBUG [run_deepagent]: Incorporating user notes into deep agent query")

            # Extract the last user message to augment with notes
            augmented_messages = []
            notes_added = False

            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user" and not notes_added:
                    # Augment the last user message with feedback
                    augmented_content = f"{msg.get('content', '')}\n\n## User Feedback for Revision:\n{user_notes}"
                    augmented_messages.append({"role": "user", "content": augmented_content})
                    notes_added = True
                    print(f"DEBUG [run_deepagent]: Successfully augmented user message with feedback")
                else:
                    augmented_messages.append(msg)

            if not notes_added:
                # If we couldn't find a user message, log a warning but continue without notes
                print(
                    f"DEBUG [run_deepagent]: WARNING - No user message found to attach feedback to. Proceeding without user notes.")
                # Keep original messages unchanged
                augmented_messages = messages

            messages = augmented_messages

        # Add token truncation logic here - BEFORE sending to agent
        total_length = sum(len(str(msg.get("content", str(msg)) if isinstance(msg, dict) else str(msg)))
                           for msg in messages)
        print("DEBUG [run_deepagent]: Total message length (chars): ", total_length)
        if total_length > 600000:  # Rough character limit (150k tokens * 4 chars/token)
            print(f"DEBUG [run_deepagent]: Message length {total_length} exceeds limit, truncating...")
            messages = truncate_messages_for_deepagent(messages, max_tokens=150000)
            print(f"DEBUG [run_deepagent]: Truncated to {len(messages)} messages")

        print("******************Final messages to deep agent**********", messages)

        #DEEPAGENT_RECURSION_LIMIT = int(os.getenv("DEEPAGENT_RECURSION_LIMIT", "10"))
        # 🔧 Recursion limit controlled by UI depth dropdown
        depth_mode = st.session_state.get("deepagent_depth", "Quick")

        if depth_mode == "Quick":
            recursion_limit = 8
        elif depth_mode == "Deep":
            recursion_limit = 20
        else:  # "Normal" or anything else
            recursion_limit = 12

        print(f"DEBUG [run_deepagent]: Using recursion_limit={recursion_limit} for depth_mode={depth_mode}")

        # Invoke the agent
        response = await agent.ainvoke(
            {"messages": messages},
            {"recursion_limit": recursion_limit}
        )

        checkpointer.save(user_id, response)
        logging.debug("*******************LLM response: %s State: %s", user_id, response)
        print("*******************LLM response received in run_deepagent: ", response)

        state = checkpointer.load(user_id)


    except langgraph.errors.GraphRecursionError as e:
        # DeepAgent hit the step limit – use partial state and summarize with existing model
        logging.error("GraphRecursionError encountered: %s", str(e), exc_info=True)
        partial_state = getattr(e, "partial_state", {"messages": []})
        partial_messages = partial_state.get("messages", [])

        print("DEBUG [run_deepagent]: Partial state messages on recursion error:", partial_messages)

        # Build a summarization prompt for your existing model
        summary_input = [
            HumanMessage(
                content=(
                    "The DeepAgent ran out of its reasoning step budget.\n"
                    "Based ONLY on the partial analysis and tool outputs below, "
                    "produce the best possible concise answer you can for the user.\n\n"
                    "Partial trace (messages):\n"
                    f"{partial_messages}"
                )
            )
        ]

        try:
            # Use your existing model (same one used for Express Analysis)
            summary_msg = await model.ainvoke(summary_input)
            print("DEBUG [run_deepagent]: Generated fallback summary from partial state.")
        except Exception as e2:
            logging.error(
                "Error while summarizing partial state after recursion error: %s",
                str(e2),
                exc_info=True,
            )
            summary_msg = AIMessage(
                content=(
                    "I reached my reasoning step limit and couldn't complete the full deep research. "
                    "I also failed to summarize the partial result due to an internal error."
                )
            )

        response = {"messages": [summary_msg]}
        checkpointer.save(user_id, response)
        logging.debug(
            "********************Checkpoint saved for user after recursion error: %s State: %s",
            user_id,
            response,
        )

    except Exception as e:
        logging.error("Unexpected error in deep agent: %s", str(e), exc_info=True)
        # Return empty response to prevent breaking the UI
        response = {"messages": [AIMessage(content=f"Error processing request: {str(e)}")]}

    print("Agent response received.")
    return response




async def process_query(prompt: str, action: str = None):
    """
    Process a user prompt. Chooses between run_agent and run_deepagent (if enabled),
    preserves HIL approval state, captures agent 'thinking' output, normalizes response
    shapes (dict/list/BaseMessage/string), stores tab-scoped last content/response/query,
    and retains only last assistant + last user in tab1_messages for context.
    """
    spinner_key = f"spinner_{action or 'tab'}"
    st.session_state.setdefault("approval_status", None)
    st.session_state.setdefault("review_data", None)
    rag_enabled = st.session_state.get("rag_enabled")
    print("DEBUG [process_query]: rag_enabled: ",rag_enabled)
    st.session_state[spinner_key] = True
    # Get user notes if this is a revision request
    user_notes = st.session_state.get("hil_user_notes", "")
    if user_notes:
        print(f"DEBUG [process_query]: Processing with user notes: '{user_notes}'")
    print("*****************************action:", action)
    print("*****************************DEEPAGENT_ENABLED:", DEEPAGENT_ENABLED)
    print(f"DEBUG [process_query]: LOCAL_MODEL_ENABLED: '{LOCAL_MODEL_ENABLED}'")
    print(f"DEBUG [process_query]: LOCAL_MODEL_NAME: '{LOCAL_MODEL_NAME}'")


    try:
        # Build messages_for_deepagent when deepagent is used and action == "tab1"
        messages_for_deepagent = None
        if DEEPAGENT_ENABLED and action == "tab1":
            tab1_messages = st.session_state.get("tab1_messages", []) or []

            # Ensure current user message appended once (UI may have already appended)
            last_item = tab1_messages[-1] if tab1_messages else None
            if not (isinstance(last_item, dict) and last_item.get("role") == "user" and last_item.get(
                    "content") == prompt):
                tab1_messages.append({"role": "user", "content": prompt})
                st.session_state["tab1_messages"] = tab1_messages

            # Find the last meaningful assistant content from the message history
            last_ai_content = None
            for msg in reversed(tab1_messages[:-1]):  # Exclude the just-added user message
                # Handle dict format messages
                if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                    last_ai_content = msg.get("content")
                    break

                # Handle BaseMessage objects (like AIMessage from the response)
                elif isinstance(msg, BaseMessage):
                    # For AIMessage objects, we need to check the type, not just role
                    if isinstance(msg, AIMessage):
                        content = getattr(msg, "content", None)
                        if content and content.strip():
                            last_ai_content = content
                            break
                    # Handle other BaseMessage types that might represent assistant responses
                    else:
                        role = getattr(msg, "role", None) or getattr(msg, "type", "").lower()
                        content = getattr(msg, "content", None)

                        if (role in ["assistant", "ai"] and content and content.strip()):
                            last_ai_content = content
                            break

            print("************************last_ai_content*****************: ", last_ai_content)

            # Build messages_for_deepagent with last exchange (user+assistant) + current user message
            messages_for_deepagent = []

            # Find the last user message (not current prompt)
            last_user_content = None
            for msg in reversed(tab1_messages[:-1]):  # Exclude the just-added current user message
                if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                    last_user_content = msg.get("content")
                    break
                elif isinstance(msg, BaseMessage):
                    role = getattr(msg, "role", None) or getattr(msg, "type", "").lower()
                    content = getattr(msg, "content", None)
                    if role == "user" and content:
                        last_user_content = content
                        break

            # Add the last exchange (user + assistant) if both exist
            if last_user_content and last_ai_content:
                messages_for_deepagent.append({"role": "user", "content": last_user_content})
                messages_for_deepagent.append({"role": "assistant", "content": last_ai_content})

            # Always add the current user message
            messages_for_deepagent.append({"role": "user", "content": prompt})

            print("************************messages_for_deepagent*****************: ", messages_for_deepagent)
            print("************************messages_for_deepagent type*****************: ", type(messages_for_deepagent))


        # Call appropriate agent
        #local model like llama doesn't support deepagent hence when local model enabled,
        # it will be normal run_agent method will be called which will execute traditional graph's node
        if LOCAL_MODEL_ENABLED :
            response = await run_agent(prompt, rag_enabled=st.session_state.get("rag_enabled", RAG_ENABLED),
                                       user_notes=user_notes)
        elif DEEPAGENT_ENABLED and action == "tab1":
            response = await run_deepagent(messages_for_deepagent,
                                           rag_enabled=st.session_state.get("rag_enabled", RAG_ENABLED),
                                           user_notes=user_notes)
        else:
            response = await run_agent(prompt, rag_enabled=st.session_state.get("rag_enabled", RAG_ENABLED),
                                       user_notes=user_notes)



        # if DEEPAGENT_ENABLED and action == "tab1":
        #     print("*********message sent to deepagent flow********", prompt)
        #     response = await run_deepagent(messages_for_deepagent,
        #                                    rag_enabled=st.session_state.get("rag_enabled", RAG_ENABLED),user_notes=user_notes)
        # else:
        #     # For LLM/HIL flow we pass the raw prompt (run_agent handles RAG + human review)
        #     response = await run_agent(prompt, rag_enabled=st.session_state.get("rag_enabled", RAG_ENABLED),user_notes=user_notes)

        tab_key = "tab1" if action == "tab1" else "tab2"
        # Persist canonical response + timestamp
        st.session_state[f"last_response_{tab_key}"] = response
        st.session_state[f"last_message_time_{tab_key}"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # If agent requested human review, mirror to session state for UI
        if isinstance(response, dict) and response.get("approval_status") == "pending":
            st.session_state["approval_status"] = "pending"
            st.session_state["review_data"] = response.get("review_data")
            # Store which tab this review came from
            st.session_state["graph_origin_tab"] = tab_key

        # Normalize different response shapes into a list of message-like dicts/objects
        messages = []
        if isinstance(response, dict) and "messages" in response:
            msgs = response.get("messages") or []
        elif isinstance(response, list):
            msgs = response
        elif hasattr(response, "messages"):
            try:
                msgs = getattr(response, "messages") or []
            except Exception:
                msgs = [response]
        elif isinstance(response, (str, bytes)):
            msgs = [str(response)]
        else:
            msgs = [response]

        for m in msgs:
            if isinstance(m, BaseMessage):
                # keep object as-is (AIMessage, HumanMessage, etc.)
                messages.append(m)
            elif isinstance(m, dict) and "role" in m and "content" in m:
                messages.append(m)
            else:
                # fallback to simple string wrapper
                messages.append({"role": "assistant", "content": str(m)})

        #Pretty-print/capture agent thinking
        pretty_output = io.StringIO()
        with contextlib.redirect_stdout(pretty_output):
            for m in messages:
                try:
                    if hasattr(m, "pretty_print"):
                        m.pretty_print()
                    else:
                        # print role/content for dict-like or BaseMessage without pretty_print
                        role = getattr(m, "role", None) if isinstance(m, BaseMessage) else (
                            m.get("role") if isinstance(m, dict) else None)
                        content = getattr(m, "content", None) if isinstance(m, BaseMessage) else (
                            m.get("content") if isinstance(m, dict) else str(m))
                        if role:
                            print(f"[{role}] {content}")
                        else:
                            print(content)
                except Exception:
                    # safe fallback
                    print(repr(m))

        #st.session_state["agent_thinking"] = pretty_output.getvalue()
        current_thinking = st.session_state.get("agent_thinking", "")
        new_thinking = pretty_output.getvalue()
        st.session_state["agent_thinking"] = current_thinking + "\n\n--------------\n\n" + new_thinking
        if st.session_state.get("agent_thinking"):
            display_agent_thinking(st.session_state["agent_thinking"])





        # Clear user notes after processing to prevent reuse
        for key in ["hil_user_notes", "user_notes"]:
            if key in st.session_state:
                st.session_state[key] = ""


        # Extract last message content robustly
        last_content = None
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, BaseMessage):
                last_content = getattr(last_msg, "content", None)
            elif isinstance(last_msg, dict):
                last_content = last_msg.get("content")
            else:
                last_content = str(last_msg)

        # Store tab-specific last content / response / query
        st.session_state[f"last_response_{tab_key}"] = response
        st.session_state[f"last_query_{tab_key}"] = prompt
        if last_content:
            st.session_state[f"last_content_{tab_key}"] = last_content

        # Maintain minimal context: keep last assistant + last user message in tab1_messages
        try:
            tab1_messages = st.session_state.get("tab1_messages", []) or []
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@tab1_messages before update:", tab1_messages)

            # append assistant response if not already present
            assistant_content = None
            # find assistant content from normalized messages
            for m in reversed(messages):
                if isinstance(m, AIMessage):  # Specifically check for AIMessage
                    content = getattr(m, "content", None)
                    if content:
                        assistant_content = content
                        break
                elif isinstance(m, dict) and m.get("role") == "assistant" and m.get("content"):
                    assistant_content = m.get("content")
                    break
                elif isinstance(m, BaseMessage):
                    role = getattr(m, "role", None) or getattr(m, "type", "").lower()
                    content = getattr(m, "content", None)
                    if role in ["assistant", "ai"] and content:
                        assistant_content = content
                        break

            if assistant_content:
                # add assistant as dict if last item is not the same assistant content
                if not (tab1_messages and isinstance(tab1_messages[-1], dict) and
                        tab1_messages[-1].get("role") == "assistant" and
                        tab1_messages[-1].get("content") == assistant_content):
                    tab1_messages.append({"role": "assistant", "content": assistant_content})
                    print("@@@@@@@@@@@@@@ Added assistant message to tab1_messages")

            # Find the last user message (not necessarily the current one)
            last_user_content = None
            for msg in reversed(tab1_messages):
                if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                    last_user_content = msg.get("content")
                    break
                elif isinstance(msg, BaseMessage):
                    role = getattr(msg, "role", None) or getattr(msg, "type", "").lower()
                    content = getattr(msg, "content", None)
                    if role == "user" and content:
                        last_user_content = content
                        break

            # Ensure we have the last user message (not necessarily the current prompt)
            if last_user_content and not (tab1_messages and isinstance(tab1_messages[-1], dict) and
                                          tab1_messages[-1].get("role") == "user" and
                                          tab1_messages[-1].get("content") == last_user_content):
                tab1_messages.append({"role": "user", "content": last_user_content})
                print("@@@@@@@@@@@@@@ Added last user message to tab1_messages")

            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@tab1_messages after update:", tab1_messages)

            # Trim to keep only last assistant + last user message
            assistant_found = False
            user_found = False
            trimmed = []

            # Find last assistant and last user messages
            for msg in reversed(tab1_messages):
                if isinstance(msg, dict):
                    role = msg.get("role")
                    content = msg.get("content")

                    if role == "assistant" and content and not assistant_found:
                        trimmed.insert(0, msg)  # Add assistant at beginning
                        assistant_found = True
                    elif role == "user" and content and not user_found:
                        trimmed.insert(0, msg)  # Add user at beginning
                        user_found = True

                elif isinstance(msg, BaseMessage):
                    role = getattr(msg, "role", None) or getattr(msg, "type", "").lower()
                    content = getattr(msg, "content", None)

                    if role == "assistant" and content and not assistant_found:
                        trimmed.insert(0, {"role": "assistant", "content": content})
                        assistant_found = True
                    elif role == "user" and content and not user_found:
                        trimmed.insert(0, {"role": "user", "content": content})
                        user_found = True

                # Stop if we found both messages
                if assistant_found and user_found:
                    break

            # If we didn't find both, keep what we have
            if not trimmed:
                trimmed = tab1_messages[-2:]  # Fallback: keep last 2 messages

            st.session_state["tab1_messages"] = trimmed
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@tab1_messages after trimming:", trimmed)

        except Exception as e:
            logging.exception("Error updating tab1_messages context; continuing gracefully.")
            print(f"Error in tab1_messages update: {e}")

        return response

    except Exception as e:
        logging.error("Unexpected error in process_query: %s", str(e), exc_info=True)
        st.error(f"An error occurred: {str(e)}")
        tab_key = "tab1" if action == "tab1" else "tab2"
        fallback = {"messages": [AIMessage(content=f"Error processing request: {str(e)}")]}
        st.session_state[f"last_response_{tab_key}"] = fallback
        st.session_state[f"last_query_{tab_key}"] = prompt
        return fallback

    finally:
        st.session_state[spinner_key] = False



def render_human_review_ui():
    """
    Render human review panel when st.session_state['approval_status'] == 'pending'.
    Centers compact approve / request revision buttons and persists resumed response
    for both tab1 and tab2 (restores minimal tab1 context).
    Returns True when shown, False otherwise.
    """
    if st.session_state.get("approval_status") != "pending":
        return False

    review = st.session_state.get("review_data") or {}
    content = review.get("response_content") or ""
    iteration = int(review.get("iteration") or 1)

    if "origin_tab" not in review:
        origin = st.session_state.get("graph_origin_tab", "tab1")
        # Make sure we have a mutable copy of review data
        review_data = review.copy()
        review_data["origin_tab"] = origin
        st.session_state["review_data"] = review_data

    st.markdown("---")
    st.markdown("### 👥 Human Review Required")
    st.markdown(
        f'<div class="time-indicator">📋 Iteration {iteration} - Please review the response</div>',
        unsafe_allow_html=True,
    )
    # Use display_analysis_results for consistent formatting
    origin = st.session_state.get("graph_origin_tab", "tab1")
    user_query = st.session_state.get(f"last_query_{origin}")

    # Create a temporary response object for display_analysis_results
    temp_response = {"messages": [AIMessage(content=content)]}

    display_analysis_results(temp_response, f"review_{origin}", user_query=user_query)

    # Scoped CSS: keep buttons compact and inline (target Streamlit wrappers)
    st.markdown(
        r"""
        <style>
        .hil-center{ display:flex; gap:10px; justify-content:center; align-items:center; margin:8px 0; flex-wrap:nowrap;}
        /* Make Streamlit button wrappers not stretch */
        .hil-center > div[data-testid="stVerticalBlock"],
        .hil-center > div[data-testid="stButton"],
        .hil-center > div[class^="css"] {
            width: auto !important;
            display: inline-block !important;
        }
        /* Style the actual buttons */
        .hil-center button {
            min-width: 120px;
            max-width: 220px;
            padding: 6px 10px;
            font-size: 0.95rem;
            border-radius: 8px;
            white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _handle_approval_rejection(approval_status):
        """Handle approval/rejection with overlay showing during processing"""
        # Store user notes in session state for processing
        user_notes = st.session_state.user_notes
        st.session_state["hil_user_notes"] = user_notes

        # DEBUG: Print what we're capturing
        if approval_status == "rejected" and user_notes:
            print(f"DEBUG [button]: User notes captured for revision: '{user_notes}'")
        elif approval_status == "rejected":
            print("DEBUG [button]: No user notes captured for revision request")

        # Show overlay immediately
        st.session_state["show_overlay_during_hil"] = True
        st.session_state["hil_processing_status"] = approval_status
        st.rerun()

    # Layout: center buttons in middle column and place them inline
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.markdown('<div class="hil-center">', unsafe_allow_html=True)

        if st.button("✅ Approve & Send", type="primary", use_container_width=False, key="approve_send"):
            _handle_approval_rejection("approved")

        if st.button("🔄 Request Revision", type="secondary", use_container_width=False, key="request_revision"):
            _handle_approval_rejection("rejected")

        st.markdown('</div>', unsafe_allow_html=True)

    return True


# Update the render_human_review_ui function to include the text area
def render_human_review_ui():
    """
    Render human review panel when st.session_state['approval_status'] == 'pending'.
    Centers compact approve / request revision buttons and text area for user notes.
    Returns True when shown, False otherwise.
    """
    if st.session_state.get("approval_status") != "pending":
        return False

    review = st.session_state.get("review_data") or {}
    content = review.get("response_content") or ""
    iteration = int(review.get("iteration") or 1)
    existing_notes = review.get("user_notes", "")

    if "origin_tab" not in review:
        origin = st.session_state.get("graph_origin_tab", "tab1")
        # Make sure we have a mutable copy of review data
        review_data = review.copy()
        review_data["origin_tab"] = origin
        st.session_state["review_data"] = review_data

    st.markdown("---")
    st.markdown("### 👥 Human Review Required")
    st.markdown(
        f'<div class="time-indicator">📋 Iteration {iteration} - Please review the response</div>',
        unsafe_allow_html=True,
    )

    # Use display_analysis_results for consistent formatting
    origin = st.session_state.get("graph_origin_tab", "tab1")
    user_query = st.session_state.get(f"last_query_{origin}")

    # Create a temporary response object for display_analysis_results
    temp_response = {"messages": [AIMessage(content=content)]}
    display_analysis_results(temp_response, f"review_{origin}", user_query=user_query)

    # Add text area for user notes
    st.markdown("#### 💬 Feedback Notes (Optional)")
    st.markdown("*Add specific feedback to guide the revision (max 100 characters)*")

    # Initialize user_notes in session state if not exists
    if "user_notes" not in st.session_state:
        st.session_state.user_notes = existing_notes

    user_notes = st.text_area(
        "Your feedback:",
        value=st.session_state.user_notes,
        placeholder="What should be improved in the next revision?",
        max_chars=100,
        key="user_notes_input",
        label_visibility="collapsed"
    )

    # Update session state with user notes
    st.session_state.user_notes = user_notes

    # Scoped CSS: keep buttons compact and inline
    st.markdown(
        r"""
        <style>
        .hil-center{ 
            display:flex; 
            gap:10px; 
            justify-content:center; 
            align-items:center; 
            margin:16px 0; 
            flex-wrap:nowrap;
        }
        .hil-center > div[data-testid="stVerticalBlock"],
        .hil-center > div[data-testid="stButton"],
        .hil-center > div[class^="css"] {
            width: auto !important;
            display: inline-block !important;
        }
        .hil-center button {
            min-width: 120px;
            max-width: 220px;
            padding: 8px 12px;
            font-size: 0.95rem;
            border-radius: 8px;
            white-space: nowrap;
        }
        .user-notes-container {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
            border-left: 4px solid #2e6bb7;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _handle_approval_rejection(approval_status):
        """Handle approval/rejection with overlay showing during processing"""
        # Store user notes in session state for processing
        st.session_state["hil_user_notes"] = st.session_state.user_notes
        st.session_state["show_overlay_during_hil"] = True
        st.session_state["hil_processing_status"] = approval_status
        st.rerun()

    # Layout: center buttons
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.markdown('<div class="hil-center">', unsafe_allow_html=True)

        if st.button("✅ Approve & Send", type="primary", use_container_width=False, key="approve_send"):
            _handle_approval_rejection("approved")

        if st.button("🔄 Request Revision", type="secondary", use_container_width=False, key="request_revision"):
            _handle_approval_rejection("rejected")

        st.markdown('</div>', unsafe_allow_html=True)

    return True



# Update the process_hil_decision function to handle user notes
async def process_hil_decision():
    """Process the human review decision with overlay showing"""
    approval_status = st.session_state.get("hil_processing_status")
    user_notes = st.session_state.get("hil_user_notes", "")

    # DEBUG: Print what we're processing
    if approval_status == "rejected" and user_notes:
        print(f"DEBUG [process_hil_decision]: Processing revision with notes: '{user_notes}'")
    elif approval_status == "rejected":
        print("DEBUG [process_hil_decision]: Processing revision without notes")


    # ALWAYS use the origin tab from the review data with fallback
    # ALWAYS use the origin tab from the review data with fallback
    review_data = st.session_state.get("review_data") or {}
    origin = review_data.get("origin_tab", st.session_state.get("graph_origin_tab", "tab1"))

    # Store this for the rest of the processing
    st.session_state["hil_origin_tab"] = origin

    # Get the current graph state before making changes
    app = st.session_state.get("graph_app")
    cfg = st.session_state.get("graph_config") or {}

    if not app or not cfg:
        logging.error("No graph app or config found for HIL decision processing")
        st.session_state[f"last_content_{origin}"] = "Error: Could not resume agent processing"
        # Clear HIL states
        st.session_state["show_overlay_during_hil"] = False
        st.session_state["hil_processing_status"] = None
        st.session_state.pop("hil_origin_tab", None)
        return

    try:
        # Get the current state snapshot and extract the actual state
        state_snapshot = app.get_state(cfg)
        current_state = state_snapshot.values if hasattr(state_snapshot, 'values') else {}

        print(f"DEBUG [process_hil_decision]: Loaded iteration_count: {current_state.get('iteration_count')}")
        print(f"DEBUG [process_hil_decision]: Loaded state ID: {getattr(state_snapshot, 'id', 'None')}")

        # DEBUG: Print current state before update
        print(f"DEBUG [process_hil_decision]: Current state before update: {current_state.keys()}")
        print(f"DEBUG [process_hil_decision]: Current iteration_count: {current_state.get('iteration_count', 0)}")
        print(f"DEBUG [process_hil_decision]: Current approval_status: {current_state.get('approval_status', 'None')}")

        # Update the state with approval_status and user_notes
        updated_state = {
            **current_state,
            "approval_status": approval_status,
            "user_notes": user_notes if approval_status == "rejected" else ""  # Only pass notes for rejections
        }
        # DEBUG: Print updated state
        print(f"DEBUG [process_hil_decision]: Updated state: {updated_state.keys()}")
        print(f"DEBUG [process_hil_decision]: Updated user_notes: '{updated_state.get('user_notes', 'None')}'")

        # Update the state in the checkpointer
        app.update_state(cfg, updated_state)

        # Resume execution from the current state
        print(f"DEBUG [process_hil_decision]: Resuming graph execution...")
        resp = await app.ainvoke(None, {"recursion_limit": 20, **cfg})
        print(f"DEBUG [process_hil_decision]: Graph resumed successfully")

        # Check if we're back in human review (another iteration)
        if isinstance(resp, dict) and resp.get("approval_status") == "pending":
            # We're back in human review - update session state
            st.session_state["approval_status"] = "pending"
            st.session_state["review_data"] = resp.get("review_data")
            # Ensure the origin tab is preserved in the new review data
            if "review_data" in st.session_state and st.session_state["review_data"]:
                st.session_state["review_data"]["origin_tab"] = origin

            # Clear processing flags but keep overlay for the new review cycle
            st.session_state["show_overlay_during_hil"] = False
            st.session_state["hil_processing_status"] = None
            st.session_state.pop("hil_origin_tab", None)
            return

        # If we reached the end, process the final response
        # Persist resumed response with tab-specific keys
        st.session_state[f"last_response_{origin}"] = resp

        # Store the user query with tab-specific key
        user_query = st.session_state.get(f"last_query_{origin}")
        if user_query:
            st.session_state[f"last_query_{origin}"] = user_query

        # Extract content from response
        extracted_text = ""
        if isinstance(resp, dict) and "messages" in resp:
            messages = resp.get("messages", []) or []
            # Find last assistant message
            for m in reversed(messages):
                if isinstance(m, BaseMessage) and isinstance(m, AIMessage):
                    extracted_text = getattr(m, "content", "") or ""
                    break
                elif isinstance(m, dict) and m.get("role") == "assistant":
                    extracted_text = m.get("content", "") or ""
                    break
            if not extracted_text and messages:
                last_msg = messages[-1]
                if isinstance(last_msg, BaseMessage):
                    extracted_text = getattr(last_msg, "content", "") or ""
                elif isinstance(last_msg, dict):
                    extracted_text = last_msg.get("content", "") or ""

        # Store the extracted content for display
        if extracted_text:
            st.session_state[f"last_content_{origin}"] = extracted_text

        # For tab1: maintain chat history
        if origin == "tab1":
            tab1_messages = st.session_state.get("tab1_messages", []) or []
            # Add user message if not already present
            user_query = st.session_state.get(f"last_query_{origin}") or st.session_state.get("last_query_tab1")
            if user_query and not any(
                    isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content") == user_query
                    for msg in tab1_messages
            ):
                tab1_messages.append({"role": "user", "content": user_query})

            # Add assistant response
            if extracted_text and not any(
                    isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get(
                        "content") == extracted_text
                    for msg in tab1_messages
            ):
                tab1_messages.append({"role": "assistant", "content": extracted_text})

            st.session_state["tab1_messages"] = tab1_messages[-4:]  # Keep last 2 exchanges

    except Exception as e:
        logging.exception("Failed to resume graph on %s", approval_status)
        # Store error message
        st.session_state[f"last_content_{origin}"] = f"Error processing your decision: {str(e)}"

    # Clear HIL UI state and return control to main UI
    st.session_state["approval_status"] = None
    st.session_state["review_data"] = None
    st.session_state.pop("graph_origin_tab", None)
    st.session_state["show_overlay_during_hil"] = False
    st.session_state["hil_processing_status"] = None
    st.session_state.pop("hil_origin_tab", None)
    st.session_state.pop("user_notes", None)  # Clear user notes
    st.session_state.pop("hil_user_notes", None)  # Clear stored notes

    # Preserve the current tab before rerun - THIS IS CRITICAL
    st.session_state["preserve_tab"] = origin

    # Debug print to verify user notes are being passed
    if approval_status == "rejected" and user_notes:
        print(f"DEBUG: User notes passed to LLM for revision: '{user_notes}'")
    elif approval_status == "rejected":
        print("DEBUG: No user notes provided for revision request")


from utils.graph_detector import extract_and_render_graphs

def display_analysis_results(content, tab_key, user_query=None):
    """
    Display analysis results for a tab.

    - Normalizes different response shapes (dict with messages, list, BaseMessage, str).
    - Preserves HIL behavior by extracting the assistant's last message when given a response object.
    - Computes and stores a final elapsed time (final_elapsed_{tab_key}) so it doesn't change across reruns.
    - Detects table-like content and renders accordingly.
    - Normalizes and renders LaTeX when appropriate.
    - Safely calls render_forecast_table_from_text and render_pdf_download and logs errors.
    - Updates session state last_content_{tab_key} and last_message_time.
    """

    # Normalize content to a plain string for rendering while preserving original for debugging
    try:

        # Helper to extract text from various message shapes
        def _extract_text_from_message(msg):
            if msg is None:
                return ""
            # BaseMessage (langchain/langchain_core)
            if isinstance(msg, BaseMessage):
                return getattr(msg, "content", "") or ""
            # AIMessage from langchain.schema
            if isinstance(msg, AIMessage):
                return getattr(msg, "content", "") or ""
            # dict with role/content
            if isinstance(msg, dict):
                # If message dict contains nested messages, skip — caller handles lists
                return msg.get("content") or msg.get("text") or ""
            # fallback for simple strings or bytes
            if isinstance(msg, (str, bytes)):
                return msg.decode("utf-8") if isinstance(msg, bytes) else msg
            # Unknown type
            return str(msg)

        # If content is a "response object" with messages (HIL style), extract last assistant message
        extracted_text = ""
        if isinstance(content, dict) and "messages" in content:
            messages = content.get("messages", []) or []
            # Prefer last assistant/AI message; fall back to last item's content
            last_assistant = None
            for m in reversed(messages):
                # m might be dict or BaseMessage
                role = None
                if isinstance(m, dict):
                    role = m.get("role")
                elif isinstance(m, BaseMessage):
                    # attempt to detect assistant by class or metadata
                    role = getattr(m, "role", None)
                if role == "assistant" or isinstance(m, AIMessage):
                    last_assistant = m
                    break
            target_msg = last_assistant or (messages[-1] if messages else None)
            extracted_text = _extract_text_from_message(target_msg)

        # If content is a list of messages, extract last assistant or last element
        elif isinstance(content, list):
            last_assistant = None
            for m in reversed(content):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    last_assistant = m
                    break
                if isinstance(m, BaseMessage) and isinstance(m, AIMessage):
                    last_assistant = m
                    break
            target_msg = last_assistant or (content[-1] if content else None)
            extracted_text = _extract_text_from_message(target_msg)

        # If content itself is a BaseMessage / AIMessage
        elif isinstance(content, BaseMessage) or isinstance(content, AIMessage):
            extracted_text = _extract_text_from_message(content)

        # If content is plain string
        elif isinstance(content, (str, bytes)):
            extracted_text = content.decode("utf-8") if isinstance(content, bytes) else content

        # Fallback: stringify
        else:
            extracted_text = str(content or "")

    except Exception as e:
        logging.error("Error normalizing content for display: %s", str(e), exc_info=True)
        extracted_text = ""

    # Store last content & message time in session state for this tab
    try:
        st.session_state[f"last_content_{tab_key}"] = extracted_text
        st.session_state["last_message_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        # don't break UI on failure to set session state
        logging.exception("Failed to update last_content or last_message_time in session_state")

    # Compute and persist final elapsed time for this tab (so rerenders don't change it)
    start_time_key = f"start_time_{tab_key}"
    final_elapsed_key = f"final_elapsed_{tab_key}"
    elapsed = None
    if final_elapsed_key in st.session_state:
        elapsed = st.session_state[final_elapsed_key]
    else:
        start_time = st.session_state.get(start_time_key)
        if start_time is not None:
            try:
                elapsed = time.time() - start_time
                st.session_state[final_elapsed_key] = elapsed
            except Exception:
                logging.exception("Error computing elapsed time")

    st.markdown("### 📋 Research Results")
    if elapsed is not None:
        st.markdown(f'<div class="time-indicator">⏱️ Analysis completed in {elapsed:.2f} seconds</div>',
                    unsafe_allow_html=True)

    # FIRST: Try to detect and render any graphs dynamically
    graphs_rendered = extract_and_render_graphs(extracted_text)

    # If graphs were rendered, show the raw text in an expander
    if graphs_rendered:
        with st.expander("📝 View Detailed Analysis Text"):
            # Render the text content normally
            render_text_content(extracted_text)
    else:
        # If no graphs detected, render text normally
        render_text_content(extracted_text)

    # Keep PDF download capability (this is different from graph rendering)
    try:
        render_pdf_download(extracted_text, key=f"pdf_{tab_key}", user_query=user_query)
    except Exception as e:
        logging.error("PDF rendering error: %s", str(e), exc_info=True)
        st.error(f"Failed to generate PDF download: {str(e)}")


def render_text_content(text: str):
    """Helper function to render text content with proper formatting"""

    # Detect whether content is table-like to choose rendering approach
    def is_table_content(text):
        if not text:
            return False
        table_patterns = [
            r"^#\s", r"\t", r"\|\s*[^|]+\s*\|",
            r"<table>|<tr>|<td>|<th>",
            r"Article & Source", r"Key Points.*Sentiment",
            r"^\s*\d+\s*\|",  # Numbered pipe rows
        ]
        return any(re.search(pat, text, re.IGNORECASE | re.MULTILINE) for pat in table_patterns)

    try:
        if is_table_content(text):
            # Render table or table-like text directly
            st.markdown(text, unsafe_allow_html=True)
        else:
            # Normalize latex and wrap inline math when needed
            body = normalize_latex(text)
            if not is_latex(body):
                body = auto_wrap_latex_math(body)
            if is_latex(body):
                body = convert_latex_display_math(body)

            # Render within the mathjax block
            st.markdown(f"<div class='mathjax-block'>{body}</div>", unsafe_allow_html=True)
    except Exception as e:
        logging.error("Unexpected rendering error: %s", str(e), exc_info=True)
        st.error("An error occurred rendering the analysis results.")


navbar = NavbarUI()


@require_auth
def main_app():
    """Main application function"""
    # Add this somewhere in your main app for debugging
    # if st.session_state.get("authenticated", False) and st.session_state.get("session_id"):
    #     # Refresh session on any user activity
    #     SessionManager.refresh_session(st.session_state.session_id)

    # for time taken this CSS is working

    st.markdown("""
    <style>
    /* GLOBAL CSS - APPLIES TO ALL TABS */

    /* Updated Start AI Analysis button to match Try Free Search Now style */
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stButton"] > button[kind="primary"] {
    background: #1f77b4 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    }
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: #1663a6 !important; /* Slightly darker for hover */
    }


    /* Center the button container */
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stButton"] > button[kind="primary"]) {
        display: flex !important;
        justify-content: center !important;
        margin: 30px 0 !important;
    }

    /* Analysis buttons */
    .main .block-container .analysis-btn-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin: 20px 0;
    }
    .main .block-container .analysis-btn {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef) !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        padding: 15px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    .main .block-container .analysis-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
        border-color: #2e6bb7 !important;
    }

    # /* FIXED RESULT CARD STYLES  SYTREMLIT SUPPORTED, CAN TRY*/
    # .result-card {
    #     background: linear-gradient(145deg, #ffffff, #f8f9fa) !important;
    #     border: 1px solid #e9ecef !important;
    #     border-radius: 12px !important;
    #     padding: 24px !important;
    #     margin: 20px 0 !important;
    #     box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    # }

    # /* RESULT CARD - THIS IS WHAT TAB2 NEEDS */
    # .main .block-container .result-card {
    #     background: linear-gradient(145deg, #ffffff, #f8f9fa);
    #     border: 1px solid #e9ecef;
    #     border-radius: 12px;
    #     padding: 24px;
    #     margin: 20px 0;
    #     box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    # }

    div[data-testid="stMarkdownContainer"] .result-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa) !important;
        border: 1px solid #e9ecef !important;
        border-radius: 12px !important;
        padding: 24px !important;
        margin: 20px 0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    }

    /* MathJax block inside result card */
    div[data-testid="stMarkdownContainer"] .result-card .mathjax-block {
        /* Add any specific styles for math content here */
        font-size: 1.1rem;
        line-height: 1.6;
    }

    .section-header {
        color: #2e6bb7;
        font-weight: 600;
        font-size: 1.2rem;
        margin: 20px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #e9ecef;
    }

    .time-indicator {
        background: #e8f4fd;
        color: #0c5460;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.9rem;
        margin: 10px 0;
        border-left: 4px solid #2e6bb7;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        /* Fix sidebar scrollbar visibility */
        [data-testid="stSidebar"] {
            overflow: auto;
            height: 100vh;
        }

        [data-testid="stSidebar"]::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        [data-testid="stSidebar"]::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }

        [data-testid="stSidebar"]::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 10px;
        }

        [data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
            background: #555;
        }

        /* Ensure sidebar content doesn't overflow */
        [data-testid="stSidebar"] > div:first-child {
            height: 100%;
            overflow-y: auto;
        }

        /* Prevent sidebar from being pushed down */
        .stSidebar {
            position: sticky !important;
            top: 0 !important;
        }
        </style>
        """, unsafe_allow_html=True)

    # st.markdown("""
    #     <style>
    #         [data-testid="stToolbar"] {display: none !important;}
    #     </style>
    #     """, unsafe_allow_html=True)

    # Initialize navbar , below 2 lines for logout funcrtionality and navbar

    navbar.render_navbar()

    # Check if we need to preserve tab state after rerun (ADD THIS)
    if "preserve_tab" in st.session_state:
        st.session_state.current_tab = st.session_state.preserve_tab
        del st.session_state.preserve_tab

    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "tab1"

    if st.session_state.get("show_overlay_during_hil", False):
        show_spinner_overlay()
        # Process the HIL decision asynchronously
        asyncio.run(process_hil_decision())
        st.rerun()

    if render_human_review_ui():
        return
    # set_custom_theme()
    create_header()

    with st.sidebar:
        st.markdown("---")
        st.markdown("##### RAG Settings")

        # CSS for the colored status indicator
        st.markdown("""
        <style>
            .status-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 8px;
                vertical-align: middle;
            }
            .status-enabled {
                background-color: #28a745; /* Green */
            }
            .status-disabled {
                background-color: #6c757d; /* Grey */
            }
            .st-emotion-cache-1vze2hr { /* Targets the toggle's label */
                font-size: 0.95rem;
            }
        </style>
        """, unsafe_allow_html=True)

        # Determine status and display the indicator
        if st.session_state.rag_enabled:
            status_class = "status-enabled"
            status_text = "Enabled"
        else:
            status_class = "status-disabled"
            status_text = "Disabled"

        st.markdown(
            f'<div><span class="status-indicator {status_class}"></span>Research Context: <b>{status_text}</b></div>',
            unsafe_allow_html=True
        )
        # Disable toggle if agent is working
        # deactivate_rag = bool(st.session_state.get("spinner_active", False))
        # Disable toggle if agent is working in any tab
        deactivate_rag = bool(
            st.session_state.get("spinner_tab1", False) or
            st.session_state.get("spinner_tab2", False)
        )

        # Toggle to change the RAG setting
        new_setting = st.toggle(
            "Toggle Research Context",
            value=st.session_state.rag_enabled,
            help="When enabled, the agent uses documents from the research database to provide the context to LLM.",
            key="rag_toggle",
            disabled=deactivate_rag
        )

        # Update session state and show a toast notification on change
        if new_setting != st.session_state.rag_enabled:
            st.session_state.rag_enabled = new_setting
            # Rerun to ensure the UI updates immediately
            st.rerun()

    if "tab1_messages" not in st.session_state:
        st.session_state["tab1_messages"] = []

    if "agent_thinking" in st.session_state:
        display_agent_thinking(st.session_state["agent_thinking"])

    # # Add a clear button in your sidebar or near the thinking display
    if st.sidebar.button("🧹 Clear Thinking History"):
        st.session_state["agent_thinking"] = ""
        st.rerun()

    # if st.session_state.get("agent_thinking"):
    #     st.markdown("---")
    #     st.markdown("##### Debug Tools")
    #     if st.button("🧹 Clear Thinking History"):
    #         st.session_state["agent_thinking"] = ""
    #         st.rerun()

    st.markdown("""
    <style>
    .mathjax-block {
        background-color: #f8f9fa; /* Light grey background */
        border-left: 4px solid #4cafef; /* Accent color for math block */
        padding: 10px 15px;
        margin: 12px 0;
        border-radius: 8px;
        font-size: 1.05rem; /* Slightly larger for readability */
        overflow-x: auto; /* Prevent overflow issues */
    }

    /* Make inline math match text flow nicely */
    .mathjax-block mjx-container {
        font-family: 'Cambria Math', 'STIX Two Math', 'Times New Roman', serif;
        font-size: 1.05rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Ensure spinner and pending prompt keys exist for both tabs

    # Create tabs for different interaction modes
    # tab1, tab2 = st.tabs(["🧠 DeepAgent Navigator", "⚡ Express Analysis"])
    # tab1, tab2 = st.tabs(["🤖 AI Research Assistant", "⚡ Express Analysis"])
    #tabs = st.tabs(["🤖 AI Research Assistant", "⚡ Express Analysis"])
    #tab1, tab2 = tabs[0], tabs[1]
    # Set active tab based on session state
    # Create tabs and get the active tab from Streamlit
    tab1, tab2 = st.tabs(["🧠 DeepAgent Research", "⚡ Express Analysis"])

    # Use Streamlit's built-in tab state tracking
    if tab1:
        st.session_state.current_tab = "tab1"
    if tab2:
        st.session_state.current_tab = "tab2"

    # In main(), after the submit button in tab1:
    with tab1:

        if LOCAL_MODEL_ENABLED:
            st.success(f"🟢 Local LLM active → {LOCAL_MODEL_NAME}")
        else:
            st.warning("⚠️ Cloud LLM (OpenAI) active")
        # 🔽 Depth control
        if DEEPAGENT_ENABLED:
            # 🔽 Only show this when DeepAgent is enabled
            depth_mode = st.selectbox(
                "DeepAgent reasoning depth",
                options=["Quick", "Normal", "Deep"],
                index=1,  # default "Normal"
                key="deepagent_depth",
                help="Quick = cheaper & faster, Deep = more thorough but more tokens."
            )
        else:
            # No DeepAgent – fall back to Express mode and hide the dropdown
            st.info(
                "DeepAgent mode is disabled with local running model"
            )
            # Set a sensible default so run_deepagent (if ever called) has something
            st.session_state.setdefault("deepagent_depth", "Quick")

        # Initialize session state for input if not exists
        if "tab1_input" not in st.session_state:
            st.session_state.tab1_input = ""
        # user_prompt = create_chat_interface()
        user_prompt = st.text_area(
            "💬 Enter your stock research query:",
            value=st.session_state.tab1_input,
            placeholder="Type your message here...\nTip: Add 'with chart data' to get interactive visualizations",
            key="tab1_text_input"
        )
        # Update session state if input changes
        if user_prompt != st.session_state.tab1_input:
            st.session_state.tab1_input = user_prompt

        # disable when empty or whitespace-only
        #start_disabled = not (user_prompt and user_prompt.strip())

        if st.button("🚀 Start AI Analysis", type="primary"):
            st.session_state["tab1_messages"].append({"role": "user", "content": user_prompt})
            st.session_state["action"] = "tab1"
            st.session_state["graph_origin_tab"] = "tab1"
            st.session_state["spinner_tab1"] = True
            st.session_state["pending_prompt_tab1"] = user_prompt
            st.session_state["start_time_tab1"] = time.time()
            # Clear input after submission
            #st.session_state.tab1_input = ""
            st.rerun()

        if st.session_state.get("spinner_tab1", False):
            show_spinner_overlay()
            if st.session_state.get("pending_prompt_tab1"):
                action = st.session_state.get("action")
                print(f"********************Processing tab1 with action: {action}")
                asyncio.run(process_query(st.session_state["pending_prompt_tab1"], action=action))
                st.session_state['last_message_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["spinner_tab1"] = False
                st.session_state["pending_prompt_tab1"] = None
                st.rerun()

        # Display results using reusable function
        if "last_content_tab1" in st.session_state and st.session_state["last_content_tab1"].strip():
            user_query = None
            # Try to find the corresponding user query
            tab1_messages = st.session_state.get("tab1_messages", [])
            for msg in reversed(tab1_messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_query = msg.get("content")
                    break

            display_analysis_results(
                st.session_state["last_content_tab1"],
                "tab1",
                user_query=user_query
            )

    with tab2:
        stock_rows = create_multi_stock_selection_ui()
        actions = create_analysis_buttons()
        prompts = format_analysis_prompts_for_rows(stock_rows, actions, ANALYSIS_PROMPTS)

        if prompts:
            action, prompt, row = prompts[0]
            user_prompt = prompt
            action_click = action
            st.session_state["spinner_tab2"] = True
            st.session_state["pending_prompt_tab2"] = user_prompt
            st.session_state["pending_action_tab2"] = action_click
            st.session_state["graph_origin_tab"] = "tab2"
            st.session_state["start_time_tab2"] = time.time()
            st.rerun()

        if st.session_state.get("spinner_tab2", False):
            show_spinner_overlay()
            if st.session_state.get("pending_prompt_tab2") and st.session_state.get("pending_action_tab2"):
                asyncio.run(process_query(st.session_state["pending_prompt_tab2"], action=None))
                st.session_state['last_message_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["spinner_tab2"] = False
                st.session_state["pending_prompt_tab2"] = None
                st.session_state["pending_action_tab2"] = None
                st.rerun()

        # Display results using reusable function
        if "last_content_tab2" in st.session_state and st.session_state["last_content_tab2"].strip():
            display_analysis_results(
                st.session_state["last_content_tab2"],
                "tab2",
                user_query=st.session_state.get("last_query_tab2", "")
            )



def main():
    main_app()


if __name__ == "__main__":
    main()

