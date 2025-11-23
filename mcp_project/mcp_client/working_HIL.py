# This ensures the project root (where utils lives) is on the Python path, so from utils.pdf_export import create_pdf will work when running with Streamlit.
import json
import sys
import os
from pprint import pprint

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END

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
    risk_analyst

# Add this right after your imports, before any function definitions

import json


# def set_tab_selection(tab_index):
#     """Use JavaScript to set the active tab by index (0 for tab1, 1 for tab2)"""
#     js = f"""
#     <script>
#     function setActiveTab() {{
#         // Wait for tabs to be fully rendered
#         const tabs = document.querySelectorAll('[data-testid="stTabButton"]');
#         if (tabs.length > {tab_index}) {{
#             tabs[{tab_index}].click();
#             return true;
#         }}
#         return false;
#     }}
#
#     // Try immediately, then retry with increasing delays
#     if (!setActiveTab()) {{
#         let attempts = 0;
#         const interval = setInterval(() => {{
#             if (setActiveTab() || attempts >= 10) {{
#                 clearInterval(interval);
#             }}
#             attempts++;
#         }}, 100);
#     }}
#     </script>
#     """
#     st.components.v1.html(js, height=0)


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
    "LANGCHAIN_PROJECT": f"Stock Analysis ReAct Agent- {unique_id}",
    "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
})

# Initialize the chat model
# model = init_chat_model("gpt-4o")

model = init_chat_model("o4-mini", model_provider="openai")
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
server_params = {
    # below 2 block is for stdio local  and deployed  running
    # "stocksAnalysisMCPServer": {
    #     "command": "python",
    #     "args": [stock_server_path],
    #     "transport": "stdio",
    # },
    # "WebCrawlerMCP": {
    #     "command": "python",
    #     "args": [webcrawler_path],
    #     "transport": "stdio",
    # },
    # for local running when MCP is running in another paycharm
    "stocksMCPServerHTTPYFinance": {
        "url": "http://localhost:8001/mcp/",
        "transport": "streamable_http",
    },

    # for deployed MCP server running in AWS
    # "stocksMCPServerHTTPYFinance": {
    #     "url": "http://stock-mcp-http-internal.gen-ai-mcp-services:8001/mcp/",
    #     "transport": "streamable_http",
    # }
    # "AlphaVantageMCPServer": {
    #     "command": "python",
    #     "args": [os.path.join(mcp_server_dir, 'mcp_stock_server_alpha.py')],
    #     "transport": "stdio",
    # }
}

# Add some debugging to verify paths
# print(f"Stock server path: {stock_server_path}")
# print(f"Webcrawler path: {webcrawler_path}")
# print(f"Path exists - Stock server: {os.path.exists(stock_server_path)}")
# print(f"Path exists - Webcrawler: {os.path.exists(webcrawler_path)}")
# Add this near your other environment variables
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from utils.rag_db import get_context_from_chroma
from typing import TypedDict, List, Optional, Literal, Annotated
import operator


class CustomState(TypedDict):
    # Multiple nodes can contribute messages in the same step (LLM, tools) → annotate with add
    messages: Annotated[List[BaseMessage], operator.add]
    # Treat these as single-writer keys; do NOT annotate. Only the node that changes them should return them.
    rag_enabled: bool
    approval_status: Optional[Literal["pending", "approved", "rejected"]]
    review_data: Optional[dict]
    # If you increment this in multiple nodes in the same step, annotate with add; else leave as plain int.
    iteration_count: Annotated[int, operator.add]


def rag_context_node(state: CustomState):
    """Node to augment user queries with RAG context"""
    print("*********rag_context_node********RAG Enabled from state:", state["rag_enabled"])
    if not state["rag_enabled"]:
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
            except Exception as e:
                logging.error("RAG context error: %s", str(e), exc_info=True)

    # Return original state if no augmentation was performed
    return state


async def call_model(state: CustomState, tools):
    """Invokes the model asynchronously with the current state and tools."""
    try:
        # Use model.ainvoke for asynchronous execution in an async graph
        response = await model.bind_tools(tools).ainvoke(state["messages"])

        # The response is already a LangChain AIMessage.
        # Append it to the state and preserve other state variables.
        return {
            "messages": state["messages"] + [response],
            "rag_enabled": state["rag_enabled"]
        }

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


import functools

# tab-2 with RAG and tools

# python
import asyncio
import functools
import logging
from langchain_core.messages import ToolMessage


async def custom_tool_node(state: CustomState, tools):
    """
    Invoke tools for each tool_call on the last assistant message and
    return ToolMessage(s) with matching tool_call_id fields so the LLM's
    tool_call handshake is satisfied.
    """
    msgs = state.get("messages") or []
    if not msgs:
        return {}

    last = msgs[-1]
    tool_calls = getattr(last, "tool_calls", None) or getattr(last, "tool_call", None) or []
    if not tool_calls:
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

    # Return only the new tool messages so the graph appends them after the assistant message.
    return {"messages": tool_response_msgs}


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
    msgs = state.get("messages") or []
    last_message = msgs[-1] if msgs else None
    content = getattr(last_message, "content", "") if last_message else ""
    display_iteration = int(state.get("iteration_count", 0)) + 1
    return {
        "review_data": {
            "response_content": content,
            "iteration": display_iteration,
        },
        "approval_status": "pending",
    }


def rejected_node(state: CustomState):
    """
    Increment iteration and clear approval status. Return only the changed keys.
    Logs the previous and updated iteration count.
    """
    old = int(state.get("iteration_count", 0))
    new = old + 1

    # Log to the configured logger
    logging.info("rejected_node: iteration_count %d -> %d", old, new)

    # Optional: also show in Streamlit UI (uncomment if desired)
    # try:
    #     import streamlit as st
    #     st.info(f"Iteration updated: {old} -> {new}")
    # except Exception:
    #     pass

    return {
        "iteration_count": new,
        "approval_status": None,
    }

# def rejected_node(state: CustomState):
#     """
#     Increment iteration and clear approval status. Return only the changed keys.
#     """
#     return {
#         "iteration_count": int(state.get("iteration_count", 0)) + 1,
#
#         "approval_status": None,
#     }


def approved_node(state: CustomState):
    """
    No-op terminal node. Return nothing.
    """
    return {}


# async def run_agent(prompt, rag_enabled=None):
#     # Use session state if not explicitly provided
#     if rag_enabled is None:
#         rag_enabled = st.session_state.get('rag_enabled', RAG_ENABLED)
#
#     # Initialize client and get tools
#     client = MultiServerMCPClient(server_params)
#     tools = await client.get_tools()
#
#     # Build the graph
#     builder = StateGraph(CustomState)
#     print("*********run_agent********RAG Enabled:", rag_enabled)
#
#     # Define the entry point based on whether RAG is enabled
#     entry_point = "rag_context" if rag_enabled else "LLM_Call_with_Tool"
#
#     # Add nodes
#     if rag_enabled:
#         builder.add_node("rag_context", rag_context_node)
#
#     model_invoker = functools.partial(call_model, tools=tools)
#     builder.add_node("LLM_Call_with_Tool", model_invoker)
#
#     # *** FIX: Use the new custom_tool_node with functools.partial ***
#     tool_invoker = functools.partial(custom_tool_node, tools=tools)
#     builder.add_node("tools", tool_invoker)
#
#     # Define graph edges
#     builder.add_edge(START, entry_point)
#     if rag_enabled:
#         builder.add_edge("rag_context", "LLM_Call_with_Tool")
#
#     builder.add_conditional_edges("LLM_Call_with_Tool", tools_condition)
#     builder.add_edge("tools", "LLM_Call_with_Tool")
#
#     # Compile and run
#     memory = MemorySaver()
#     app = builder.compile(checkpointer=memory)
#
#     config = {"configurable": {"thread_id": st.session_state.get("username")}}
#     initial_state = {
#         "messages": [HumanMessage(content=prompt)],
#         "rag_enabled": rag_enabled
#     }
#
#     # Clear previous state for this user to ensure a clean run
#     app.get_state(config) # Load the state
#     app.update_state(config, None) # Reset it
#
#     agent_response = await app.ainvoke(
#         initial_state,
#         {"recursion_limit": 20, **config}
#     )
#
#     return agent_response


# python


# python
async def run_agent(prompt, rag_enabled=None):
    """
    Run the agent graph for a single prompt with HIL support.
    - Adds optional RAG node.
    - Uses model invoker and custom tool node.
    - Adds HIL nodes (human_review, rejected, approved).
    - Wires conditional edges so LLM -> (tools | human_review),
      human_review -> (approved | rejected),
      rejected -> LLM (until iteration limit) -> approved.
    - Compiles with interrupt_after=['human_review'] so UI can handle approval.
    """
    # Resolve RAG toggle
    if rag_enabled is None:
        rag_enabled = st.session_state.get("rag_enabled", RAG_ENABLED)

    # Initialize MCP client and tools
    client = MultiServerMCPClient(server_params)
    tools = await client.get_tools()

    # Build graph
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

    # Edges and routing
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

    # Approved is terminal
    builder.add_edge("approved", END)
    # MAX_USER_REVISIONS = 3
    #Initial run: iteration_count = 0 → goes to human review
    #First rejection: iteration_count increments to 1 → 1 < 3 → true → goes back to LLM → then to human review again
    #Second rejection: iteration_count increments to 2 → 2 < 3 → true → goes back to LLM → then to human review again
    #Third rejection: iteration_count increments to 3 → 3 < 3 → false → goes to approved (no more reviews)
    MAX_USER_REVISIONS = 3
    # Rejected loops back to LLM until iteration limit, otherwise go to approved
    builder.add_conditional_edges(
        "rejected",
        lambda state: "LLM_Call_with_Tool" if state.get("iteration_count", 0) < MAX_USER_REVISIONS else "approved",
    )

    # Checkpointer and compile — interrupt AFTER human_review so UI can take over
    memory = MemorySaver()
    app = builder.compile(checkpointer=memory, interrupt_after=["human_review"])

    # Persist app/config for UI resume
    thread_id = st.session_state.get("username") or st.session_state.get("unique_id") or "anon"
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
    }

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


DEEPAGENT_ENABLED = os.getenv("DEEPAGENT_ENABLED", "false").lower() == "true"

async def run_deepagent(messages, rag_enabled=None):
    logging.debug("^^^^^^^^^^^^^^^^^^^^Starting run_deepagent:^^^^^^^^^^^^^^^^^^^^^^")

    # Use provided rag_enabled, or fallback to environment variable
    # Use session state if not explicitly provided
    if rag_enabled is None:
        rag_enabled = st.session_state.get('rag_enabled', RAG_ENABLED)

    from deepagents import create_deep_agent
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from mcp_client.agents.user_checkpointer import UserCheckpointer

    # Initialize client and get tools
    client = MultiServerMCPClient(server_params)
    tools = await client.get_tools()

    #logging.debug("************tools from mcp client *****************: %s", tools)
    subagents = [fundamental_analyst, technical_analyst, risk_analyst]
    #logging.debug("************subagents *****************: %s", subagents)

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
        # RAG context retrieval - only if enabled
        # rag_context = ""
        # if rag_enabled:
        #     try:
        #         rag_context = get_context_from_chroma(messages)
        #         if rag_context and rag_context.strip():
        #             logging.debug("RAG Context retrieved for deep agent: %s",
        #                           rag_context[:200] + "..." if len(rag_context) > 200 else rag_context)
        #         else:
        #             logging.debug("No RAG context found for deep agent query: %s", messages)
        #     except Exception as e:
        #         logging.error("RAG context error in deep agent: %s", str(e), exc_info=True)
        #         # Continue without RAG context - don't break the flow
        #
        # # Augment the prompt with RAG context if available and enabled
        # augmented_prompt = messages
        # if rag_enabled and rag_context and rag_context.strip():
        #     augmented_prompt = f"## Relevant Research Context:\n{rag_context}\n\n## User Query:\n{messages}"
        #
        # # Invoke the agent with the (possibly augmented) prompt
        # if rag_enabled:
        #
        #     response = await agent.ainvoke(
        #         {"messages": [{"role": "user", "content": augmented_prompt}]},
        #         {"recursion_limit": 30}
        #     )
        # else:
        #     response = await agent.ainvoke(
        #         {"messages": messages},
        #         {"recursion_limit": 30}
        #     )

        # Python
        if rag_enabled:
            # Extract last user message content for RAG context
            last_user_msg_content = None
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    last_user_msg_content = msg.get("content")
                    break
            if last_user_msg_content:
                rag_context = get_context_from_chroma(last_user_msg_content)
                if rag_context and rag_context.strip():
                    augmented_prompt = f"## Relevant Research Context:\n{rag_context}\n\n## User Query:\n{last_user_msg_content}"
                else:
                    augmented_prompt = last_user_msg_content
                    print("******************augmented_prompt to LLM**********",augmented_prompt)


                response = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": augmented_prompt}]},
                    {"recursion_limit": 30}
                )
            else:
                # Fallback if no user message found
                response = await agent.ainvoke(
                    {"messages": messages},
                    {"recursion_limit": 30}
                )
        else:
            response = await agent.ainvoke(
                {"messages": messages},
                {"recursion_limit": 30}
            )

        checkpointer.save(user_id, response)
        logging.debug("*******************LLM response: %s State: %s", user_id, response)

        state = checkpointer.load(user_id)
        #logging.debug("######################Checkpoint loaded for user: %s State: %s", user_id, state)

    except langgraph.errors.GraphRecursionError as e:
        # Return partial state if available
        logging.error("GraphRecursionError encountered: %s", str(e), exc_info=True)
        response = getattr(e, "partial_state", {"messages": []})
        checkpointer.save(user_id, response)
        logging.debug("********************Checkpoint saved for user after recursion error: %s State: %s", user_id,
                      response)

    except Exception as e:
        logging.error("Unexpected error in deep agent: %s", str(e), exc_info=True)
        # Return empty response to prevent breaking the UI
        response = {"messages": [AIMessage(content=f"Error processing request: {str(e)}")]}

    print("Agent response received.")
    #logging.debug("######################Checkpoint: Agent response: %s", response)
    return response





# added for new UI components

# python
async def process_query(prompt: str, action: str = None):
    """
    Process a user prompt. Chooses between run_agent and run_deepagent (if enabled),
    preserves HIL approval state, captures agent 'thinking' output, normalizes response
    shapes (dict/list/BaseMessage/string), stores tab-scoped last content/response/query,
    and retains only last assistant + current user in tab1_messages for context.
    """
    spinner_key = f"spinner_{action or 'tab'}"
    st.session_state.setdefault("approval_status", None)
    st.session_state.setdefault("review_data", None)
    st.session_state[spinner_key] = True
    print("*****************************action:", action)
    print("*****************************DEEPAGENT_ENABLED:", DEEPAGENT_ENABLED)

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

            # Find last assistant message (before the newly appended user)
            last_ai_msg = None
            for msg in reversed(tab1_messages[:-1]):
                if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                    last_ai_msg = msg
                    break
                # Accept BaseMessage-like objects
                if isinstance(msg, BaseMessage):
                    role = getattr(msg, "role", None) or getattr(msg, "type", None) or None
                    content = getattr(msg, "content", None)
                    if role == "assistant" and content:
                        last_ai_msg = {"role": "assistant", "content": content}
                        break
            print("************************last_ai_msg*****************: ", last_ai_msg)
            messages_for_deepagent = []
            if last_ai_msg:
                messages_for_deepagent.append(last_ai_msg)
            messages_for_deepagent.append({"role": "user", "content": prompt})
        print("************************messages_for_deepagent*****************: ", messages_for_deepagent)
        rag_enabled = st.session_state.get("rag_enabled", RAG_ENABLED)
        print("************************RAG Enabled*****************: ", rag_enabled)
        # Call appropriate agent
        if DEEPAGENT_ENABLED and action == "tab1":
            response = await run_deepagent(messages_for_deepagent,
                                           rag_enabled=st.session_state.get("rag_enabled", RAG_ENABLED))
        else:
            # For LLM/HIL flow we pass the raw prompt (run_agent handles RAG + human review)
            response = await run_agent(prompt, rag_enabled=st.session_state.get("rag_enabled", RAG_ENABLED))

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

        # Pretty-print/capture agent thinking
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

        st.session_state["agent_thinking"] = pretty_output.getvalue()
        if st.session_state.get("agent_thinking"):
            display_agent_thinking(st.session_state["agent_thinking"])

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

        # Maintain minimal context: keep last assistant + current user in tab1_messages
        try:
            tab1_messages = st.session_state.get("tab1_messages", []) or []
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@tab1_messages:", tab1_messages)
            # append assistant response if not already present
            assistant_content = None
            # find assistant content from normalized messages (prefer BaseMessage then dict)
            for m in reversed(messages):
                role = getattr(m, "role", None) if isinstance(m, BaseMessage) else (
                    m.get("role") if isinstance(m, dict) else None)
                content = getattr(m, "content", None) if isinstance(m, BaseMessage) else (
                    m.get("content") if isinstance(m, dict) else str(m))
                if role == "assistant" and content:
                    assistant_content = content
                    break
            if assistant_content:
                # add assistant as dict if last item is not the same assistant content
                if not (tab1_messages and isinstance(tab1_messages[-1], dict) and tab1_messages[-1].get(
                        "role") == "assistant" and tab1_messages[-1].get("content") == assistant_content):
                    tab1_messages.append({"role": "assistant", "content": assistant_content})
            # Ensure last element is the current user (may already exist)
            if not (tab1_messages and isinstance(tab1_messages[-1], dict) and tab1_messages[-1].get(
                    "role") == "user" and tab1_messages[-1].get("content") == prompt):
                tab1_messages.append({"role": "user", "content": prompt})
            # Trim to last assistant + last user
            # Find last assistant index
            last_assistant_index = None
            for i in range(len(tab1_messages) - 1, -1, -1):
                itm = tab1_messages[i]
                if (isinstance(itm, dict) and itm.get("role") == "assistant") or (
                        isinstance(itm, BaseMessage) and getattr(itm, "role", None) == "assistant"):
                    last_assistant_index = i
                    break
            if last_assistant_index is None:
                # keep only the last user
                trimmed = tab1_messages[-1:]
            else:
                trimmed = tab1_messages[last_assistant_index:]
            st.session_state["tab1_messages"] = trimmed
        except Exception:
            logging.exception("Error updating tab1_messages context; continuing gracefully.")

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


# async def process_query(prompt: str, action: str = None):
#     """Process a query using the agent"""
#     with st.spinner("🤖 Processing..."):
#         rag_enabled = st.session_state.get('rag_enabled', RAG_ENABLED)
#         print("*********process_query*********RAG Enabled:", rag_enabled)
#         print("*****************************action:", action)
#         print("*****************************DEEPAGENT_ENABLED:", DEEPAGENT_ENABLED)
#
#         try:
#             if DEEPAGENT_ENABLED and action == "tab1":
#                 logging.debug("************deepagent  called from process_query*****************")
#                 #===================start of logic  to extract 2 set of last messages for context========================
#                 # Python
#                 # tab1_messages = st.session_state.get("tab1_messages", [])
#                 # pairs = []
#                 # i = len(tab1_messages) - 1
#                 # while i > 0 and len(pairs) < 2:
#                 #     user_msg = tab1_messages[i - 1] if i - 1 >= 0 and tab1_messages[i - 1].get(
#                 #         "role") == "user" else None
#                 #     ai_msg = tab1_messages[i] if tab1_messages[i].get("role") == "assistant" else None
#                 #     if user_msg and ai_msg:
#                 #         pairs.insert(0, user_msg)
#                 #         pairs.insert(1, ai_msg)
#                 #         i -= 2
#                 #     else:
#                 #         i -= 1
#                 # messages = pairs + [{"role": "user", "content": prompt}]
#                 #===================end of logic  to extract 2 set of last messages for context========================
#                 # Only keep last user and last AI message for context
#                 tab1_messages = st.session_state.get("tab1_messages", [])
#                 print("@@@@@@@@@@@@@@@@@@@@@@@@@@@tab1_messages:", tab1_messages)
#                 # Get last user message
#                 last_user_msg = {"role": "user", "content": prompt}
#                 print("@@@@@@@@@@@@@@@@@@@@@@@@@@@last_user_msg:", last_user_msg)
#                 # Get last AI message if exists
#                 last_ai_msg = None
#                 for msg in reversed(tab1_messages):
#                     if isinstance(msg, AIMessage) or (isinstance(msg, dict) and msg.get("role") == "assistant"):
#                         last_ai_msg = msg
#                         break
#                 print("@@@@@@@@@@@@@@@@@@@@@@@@@@@last_ai_msg:", last_ai_msg)
#                 # Build messages for context (last AI + current user)
#                 messages = [last_ai_msg, last_user_msg] if last_ai_msg else [last_user_msg]
#                 print("@@@@@@@@@@@@@@@@@@@@@@@@@@@messages to deepagent:", messages)
#
#                 response = await run_deepagent(messages, rag_enabled=rag_enabled)
#                 print("@@@@@@@@@@@@@@@@@@@@@@@@@@@response from deepagent:", response)
#                 st.session_state["tab1_messages"] = [last_user_msg, response.get("messages", [])[-1]]
#                 st.session_state["last_content_tab1"] = response.get("content", "")
#             else:
#                 #this is for tab2 without deepagent and no conversational memory
#                 response = await run_agent(prompt, rag_enabled=rag_enabled)
#
#                 #st.session_state["tab1_messages"].append(response)
#                 #st.session_state["last_content_tab1"] = response.get("content", "")
#
#         except Exception as e:
#             error_msg = str(e)
#             logging.error(f"Agent/model error: {error_msg}", exc_info=True)
#             if "context_length_exceeded" in error_msg or "maximum context length" in error_msg:
#                 st.warning(
#                     "⚠️ The request was too large for the AI model to process. "
#                     "This may be due to a long conversation or complex analysis. "
#                     "Please try again, or report this issue if it persists."
#                 )
#             else:
#                 st.error(f"An error occurred: {error_msg}")
#             return  # Optionally stop further processing
#
#         if action == "tab1":
#             st.session_state["last_content_tab1"] = response
#         else:
#             st.session_state["last_content_tab2"] = response
#
#         #response = await run_agent(prompt)
#         messages = response.get("messages", [])
#         if messages:
#             pretty_output = io.StringIO()
#             with contextlib.redirect_stdout(pretty_output):
#                 for m in messages:
#                     if hasattr(m, "pretty_print"):
#                         m.pretty_print()
#                     else:
#                         print(m)
#             st.session_state["agent_thinking"] = pretty_output.getvalue()
#             if "agent_thinking" in st.session_state:
#                 display_agent_thinking(st.session_state["agent_thinking"])
#             # --- Only show last AI message in main UI ---
#             last_msg = messages[-1]
#             content = getattr(last_msg, "content", None)
#             if content:
#                 if action == "tab1":
#                     st.session_state["last_content_tab1"] = content
#                 else:
#                     st.session_state["last_content_tab2"] = content


import re


# python
# python
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


# async def process_hil_decision():
#     """Process the human review decision with overlay showing"""
#     approval_status = st.session_state.get("hil_processing_status")
#     origin = st.session_state.get("graph_origin_tab", "tab1")
#
#     # === PRESERVE USER QUERY BEFORE RESUMING ===
#     user_query = st.session_state.get(f"last_query_{origin}")
#
#     # If we don't have the query stored, try to find it in tab1 messages
#     if not user_query and origin == "tab1":
#         tab1_messages = st.session_state.get("tab1_messages", [])
#         for msg in reversed(tab1_messages):
#             if isinstance(msg, dict) and msg.get("role") == "user":
#                 user_query = msg.get("content")
#                 break
#
#     # Store the query for later use
#     if user_query:
#         st.session_state[f"last_query_{origin}"] = user_query
#     # === END USER QUERY PRESERVATION ===
#
#     app = st.session_state.get("graph_app")
#     cfg = st.session_state.get("graph_config") or {}
#
#     if app and cfg:
#         try:
#             app.update_state(cfg, {"approval_status": approval_status})
#             resp = await app.ainvoke(None, {"recursion_limit": 20, **cfg})
#
#             # Persist resumed response
#             st.session_state[f"last_response_{origin}"] = resp
#
#             # Extract content from response
#             extracted_text = ""
#             if isinstance(resp, dict) and "messages" in resp:
#                 messages = resp.get("messages", []) or []
#                 # Find last assistant message
#                 for m in reversed(messages):
#                     if isinstance(m, BaseMessage) and isinstance(m, AIMessage):
#                         extracted_text = getattr(m, "content", "") or ""
#                         break
#                     elif isinstance(m, dict) and m.get("role") == "assistant":
#                         extracted_text = m.get("content", "") or ""
#                         break
#                 if not extracted_text and messages:
#                     last_msg = messages[-1]
#                     if isinstance(last_msg, BaseMessage):
#                         extracted_text = getattr(last_msg, "content", "") or ""
#                     elif isinstance(last_msg, dict):
#                         extracted_text = last_msg.get("content", "") or ""
#
#             # Store the extracted content for display
#             if extracted_text:
#                 st.session_state[f"last_content_{origin}"] = extracted_text
#
#             # For tab1: maintain chat history
#             if origin == "tab1":
#                 tab1_messages = st.session_state.get("tab1_messages", []) or []
#                 # Add user message if not already present
#                 user_query = st.session_state.get(f"last_query_{origin}") or st.session_state.get("last_query_tab1")
#                 if user_query and not any(
#                         isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content") == user_query
#                         for msg in tab1_messages
#                 ):
#                     tab1_messages.append({"role": "user", "content": user_query})
#
#                 # Add assistant response
#                 if extracted_text and not any(
#                         isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get(
#                             "content") == extracted_text
#                         for msg in tab1_messages
#                 ):
#                     tab1_messages.append({"role": "assistant", "content": extracted_text})
#
#                 st.session_state["tab1_messages"] = tab1_messages[-4:]  # Keep last 2 exchanges
#
#         except Exception as e:
#             logging.exception("Failed to resume graph on %s", approval_status)
#             st.error(f"Failed to process: {e}")
#             # Store error message
#             st.session_state[f"last_content_{origin}"] = f"Error: {str(e)}"
#
#     # Clear HIL UI state and return control to main UI
#     st.session_state["approval_status"] = None
#     st.session_state["review_data"] = None
#     st.session_state.pop("graph_origin_tab", None)
#     st.session_state["show_overlay_during_hil"] = False
#     st.session_state["hil_processing_status"] = None
#
#     # Preserve the current tab before rerun
#     st.session_state["preserve_tab"] = origin

import streamlit as st


async def process_hil_decision():
    """Process the human review decision with overlay showing"""
    approval_status = st.session_state.get("hil_processing_status")

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

        # Update only the approval_status in the current state
        updated_state = {
            **current_state,
            "approval_status": approval_status
        }

        # Update the state in the checkpointer
        app.update_state(cfg, updated_state)

        # Resume execution from the current state
        resp = await app.ainvoke(None, {"recursion_limit": 20, **cfg})

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

    # Preserve the current tab before rerun - THIS IS CRITICAL
    st.session_state["preserve_tab"] = origin

# def render_human_review_ui():
#     """Render the human review panel. Returns True if shown, else False."""
#     if st.session_state.get("approval_status") != "pending":
#         return False
#
#     review = st.session_state.get("review_data") or {}
#     content = review.get("response_content") or ""
#     iteration = int(review.get("iteration") or 1)
#
#     st.markdown("---")
#     st.markdown("### 👥 Human Review Required")
#     st.markdown(
#         f'<div class="time-indicator">📋 Iteration {iteration} - Please review the response</div>',
#         unsafe_allow_html=True,
#     )
#     st.markdown("#### Response Preview")
#     # Keep preview simple; the main UI will show final content after resume
#     st.markdown(f"<div class='result-card'><div class='mathjax-block'>{content}</div></div>", unsafe_allow_html=True)
#
#     # Scoped CSS to equalize button heights and vertically center
#     st.markdown(
#         """
#         <style>
#         .hil-button-col { display: flex !important; align-items: center !important; justify-content: center !important; height: 56px !important; margin: 0 !important; padding: 0 !important; }
#         .hil-button-col > div[data-testid="stButton"] { width: 100% !important; height: 100% !important; display: flex !important; align-items: center !important; justify-content: center !important; margin: 0 !important; }
#         .hil-button-col > div[data-testid="stButton"] > button { min-height: 40px !important; padding: 8px 18px !important; font-size: 0.95rem !important; border-radius: 8px !important; margin: 0 !important; }
#         .hil-gutter { width: 16px; }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )
#
#     # Determine originating tab for correct state update (default to tab1)
#     origin = st.session_state.get("graph_origin_tab", "tab1")
#
#     col_left, col_right = st.columns([1, 1])
#     # python
#     with col_left:
#         st.markdown('<div class="hil-button-col">', unsafe_allow_html=True)
#         if st.button("✅ Approve & Send", type="primary", use_container_width=True, key="approve_send"):
#             app = st.session_state.get("graph_app")
#             cfg = st.session_state.get("graph_config") or {}
#             origin = st.session_state.get("graph_origin_tab", "tab1")
#
#             if app and cfg:
#                 try:
#                     # Mark approval and resume the graph so approved_node runs
#                     app.update_state(cfg, {"approval_status": "approved"})
#                     resp = asyncio.run(app.ainvoke(None, {"recursion_limit": 20, **cfg}))
#
#                     # Persist resumed response into originating tab keys
#                     st.session_state[f"last_response_{origin}"] = resp
#
#                     # Normalize messages from resp into a list for scanning
#                     msgs = []
#                     if isinstance(resp, dict) and "messages" in resp:
#                         msgs = resp.get("messages") or []
#                     elif hasattr(resp, "messages"):
#                         try:
#                             msgs = getattr(resp, "messages") or []
#                         except Exception:
#                             msgs = [resp]
#                     elif isinstance(resp, list):
#                         msgs = resp
#                     else:
#                         msgs = [resp]
#
#                     # Extract last assistant content and last user content
#                     last_content = None
#                     last_user_content = None
#                     for m in reversed(msgs):
#                         if isinstance(m, AIMessage):
#                             if last_content is None:
#                                 last_content = getattr(m, "content", None)
#                             # AIMessage may include metadata pointing to the user - continue
#                         elif isinstance(m, dict):
#                             role = m.get("role")
#                             content = m.get("content")
#                             if role == "assistant" and content and last_content is None:
#                                 last_content = content
#                             if role == "user" and content and last_user_content is None:
#                                 last_user_content = content
#                         elif isinstance(m, (str, bytes)) and last_content is None:
#                             last_content = str(m)
#                         # Stop early if both found
#                         if last_content and last_user_content:
#                             break
#
#                     # Fallback to session values if user content not present in resumed messages
#                     if not last_user_content:
#                         last_user_content = st.session_state.get(f"last_query_{origin}") or st.session_state.get(
#                             "last_query_tab1")
#
#                     if last_content:
#                         st.session_state[f"last_content_{origin}"] = last_content
#
#                     if last_user_content:
#                         st.session_state[f"last_query_{origin}"] = last_user_content
#
#                         # If originating tab is tab1, maintain minimal conversational context
#                         if origin == "tab1":
#                             try:
#                                 tab1_messages = st.session_state.get("tab1_messages", []) or []
#                                 # Append assistant if present and not duplicate
#                                 if last_content and not (
#                                         tab1_messages and isinstance(tab1_messages[-1], dict) and tab1_messages[-1].get(
#                                         "role") == "assistant" and tab1_messages[-1].get("content") == last_content):
#                                     tab1_messages.append({"role": "assistant", "content": last_content})
#                                 # Append user if present and not duplicate
#                                 if last_user_content and not (
#                                         tab1_messages and isinstance(tab1_messages[-1], dict) and tab1_messages[-1].get(
#                                         "role") == "user" and tab1_messages[-1].get("content") == last_user_content):
#                                     tab1_messages.append({"role": "user", "content": last_user_content})
#                                 # Trim to last assistant + last user
#                                 last_assistant_index = None
#                                 for i in range(len(tab1_messages) - 1, -1, -1):
#                                     itm = tab1_messages[i]
#                                     if isinstance(itm, dict) and itm.get("role") == "assistant":
#                                         last_assistant_index = i
#                                         break
#                                 if last_assistant_index is None:
#                                     trimmed = tab1_messages[-1:]
#                                 else:
#                                     trimmed = tab1_messages[last_assistant_index:]
#                                 st.session_state["tab1_messages"] = trimmed
#                             except Exception:
#                                 logging.exception("Error updating tab1_messages after approve; continuing.")
#
#                 except Exception as e:
#                     logging.exception("Failed to resume graph on approve")
#                     st.error(f"Failed to resume agent: {e}")
#
#             # Clear HIL UI state and origin so the main tab can show result
#             st.session_state["approval_status"] = None
#             st.session_state["review_data"] = None
#             st.session_state.pop("graph_origin_tab", None)
#             st.rerun()
#         st.markdown('</div>', unsafe_allow_html=True)
#
#     with col_right:
#         st.markdown('<div class="hil-button-col">', unsafe_allow_html=True)
#         if st.button("🔄 Request Revision", type="secondary", use_container_width=True, key="request_revision"):
#             app = st.session_state.get("graph_app")
#             cfg = st.session_state.get("graph_config") or {}
#             origin = st.session_state.get("graph_origin_tab", "tab1")
#
#             if app and cfg:
#                 try:
#                     # Mark the checkpoint as rejected and resume the graph so rejected_node runs
#                     app.update_state(cfg, {"approval_status": "rejected"})
#                     resp = asyncio.run(app.ainvoke(None, {"recursion_limit": 20, **cfg}))
#
#                     # Persist full resumed response
#                     st.session_state[f"last_response_{origin}"] = resp
#
#                     # Normalize messages
#                     msgs = []
#                     if isinstance(resp, dict) and "messages" in resp:
#                         msgs = resp.get("messages") or []
#                     elif hasattr(resp, "messages"):
#                         try:
#                             msgs = getattr(resp, "messages") or []
#                         except Exception:
#                             msgs = [resp]
#                     elif isinstance(resp, list):
#                         msgs = resp
#                     else:
#                         msgs = [resp]
#
#                     # Extract last assistant content and last user content
#                     last_content = None
#                     last_user_content = None
#                     for m in reversed(msgs):
#                         if isinstance(m, AIMessage):
#                             if last_content is None:
#                                 last_content = getattr(m, "content", None)
#                         elif isinstance(m, dict):
#                             role = m.get("role")
#                             content = m.get("content")
#                             if role == "assistant" and content and last_content is None:
#                                 last_content = content
#                             if role == "user" and content and last_user_content is None:
#                                 last_user_content = content
#                         elif isinstance(m, (str, bytes)) and last_content is None:
#                             last_content = str(m)
#                         if last_content and last_user_content:
#                             break
#
#                     # Fallback: if no user found in resumed messages, preserve existing session value
#                     if not last_user_content:
#                         last_user_content = st.session_state.get(f"last_query_{origin}") or st.session_state.get(
#                             "last_query_tab1")
#
#                     if last_content:
#                         st.session_state[f"last_content_{origin}"] = last_content
#
#                     if last_user_content:
#                         st.session_state[f"last_query_{origin}"] = last_user_content
#
#                         # If this is tab1, maintain minimal conversational context (last assistant + last user)
#                         if origin == "tab1":
#                             try:
#                                 tab1_messages = st.session_state.get("tab1_messages", []) or []
#                                 if last_content and not (
#                                         tab1_messages and isinstance(tab1_messages[-1], dict) and tab1_messages[-1].get(
#                                         "role") == "assistant" and tab1_messages[-1].get("content") == last_content):
#                                     tab1_messages.append({"role": "assistant", "content": last_content})
#                                 if last_user_content and not (
#                                         tab1_messages and isinstance(tab1_messages[-1], dict) and tab1_messages[-1].get(
#                                         "role") == "user" and tab1_messages[-1].get("content") == last_user_content):
#                                     tab1_messages.append({"role": "user", "content": last_user_content})
#                                 last_assistant_index = None
#                                 for i in range(len(tab1_messages) - 1, -1, -1):
#                                     itm = tab1_messages[i]
#                                     if isinstance(itm, dict) and itm.get("role") == "assistant":
#                                         last_assistant_index = i
#                                         break
#                                 if last_assistant_index is None:
#                                     trimmed = tab1_messages[-1:]
#                                 else:
#                                     trimmed = tab1_messages[last_assistant_index:]
#                                 st.session_state["tab1_messages"] = trimmed
#                             except Exception:
#                                 logging.exception("Error updating tab1_messages after request revision; continuing.")
#
#                 except Exception as e:
#                     logging.exception("Failed to resume graph on request revision")
#                     st.error(f"Failed to request revision: {e}")
#
#             # Clear HIL UI state and return control to main UI
#             st.session_state["approval_status"] = None
#             st.session_state["review_data"] = None
#             st.session_state.pop("graph_origin_tab", None)
#             st.rerun()
#         st.markdown('</div>', unsafe_allow_html=True)
#
#     return True


# python
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

    # Render based on content type
    try:
        if is_table_content(extracted_text):
            # Render table or table-like text directly
            st.markdown(extracted_text, unsafe_allow_html=True)
        else:
            # Normalize latex and wrap inline math when needed
            body = normalize_latex(extracted_text)
            if not is_latex(body):
                body = auto_wrap_latex_math(body)
            if is_latex(body):
                body = convert_latex_display_math(body)

            # Render within the mathjax block (keeps previous visual style)
            st.markdown(f"<div class='mathjax-block'>{body}</div>", unsafe_allow_html=True)

        # Attempt to render any forecast table and expose PDF download; don't let failures break the UI
        try:
            render_forecast_table_from_text(extracted_text, key=f"pdf_{tab_key}_1")
            render_pdf_download(extracted_text, key=f"pdf_{tab_key}_2", user_query=user_query)
        except Exception as e:
            logging.error("PDF/table rendering error for %s: %s", tab_key, str(e), exc_info=True)
            # Show non-blocking error to the user
            st.error(f"Failed to render/download PDF or table: {str(e)}")
    except Exception as e:
        logging.error("Unexpected rendering error in display_analysis_results: %s", str(e), exc_info=True)
        st.error("An error occurred rendering the analysis results.")


# def display_analysis_results(content, tab_key, user_query=None):
#     """
#     Reusable function to display analysis results with consistent styling.
#     Stores the final elapsed time to prevent recalculation on re-renders.
#     """
#     elapsed = None
#     start_time_key = f"start_time_{tab_key}"
#     final_elapsed_key = f"final_elapsed_{tab_key}"  # Key for storing final elapsed time
#
#     # Check if we already have a final elapsed time stored
#     if final_elapsed_key in st.session_state:
#         elapsed = st.session_state[final_elapsed_key]
#         print(f"Using stored elapsed time for {tab_key}: {elapsed}")
#     else:
#         # Calculate elapsed time for the first time
#         start_time = st.session_state.get(start_time_key)
#         if start_time is not None:
#             elapsed = time.time() - start_time
#             # Store the final elapsed time so it doesn't change on re-renders
#             st.session_state[final_elapsed_key] = elapsed
#             print(f"Calculated and stored final elapsed time for {tab_key}: {elapsed}")
#             # Optional: Clean up the start time since we don't need it anymore
#             # st.session_state.pop(start_time_key, None)
#
#     st.markdown("### 📋 Research Results")
#
#     if elapsed is not None:
#         st.markdown(f'<div class="time-indicator">⏱️ Analysis completed in {elapsed:.2f} seconds</div>',
#                     unsafe_allow_html=True)
#
#     # Rest of your function remains the same...
#     # Check if content is clearly a table first
#     def is_table_content(content):
#         table_patterns = [
#             r"^#\s", r"\t", r"\|\s*[^|]+\s*\|",
#             r"<table>|<tr>|<td>|<th>",
#             r"Article & Source", r"Key Points.*Sentiment",
#             r"^\s*\d+\s*\|",  # Numbered pipe rows
#         ]
#         return any(re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
#                    for pattern in table_patterns)
#
#     if is_table_content(content):
#         # For table content, render directly without any wrapper
#         st.markdown(content, unsafe_allow_html=True)
#     else:
#         # For regular content, process as before
#         content = normalize_latex(content)
#         if not is_latex(content):
#             content = auto_wrap_latex_math(content)
#
#         if is_latex(content):
#             content = convert_latex_display_math(content)
#
#
#         #st.markdown(f"<div class='result-card'><div class='mathjax-block'>{content}</div></div>",
#                     #unsafe_allow_html=True)
#         # result_content = f"""
#         #     <div class="result-card">
#         #         <div class='mathjax-block'>{content}</div>
#         #     </div>
#         #     """
#         # st.markdown(result_content, unsafe_allow_html=True)
#         st.markdown(f"<div class='mathjax-block'>{content}</div>", unsafe_allow_html=True)
#
#     try:
#         render_forecast_table_from_text(content, key=f"pdf_{tab_key}_1")
#         render_pdf_download(content, key=f"pdf_{tab_key}_2", user_query=user_query)
#     except Exception as e:
#         error_msg = str(e)
#         logging.error(f"PDF/table rendering error: {error_msg}", exc_info=True)
#         st.error(f"Failed to render/download PDF or table: {error_msg}")


navbar = NavbarUI()


@require_auth
def main_app():
    """Main application function"""
    # Debug: Show current tab state
    st.sidebar.write(f"Current tab: {st.session_state.get('current_tab', 'unknown')}")
    st.sidebar.write(f"Tab1 content: {'exists' if 'last_content_tab1' in st.session_state else 'none'}")
    st.sidebar.write(f"Tab2 content: {'exists' if 'last_content_tab2' in st.session_state else 'none'}")

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

    st.markdown("""
        <style>
            [data-testid="stToolbar"] {display: none !important;}
        </style>
        """, unsafe_allow_html=True)

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
    tab1, tab2 = st.tabs(["🤖 AI Research Assistant", "⚡ Express Analysis"])

    # Use Streamlit's built-in tab state tracking
    if tab1:
        st.session_state.current_tab = "tab1"
    if tab2:
        st.session_state.current_tab = "tab2"

    # In main(), after the submit button in tab1:
    with tab1:
        # Initialize session state for input if not exists
        if "tab1_input" not in st.session_state:
            st.session_state.tab1_input = ""
        # user_prompt = create_chat_interface()
        user_prompt = st.text_area(
            "💬 Enter your stock research query:",
            value=st.session_state.tab1_input,
            placeholder="Type your message here...",
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

