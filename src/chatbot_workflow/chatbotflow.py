import streamlit as st
import uuid
import html

from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------
# 1. Configuration
# ---------------------------------

load_dotenv()

st.set_page_config(
    page_title="Zarvo AI Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS = {
    "GPT-OSS 20B (Fast)": "openai/gpt-oss-20b",
    "GPT-OSS 120B (Smart)": "openai/gpt-oss-120b",
    "Llama 3.3 70B (Balanced)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Instant)": "llama-3.1-8b-instant",
}

PERSONAS = {
    "🤖 General Assistant": "You are a helpful, friendly and accurate AI assistant.",
    "💻 Coding Expert": (
        "You are an expert software engineer. Give correct, clean, well-explained "
        "code and mention edge cases."
    ),
    "✍️ Writing Helper": (
        "You are a skilled writing assistant. Help with drafting, editing and "
        "improving text with clear, polished language."
    ),
    "🎓 Tutor": (
        "You are a patient tutor. Explain concepts step by step using simple "
        "language and examples."
    ),
}

STYLES = {
    "Balanced": "Give balanced answers with enough detail.",
    "Concise": "Keep answers short and to the point.",
    "Detailed": "Give thorough, detailed answers with examples.",
}

SUGGESTIONS = [
    ("💡 Explain a concept", "Explain how neural networks work in simple words."),
    ("💻 Help me code", "Write a Python function to remove duplicates from a list."),
    ("✍️ Write something", "Write a professional email asking for a project deadline extension."),
    ("🚀 Give me ideas", "Give me 5 creative project ideas for a MERN stack developer."),
]

# ---------------------------------
# 2. Custom CSS
# ---------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
}

/* ---------- App background ---------- */
.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(99, 102, 241, 0.12), transparent 40%),
        radial-gradient(circle at 90% 10%, rgba(236, 72, 153, 0.08), transparent 35%),
        #0d0d12;
    color: #f1f1f5;
}

/* Hide default Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

.block-container {
    padding-top: 2rem;
    padding-bottom: 6rem;
    max-width: 920px;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #15151d 0%, #101016 100%);
    border-right: 1px solid #25252f;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 4px;
}

.brand-logo {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, #6366f1, #ec4899);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.35);
}

.brand-name {
    font-size: 1.5rem;
    font-weight: 800 !important;
    letter-spacing: 0.3px;
    color: #ffffff;
    line-height: 1.1;
}

.brand-tag {
    font-size: 0.75rem;
    color: #8b8b9a;
}

.side-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7c7c8d;
    margin: 6px 0 4px 2px;
}

[data-testid="stSidebar"] hr {
    border-color: #25252f;
    margin: 0.9rem 0;
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stDownloadButton > button {
    background-color: #1d1d27;
    color: #e6e6ee;
    border: 1px solid #2c2c39;
    border-radius: 10px;
    font-weight: 500;
    text-align: left;
    transition: all 0.2s ease;
}

[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] .stDownloadButton > button:hover {
    background-color: #2a2a3a;
    border-color: #6366f1;
    color: #ffffff;
    transform: translateY(-1px);
}

/* Primary buttons (New Chat + active chat) */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border: none;
    color: white;
    font-weight: 600;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    filter: brightness(1.1);
    border: none;
}

/* Sidebar inputs */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] input {
    background-color: #1d1d27 !important;
    border-color: #2c2c39 !important;
    border-radius: 10px !important;
    color: #f1f1f5 !important;
}

/* ---------- Hero / header ---------- */
.hero {
    text-align: center;
    padding: 2.2rem 0 1.2rem 0;
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}

.hero-sub {
    color: #9a9aab;
    font-size: 1.05rem;
}

.chat-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.2rem 0 0.8rem 0;
    border-bottom: 1px solid #25252f;
    margin-bottom: 1rem;
}

.chat-header-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #ffffff;
}

.chat-header-meta {
    font-size: 0.8rem;
    color: #8b8b9a;
}

.pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.35);
    color: #a5b4fc;
    font-size: 0.72rem;
    font-weight: 500;
    margin-left: 6px;
}

/* ---------- Suggestion cards (main area buttons) ---------- */
.main .stButton > button,
[data-testid="stMain"] .stButton > button {
    background-color: #17171f;
    color: #e6e6ee;
    border: 1px solid #2c2c39;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    height: 100%;
    min-height: 64px;
    font-weight: 500;
    transition: all 0.2s ease;
}

.main .stButton > button:hover,
[data-testid="stMain"] .stButton > button:hover {
    border-color: #8b5cf6;
    background-color: #20202c;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(139, 92, 246, 0.18);
    color: #ffffff;
}

/* ---------- Chat messages ---------- */
/* AI message: left side, clean, full width, no box */
[data-testid="stChatMessage"] {
    background-color: transparent;
    border: none;
    padding: 10px 4px;
    margin-bottom: 8px;
}

/* User message: right side, transparent grey rounded box */
.user-row {
    display: flex;
    justify-content: flex-end;
    width: 100%;
    margin: 10px 0 16px 0;
}

.user-bubble {
    max-width: 75%;
    background-color: rgba(255, 255, 255, 0.09);
    border: 1px solid rgba(255, 255, 255, 0.10);
    color: #ececf3;
    padding: 12px 18px;
    border-radius: 20px;
    line-height: 1.6;
    font-size: 1rem;
    text-align: left;
    overflow-wrap: anywhere;
    word-break: break-word;
}

.stChatMessage p, .stChatMessage li {
    color: #ececf3;
    line-height: 1.65;
}

.stChatMessage code {
    background-color: #0b0b10;
    color: #f9a8d4;
    border-radius: 6px;
}

.stChatMessage pre {
    border: 1px solid #25252f;
    border-radius: 12px;
}

/* ---------- Chat input ---------- */
[data-testid="stChatInput"] {
    background-color: #16161e;
    border: 1px solid #2c2c39;
    border-radius: 16px;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #8b5cf6;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.18);
}

[data-testid="stChatInput"] textarea {
    color: #ffffff;
}

[data-testid="stBottom"] > div {
    background: transparent;
}

/* ---------- Misc ---------- */
h1, h2, h3, p, label {
    color: #f1f1f5;
}

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb { background: #2c2c39; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #3c3c4f; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# 3. LangGraph Backend
# ---------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@st.cache_resource
def get_llm(model_name: str, temperature: float):
    return ChatGroq(
        model=model_name,
        temperature=temperature,
    )


@st.cache_resource
def get_workflow():

    def chatbot(state: ChatState, config):

        settings = config.get("configurable", {})

        model_name = settings.get("model_name", "openai/gpt-oss-20b")
        temperature = settings.get("temperature", 0.7)
        system_prompt = settings.get(
            "system_prompt",
            "You are a helpful, friendly and accurate AI assistant."
        )

        llm = get_llm(model_name, temperature)

        # System prompt is added at call time, so it is not saved in history
        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        response = llm.invoke(messages)

        return {
            "messages": [response]
        }

    checkpoint = MemorySaver()

    graph = StateGraph(ChatState)

    graph.add_node("chatbot", chatbot)

    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)

    workflow = graph.compile(
        checkpointer=checkpoint
    )

    return workflow


workflow = get_workflow()

# ---------------------------------
# 4. Session State
# ---------------------------------

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


def create_new_chat():

    thread_id = str(uuid.uuid4())

    st.session_state.chats[thread_id] = "New chat"

    st.session_state.active_thread_id = thread_id


def delete_chat(thread_id):

    st.session_state.chats.pop(thread_id, None)

    if not st.session_state.chats:
        create_new_chat()
    else:
        st.session_state.active_thread_id = next(
            reversed(st.session_state.chats)
        )


def message_text(message) -> str:

    content = message.content

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return "".join(parts)

    return str(content)


def render_user_message(text: str):

    safe_text = html.escape(text).replace("\n", "<br>")

    st.markdown(
        f'<div class="user-row"><div class="user-bubble">{safe_text}</div></div>',
        unsafe_allow_html=True,
    )


if not st.session_state.chats:
    create_new_chat()

if st.session_state.active_thread_id not in st.session_state.chats:
    st.session_state.active_thread_id = next(
        iter(st.session_state.chats)
    )

# ---------------------------------
# 5. Active Conversation Data
# ---------------------------------

active_thread_id = st.session_state.active_thread_id

# ---------------------------------
# 6. Sidebar
# ---------------------------------

with st.sidebar:

    st.markdown("""
    <div class="brand">
        <div class="brand-logo">🤖</div>
        <div>
            <div class="brand-name">Zarvo AI</div>
            <div class="brand-tag">Your smart AI assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    if st.button(
        "✨  New Chat",
        use_container_width=True,
        type="primary",
        key="new_chat_btn"
    ):
        create_new_chat()
        st.rerun()

    st.divider()

    # ----- AI Settings -----

    st.markdown('<div class="side-label">AI Settings</div>', unsafe_allow_html=True)

    model_label = st.selectbox(
        "Model",
        list(MODELS.keys()),
        index=0,
        key="model_label",
    )

    persona_label = st.selectbox(
        "Assistant Mode",
        list(PERSONAS.keys()),
        index=0,
        key="persona_label",
    )

    style_label = st.selectbox(
        "Answer Style",
        list(STYLES.keys()),
        index=0,
        key="style_label",
    )

    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Lower = more accurate and focused. Higher = more creative.",
        key="temperature",
    )

    st.divider()

    # ----- Chat History -----

    st.markdown('<div class="side-label">Chat History</div>', unsafe_allow_html=True)

    search_query = st.text_input(
        "Search chats",
        placeholder="🔍 Search chats...",
        label_visibility="collapsed",
        key="chat_search",
    )

    chat_items = list(reversed(list(st.session_state.chats.items())))

    if search_query.strip():
        chat_items = [
            (tid, title) for tid, title in chat_items
            if search_query.lower() in title.lower()
        ]

    if not chat_items:
        st.caption("No chats found.")

    history_box = st.container(height=260, border=False)

    with history_box:
        for thread_id, title in chat_items:

            is_active = thread_id == active_thread_id

            if st.button(
                ("💬  " + title),
                key=f"chat_{thread_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.active_thread_id = thread_id
                st.rerun()

    st.divider()

    # ----- Chat Actions -----

    st.markdown('<div class="side-label">Chat Actions</div>', unsafe_allow_html=True)

    _state_for_export = workflow.get_state(
        {"configurable": {"thread_id": active_thread_id}}
    )
    _export_messages = _state_for_export.values.get("messages", [])

    export_lines = []
    for m in _export_messages:
        if m.type == "human":
            export_lines.append("You: " + message_text(m))
        elif m.type == "ai":
            export_lines.append("Zarvo AI: " + message_text(m))
    export_text = "\n\n".join(export_lines) if export_lines else "No messages yet."

    st.download_button(
        "⬇️  Export Chat",
        data=export_text,
        file_name="zarvo_chat.txt",
        mime="text/plain",
        use_container_width=True,
        disabled=not _export_messages,
    )

    if st.button("🗑️  Delete This Chat", use_container_width=True):
        delete_chat(active_thread_id)
        st.rerun()

    st.divider()

    st.caption("⚡ Powered by LangGraph + Groq")

# ---------------------------------
# 7. Build config from settings
# ---------------------------------

system_prompt = PERSONAS[persona_label] + " " + STYLES[style_label]

config = {
    "configurable": {
        "thread_id": active_thread_id,
        "model_name": MODELS[model_label],
        "temperature": temperature,
        "system_prompt": system_prompt,
    }
}

# Retrieve conversation history

state = workflow.get_state(config)

messages = state.values.get("messages", [])

visible_messages = [m for m in messages if m.type in ("human", "ai")]

# ---------------------------------
# 8. Main Chat Interface
# ---------------------------------

if not visible_messages:

    # Welcome screen

    st.markdown("""
    <div class="hero">
        <div class="hero-title">Hello, how can I help you today?</div>
        <div class="hero-sub">Ask me anything — code, writing, learning, ideas and more.</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2 = st.columns(2)

    for i, (label, prompt_text) in enumerate(SUGGESTIONS):

        target_col = col1 if i % 2 == 0 else col2

        with target_col:
            if st.button(
                label,
                key=f"suggest_{i}",
                use_container_width=True,
                help=prompt_text,
            ):
                st.session_state.pending_prompt = prompt_text
                st.rerun()

else:

    st.markdown(f"""
    <div class="chat-header">
        <div class="chat-header-title">{st.session_state.chats[active_thread_id]}</div>
        <div class="chat-header-meta">
            {len(visible_messages)} messages
            <span class="pill">{model_label}</span>
            <span class="pill">{persona_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Display previous messages

for message in visible_messages:

    if message.type == "human":

        render_user_message(message_text(message))

    else:

        with st.chat_message("assistant", avatar="🤖"):

            st.markdown(message_text(message))

# ---------------------------------
# 9. User Input
# ---------------------------------

typed_input = st.chat_input(
    "Type your message here..."
)

user_input = typed_input or st.session_state.pending_prompt

st.session_state.pending_prompt = None

if user_input:

    # Set chat title using first message

    if st.session_state.chats[active_thread_id] == "New chat":

        st.session_state.chats[active_thread_id] = (
            user_input[:35] + ("..." if len(user_input) > 35 else "")
        )

    # Show the user's message immediately

    render_user_message(user_input)

    # Invoke LangGraph

    with st.chat_message("assistant", avatar="🤖"):

        with st.spinner("AI is thinking..."):

            workflow.invoke(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=config
            )

            new_state = workflow.get_state(config)
            new_messages = new_state.values.get("messages", [])

            reply = message_text(new_messages[-1]) if new_messages else ""

        st.markdown(reply)

    # Refresh interface

    st.rerun()