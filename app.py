# app.py
import streamlit as st
from rag_chat import load_vectorstore_from_file, get_llm_response, load_vectorstore_from_text
from utils import save_uploaded_file

st.set_page_config(page_title="IIT Roorkee Chatbot", layout="wide")
st.title("🤖 Chatbot")

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for personality & knowledge upload
with st.sidebar:
    st.header("🧠 Customize Your IITR Bot")

    # Combined instructions input
    st.markdown("### ✍️ Define Bot Role & Knowledge")
    st.markdown("Describe how the bot should behave and what it should know.")
    default_prompt = (
        "You are a helpful academic advisor for IIT Roorkee students. "
        "You assist with academic queries, hostel info, and deadlines."
    )
    combined_prompt = st.text_area("Bot Instructions + Knowledge", value=default_prompt)
    st.session_state.system_prompt = combined_prompt

    if st.button("💾 Load Text Knowledge"):
        st.session_state.vectorstore = load_vectorstore_from_text(combined_prompt)
        st.success("Instructions + knowledge added to bot memory!")

    st.markdown("### 📄 Upload PDF or Text File")
    st.markdown("Upload documents that the bot should use to answer questions.")
    uploaded_file = st.file_uploader("Upload PDF or .txt", type=["pdf", "txt"])
    if uploaded_file:
        file_path = save_uploaded_file(uploaded_file)
        st.session_state.vectorstore = load_vectorstore_from_file(file_path)
        st.success("File uploaded and indexed!")


# Helper to display chat bubbles
def display_message(msg):
    bubble_style = """
        display: inline-block;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 80%;
        word-wrap: break-word;
    """
    if msg["role"] == "user":
        col1, col2 = st.columns([1, 5])
        with col2:
            st.markdown(
                f"<div style='background-color: #d4edc9; color: black; text-align: right; {bubble_style}; float: right;'>{msg['content']}</div>",
                unsafe_allow_html=True
            )
    else:
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(
                f"<div style='background-color: #f1f0f0; color: black; text-align: left; {bubble_style}; float: left;'>{msg['content']}</div>",
                unsafe_allow_html=True
            )

# Main chat interface
st.markdown("---")
st.subheader("💬 Chat Interface")
user_input = st.chat_input("Type your message here...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": "🤖 Thinking..."})
    for msg in st.session_state.messages:
        display_message(msg)

    # Get response from LLM
    response = get_llm_response(
        query=user_input,
        vectorstore=st.session_state.vectorstore,
        system_prompt=st.session_state.system_prompt
    )

    # Update last assistant message
    st.session_state.messages[-1] = {"role": "assistant", "content": response}
    st.rerun()

# Re-display chat after rerun
for msg in st.session_state.messages:
    display_message(msg)

# Auto scroll to bottom
st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)
