import streamlit as st 
from app.auth.users import get_user_role
from app.guardrails.guardrail import check_input_guardrail, check_output_guardrail
from app.pipeline.rag_chain import build_rag_chain
import os

st.set_page_config(page_title="Company AI Assistant", page_icon="🤖", layout="centered")

@st.cache_resource
def get_cached_rag_chain(role):
    """
        This prevents reloading the embedding model on every click
    """
    return build_rag_chain("./chroma_db", role)


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_info" not in st.session_state:
    st.session_state.user_info = {"username": "", "role": ""}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def show_login_page():
    st.title("🔐 Company AI Assistant")
    with st.container(border = True):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            user_role = get_user_role(username)
            if user_role: #and verify_user(user, password): 
                st.session_state.logged_in = True
                st.session_state.user_info = {"username": username, "role": user_role}
                st.rerun()
            else:
                st.error("Invalid credentials. Please check your username/password.")


def show_chat_page():
    with st.sidebar:
        st.title("User Profile")
        st.write(f"**User:** {st.session_state.user_info['username']}")
        st.write(f"**Role:** :blue[{st.session_state.user_info['role'].upper()}]")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.chat_history = []
            st.rerun()

    st.title("💬 Company RAG Assistant")
    st.caption(f"Context-aware assistant for {st.session_state.user_info['role']} department")

    # 1. Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📄 Sources used"):
                    for src in message["sources"]:
                        st.write(f"• {src}")

    # 2. Handle New Input
    if prompt := st.chat_input("Ask a question about company policy..."):
        # Save and display user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # --- PRE-PROCESSING: Greetings & Simple Responses ---
            GREETING_RESPONSES = {
                "hi": "Hello! I'm your Company AI. How can I help you today?",
                "hello": "Hi there! Ready to help with your company queries.",
                "who are you": "I am the official RAG Assistant for our company.",
            }
            
            clean_prompt = prompt.lower().strip().rstrip("?")
            
            if clean_prompt in GREETING_RESPONSES:
                response_content = GREETING_RESPONSES[clean_prompt]
                sources = []
            else:
                # --- RAG PROCESSING: Guardrails & Chain ---
                violation_msg = check_input_guardrail(prompt)
                if violation_msg:
                    response_content = f"⚠️ {violation_msg}"
                    sources = []
                else:
                    chain = get_cached_rag_chain(st.session_state.user_info["role"])
                    with st.spinner("Processing Query..."):
                        try:
                            raw_answer = chain.invoke(prompt)
                            response_content = check_output_guardrail(raw_answer)
                            sources = ["Retrieved from company policy docs"] 
                        except Exception as e:
                            response_content = f"Error: {str(e)}"
                            sources = []

            # 3. Display Assistant Response
            st.markdown(response_content)
            if sources:
                with st.expander("📄 Sources used"):
                    for source in sources:
                        st.write(f"• {source}")

            # 4. Save to History
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response_content,
                "sources": sources
            })
            
if not st.session_state.logged_in:
    show_login_page()
else:
    show_chat_page()