
import sys, os
from PIL import Image
from streamlit.web import cli as stcli
import streamlit as st
from streamlit import runtime

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
# -----------------------------------------------------------------------------
# Check if ollama running models available
# -----------------------------------------------------------------------------
import subprocess
import re
completed = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
models_info = completed.stdout.split('\n')
model_names = []
model_map = {}

tt = re.split(r'\s+', models_info[0])
if len(models_info) > 1:
    for i in range(1, len(models_info)):
        s = re.split(r'\s+', models_info[i])
        if len(s) > 1:
            model_names.append(s[0])
            model_map[s[0]] = "<p>" + tt[0] + ": " + s[0] + "</p><p>" + tt[1] + ": " + \
                              s[1] + "</p><p>" + tt[2] + ": " + s[2] + " " + s[3] + "</p>"
    model_names = sorted(model_names)
else:
    model_map[""] = "<p>" + tt[0] + ": </p><p>" + tt[1] + ": </p><p>" + tt[2] + ": </p>"
    model_names.append("")

im = Image.open("static/deepseek-color.png")

def main():
    if 'Model' not in st.session_state:
        st.session_state['Model'] = 'value'

    st.set_page_config(
        page_title="DeepSeek AI",
        page_icon=im,
        layout="wide",
    )
    model_option = model_names[0]

    st.markdown("""
    <style>
        /* Main container background */
        .stApp {
            background: #f0f2f6;
        }

        /* Chat message styling */
        .message {
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 1.5rem;
            max-width: 80%;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            line-height: 1.6;
            font-size: 16px;
        }

        .user-message {
            background: #ffffff;
            color: black;
            margin-left: auto;
        }

        .assistant-message {
            background: #ffffff;
            margin-right: auto;
            border: 1px solid #e0e0e0;
        }

        /* Input form styling */
        .stForm {
            background: white;
            padding: 1.5rem;
            border-radius: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* History container */
        .history-container {
            background: white;
            padding: 2rem;
            border-radius: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* Submit button styling */
        .stButton>button {
            background: #4a90e2 !important;
            color: white !important;
            border: none !important;
            padding: 12px 25px !important;
            border-radius: 12px !important;
            font-size: 16px !important;
            transition: all 0.3s ease !important;
            width: 100%;
        }
        
        .stButton>button:hover {
            background: #357abd !important;
            transform: translateY(-2px);
        }
        select { width: 100%; }
    </style>
    """, unsafe_allow_html=True)
    st.title("🤖 :blue[DeepSeek AI Assistant]")
    html = '''
    <p style="color:black">🔆 Powered by Ollama, LangChain and local running DeepSeek LLM</p>
    '''
    st.markdown(html, unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.image("static/deepseek_w.png", width=120)

        def get_new_values_list():
            st.session_state['chat_history'] = []

        model_option = st.selectbox( "Model", model_names, on_change=get_new_values_list)
        st.markdown("""
        <div style="text-align: left; color: white; margin-top: 0rem;">""" +
            model_map[model_option] + """
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: left; color: white; margin-top: 0rem;">
            <p>🛡️ Your conversations are private and never stored</p>
            <p>🔋 Powered by """ + model_option + """ LLM</p>
            <p>🖥️ Running locally via Ollama</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        if st.button("🧹 Clear Chat History"):
            st.session_state['chat_history'] = []
            st.rerun()
    if model_option != "":
        model = ChatOllama(model=model_option, base_url="http://localhost:11434")

    if "chat_history" not in st.session_state:
        st.session_state['chat_history'] = []

    def generate_response(chat_history):
        chat_template = ChatPromptTemplate.from_messages(chat_history)
        chain = chat_template | model | StrOutputParser()
        return chain.invoke({})

    def get_history():
        chat_history = []
        for chat in st.session_state['chat_history']:
            chat_history.append(HumanMessagePromptTemplate.from_template(chat['user']))
            chat_history.append(AIMessagePromptTemplate.from_template(chat['assistant']))
        return chat_history

    with st.container():
        for chat in st.session_state['chat_history']:
            cols = st.columns([1, 10])
            with cols[1]:
                st.markdown(f"""
                <div class="message user-message">
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <img src="app/static/user.png" width="40px" height="40px">
                        <strong style="font-size: 18px;">You:</strong>
                    </div>
                    <div style="height:100px; overflow-y:auto">
                    {chat['user']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            cols = st.columns([10, 1])
            with cols[0]:
                st.markdown(f"""
                <div class="message assistant-message">
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <img src="app/static/deepseek.png" width="40px" height="40px">
                        <strong style="font-size: 18px;color: black;">DeepSeek:</strong>
                    </div>
                    <div style="height:300px; overflow-y:auto">
                    <strong style="font-size: 14px;color: black;">
                    {chat['assistant']}
                    </strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with st.form("chat_form"):
        cols = st.columns([8, 1])
        user_input = cols[0].text_input(
            "Ask me anything...",
            placeholder="Type your message here...",
            label_visibility="collapsed"
        )
        if model_option != "":
            submitted = cols[1].form_submit_button("🪄 Send", disabled=False)
        else:
            submitted = cols[1].form_submit_button("🪄 Send", disabled=True)

        if submitted and user_input:
            #: blue, green, orange, red, violet, gray/grey, rainbow
            with st.spinner("✨ :gray[Generating response...]"):
                prompt = HumanMessagePromptTemplate.from_template(user_input)
                chat_history = get_history()
                chat_history.append(prompt)
                response = generate_response(chat_history)
                st.session_state['chat_history'].append({
                    'user': user_input,
                    'assistant': response
                })
                st.rerun()

if __name__ == '__main__':
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
