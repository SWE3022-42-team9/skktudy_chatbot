from pathlib import Path

import streamlit as st

from chatbot import Chatbot

FILE_SAVE_PATH = str(Path(__file__).parent.parent) + "/data/"
files = []

@st.cache_resource
def load_model(model_name: str):
    return Chatbot(model_name=model_name)

def main():
    # App configurations
    st.title("💬 SKKTUDY CHATBOT")
    st.caption("🚀 기능 데모 프로그램")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요! 저는 SKKutor, 교육용 챗봇입니다.\n 어떤 도움이 필요하신가요?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # uploaded_file = st.file_uploader("파일 업로드", type=['png', 'jpg', 'jpeg', 'pdf'])

    # # Save file to data directory
    # if uploaded_file:
    #     with open(FILE_SAVE_PATH + uploaded_file.name, "wb") as f:
    #         f.write(uploaded_file.getbuffer())
    #         files.append(FILE_SAVE_PATH + uploaded_file.name)

    # Chatbot
    Chatbot = load_model(model_name='gpt-4')
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        ret = Chatbot(messages=prompt, files=files)
        st.session_state.messages.append({"role": "assistant", "content": ret})
        st.chat_message("assistant").write(ret)


if __name__ == "__main__":
    main()