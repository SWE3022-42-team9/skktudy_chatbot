import streamlit as st

from chatbot import Chatbot


@st.cache_resource
def load_model(model_name: str):
    return Chatbot(model_name=model_name)


def main():
    # App configurations
    st.title("💬 SKKTUDY CHATBOT")
    st.caption("🚀 기능 데모 프로그램")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "안녕하세요! 저는 교육용 챗봇: SKKutor입니다. 어떤 도움이 필요하신가요?",
            }
        ]

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chatbot
    Chatbot = load_model(model_name="gpt-4")
    if prompt := st.chat_input():
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        response = Chatbot(messages=prompt)

        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)


if __name__ == "__main__":
    main()
