from typing import Any, List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, Tool
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.agents.conversational_chat.output_parser import ConvoOutputParser
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import HumanMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import Tool
from langchain.vectorstores import FAISS

from templates import PREFIX, QA_TEMPLATE, SUFFIX

load_dotenv()


class Chatbot:
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
    ):
        # Initialize the chatbot with model, memory, tool and prompt
        if "memory" not in st.session_state:
            st.session_state["memory"] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
            )
        if "chat_model" not in st.session_state:
            st.session_state["chat_model"] = ChatOpenAI(
                model_name=model_name, temperature=0
            )

        self.update_agent()

    def __call__(
        self,
        messages: List[HumanMessage],
        files: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if files:
            self.load_documents(files)

        self.update_agent()

        response = self.agent.invoke({"input": messages})["output"]
        return response

    @classmethod
    def load_documents(self, files):
        total_documents, document = [], []

        for file in files:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file)
                document.extend(loader.load())
            elif (
                file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")
            ):
                loader = UnstructuredImageLoader(file)
                document.extend(loader.load())

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
            document = text_splitter.split_documents(document)

        total_documents.extend(document)
        st.session_state["retriever"] = FAISS.from_documents(
            total_documents, OpenAIEmbeddings()
        )

    @classmethod
    def update_agent(self):
        tools = [
            Tool(
                name="Dummy Tool for creating agents",
                func=lambda x: x,
                description="Do not use this tool",
                return_direct=True,
            )
        ]

        if st.session_state["retriever"]:
            prompt = PromptTemplate(
                template=QA_TEMPLATE, input_variables=["context", "question"]
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state["chat_model"],
                retriever=st.session_state["retriever"].as_retriever(),
                chain_type="stuff",
                memory=st.session_state["memory"],
                chain_type_kwargs={"prompt": prompt},
                verbose=False,
            )

            tools.append(
                Tool(
                    name="QA",
                    func=qa_chain.run,
                    description="useful for when you need to answer a question about a document, image, or file",
                    return_direct=True,
                )
            )

        agent = ConversationalChatAgent.from_llm_and_tools(
            llm=st.session_state["chat_model"],
            tools=tools,
            system_message=PREFIX,
            human_message=SUFFIX,
            output_parser=ConvoOutputParser(),
        )
        self.agent = AgentExecutor(
            agent=agent, tools=tools, verbose=False, memory=st.session_state["memory"]
        )
