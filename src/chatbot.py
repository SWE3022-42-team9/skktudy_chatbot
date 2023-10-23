from typing import Any, List, Optional, Sequence
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import ConversationalRetrievalChain


from langchain.text_splitter import CharacterTextSplitter

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, Tool
from langchain.agents.conversational_chat.output_parser import ConvoOutputParser
from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import BaseOutputParser, BasePromptTemplate, HumanMessage
from langchain.tools import BaseTool, Tool

from output_parsers import JSONAgentOutputParser
from templates import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX, TEMPLATE_TOOL_RESPONSE
import streamlit as st
load_dotenv()


class Chatbot:
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
    ):
        # Initialize the chatbot with model, memory, tool and prompt
        if 'chat_model' not in st.session_state:  
            st.session_state['chat_model'] = ChatOpenAI(model_name=model_name, temperature=0)
        
        if 'memory' not in st.session_state:
            st.session_state['memory'] = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
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
    def create_prompt(
        self,
        tools: Sequence[BaseTool],
        system_message: str = PREFIX,
        human_message: str = SUFFIX,
        input_variables: Optional[List[str]] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> BasePromptTemplate:
        tool_strings = "\n".join(
            [f"> {tool.name}: {tool.description}" for tool in tools]
        )
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = human_message.format(
            format_instructions=FORMAT_INSTRUCTIONS
        )
        final_prompt = format_instructions.format(
            tool_names=tool_names, tools=tool_strings
        )
        if input_variables is None:
            input_variables = ["input", "chat_history", "agent_scratchpad"]
        messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(final_prompt),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    @classmethod
    def load_documents(self, files):
        total_documents, document = [], []

        for file in files:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file)
                document.extend(loader.load())
            elif file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                loader = UnstructuredImageLoader(file)
                document.extend(loader.load())

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
            document = text_splitter.split_documents(document)

        total_documents.extend(document)
        st.session_state['retriever'] = FAISS.from_documents(total_documents, OpenAIEmbeddings())


    
    @classmethod
    def update_agent(self):
        if st.session_state['retriever'] is None:
            tools = [
                Tool(
                    name="Dummy Tool for creating agents",
                    func=lambda x: x,
                    description="Do not use this tool",
                    return_direct=True,
                )
            ]

        else:
            qa_chain = ConversationalRetrievalChain.from_llm(
                st.session_state['chat_model'],
                st.session_state['retriever'].as_retriever(),
                # condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                memory=st.session_state['memory']
            )

            tools = [
                Tool(
                    name="QA",
                    func=qa_chain(st.session_state['chat_model']),
                    description="useful for when you need to answer a question about a given document",
                    return_direct=True,
                )
            ]
        prompt = self.create_prompt(
            tools,
            system_message=PREFIX,
            human_message=SUFFIX,
            input_variables=None,
            output_parser=ConvoOutputParser(),
        )
        chat_model_with_stop = st.session_state['chat_model'].bind(stop=["\nObservation"])

        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_messages(
                    x["intermediate_steps"],
                    template_tool_response=TEMPLATE_TOOL_RESPONSE,
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | chat_model_with_stop
            | JSONAgentOutputParser()
        )

        self.agent = AgentExecutor(
            agent=agent, tools=tools, verbose=True, memory=st.session_state['memory']
        )


