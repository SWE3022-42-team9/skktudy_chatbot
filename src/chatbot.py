from typing import Any, Optional, Type

import openai
from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun,
                                         CallbackManagerForToolRun)
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.utilities import SerpAPIWrapper
from langchain.document_loaders import PyPDFLoader

load_dotenv()


class Chatbot:
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
    ):
        self.chatbot = ChatOpenAI(model_name=model_name)

    def __call__(
        self,
        messages: list[AIMessage], 
        files: Optional[list[str]] = None,
        **kwargs: Any
    ) -> Any:
        pass


def chat(prompt):
    chatbot = ChatOpenAI(temperature=0)
    ret = chatbot([HumanMessage(content=prompt)])
    ret = ret[0].content

    


    return ret