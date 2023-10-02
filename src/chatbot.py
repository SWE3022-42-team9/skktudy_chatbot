from typing import Any, Optional, Type
from typing import Any, List, Optional, Sequence, Tuple

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
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.agents import AgentExecutor
# from langchain.agents.output_parsers import JSONAgentOutputParser
from output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_messages

from langchain.agents.conversational_chat.output_parser import ConvoOutputParser

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AgentAction, BaseOutputParser, BasePromptTemplate

from langchain.chains import LLMChain
from templates import (
    PREFIX,
    SUFFIX,
    TEMPLATE_TOOL_RESPONSE,
    FORMAT_INSTRUCTIONS,
)

load_dotenv()

base_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(PREFIX),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

def _handle_error(error) -> str:
    return str(error)[:50]

def summarize_template(query: str) -> str:
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    """You are a helpful assistant that summarizes a document.
                    Give clear and concise summaries of the document."""
                )
            ),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )



class Chatbot:
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
    ):
        chat_model = ChatOpenAI(model_name=model_name, temperature=0)

        # self.summaryChain = LLMChain(llm=self.llm, prompt="Prompt")

        chatbot = LLMChain(
            llm=chat_model,
            prompt=base_prompt,
            verbose=True,
        )
        tools = [
            Tool(
                name="Summary",
                func=chatbot.run,
                description="useful only when you need summarize a given pdf document",
                return_direct=True,
            )
        ]


        prompt = self.create_prompt(
            tools,
            system_message=PREFIX,
            human_message=SUFFIX,
            input_variables=None,
            output_parser=ConvoOutputParser()
        )
        chat_model_with_stop = chat_model.bind(stop=["\nObservation"])


        agent = {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_messages(x['intermediate_steps'], template_tool_response=TEMPLATE_TOOL_RESPONSE),
            "chat_history": lambda x: x["chat_history"],
        } | prompt | chat_model_with_stop | JSONAgentOutputParser()


        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)



    def __call__(
        self,
        messages: list[HumanMessage], 
        files: Optional[list[str]] = None,
        **kwargs: Any
    ) -> Any:
        ret = self.agent.invoke({'input': messages})['output']
        print(ret)
        return ret

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
        _output_parser = output_parser
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
