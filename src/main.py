import streamlit as st
from openai import OpenAI
from datetime import datetime
import dotenv
import os
import json
from typing import Callable

from get_current_temperature import get_current_temperature
from search_wikipedia import search_wikipedia
from centigrade_to_fahrenheit import centigrade_to_fahrenheit
from get_forecast_weather import ForecastWeather, OpenMeteoInput
from get_historical_weather import HistoricalTemperature, OpenMeteoHistoryInput
from langchain.pydantic_v1 import Field, create_model
from langchain.schema.agent import AgentFinish
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.tools.render import format_tool_to_openai_function
from langchain.tools import StructuredTool
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents import AgentExecutor
from langchain.memory import ChatMessageHistory

def run_agent(user_input, agent_chain):
    intermediate_steps = []
    while True:
        result = agent_chain.invoke({
            "input": user_input, 
            "intermediate_steps": intermediate_steps
        })
        if isinstance(result, AgentFinish):
            return result
        tool = {
            "search_wikipedia": search_wikipedia, 
            "get_current_temperature": get_current_temperature,
            # "get_forecast_weather": get_forecast_weather
        }[result.tool]
        observation = tool.run(result.tool_input)
        intermediate_steps.append((result, observation))

def create_tool(callable:Callable):
    method = callable
    args = {k:v for k,v in method.__annotations__.items() if k != "self"}
    name = method.__name__
    doc = method.__doc__
    func_desc = doc[doc.find("<desc>") + len("<desc>"):doc.find("</desc>")]
    arg_desc = dict()
    for arg in args.keys():
        desc = doc[doc.find(f"{arg}: ")+len(f"{arg}: "):]
        desc = desc[:desc.find("\n")]
        arg_desc[arg] = desc
    arg_fields = dict()
    for k,v in args.items():
        arg_fields[k] = (v, Field(description=arg_desc[k]))

    Model = create_model('Model', **arg_fields)

    tool = StructuredTool.from_function(
        func=method,
        name=name,
        description=func_desc,
        args_schema=Model,
        return_direct=False,
    )
    return tool

def main():
    st.title("ChatGPT-like clone")


    # Set OpenAI API key from Streamlit secrets
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv()) # read local .env file
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    forecast_weather = ForecastWeather()
    # Create agent
    forecast_weather_tool = StructuredTool.from_function(
        func=forecast_weather.get_forecast_weather,
        name="get_forecast_weather",
        description="Fetch forecast weather for given coordinates.",
        args_schema=OpenMeteoInput,
        return_direct=False,
    )


    
    historical_temperature = HistoricalTemperature()

    historical_temperature_tool = StructuredTool.from_function(
        func=historical_temperature.get_historical_temperature,
        name="get_historical_temperature",
        description="Fetch historical temperatures for given coordinates across a given time period.",
        args_schema=OpenMeteoHistoryInput,
        return_direct=False
    )

    tools = [get_current_temperature, search_wikipedia, forecast_weather_tool, historical_temperature_tool, centigrade_to_fahrenheit]
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.tools.render import format_tool_to_openai_function
    from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are helpful assistant. You have tools available to you to help you answer users questions. The current date is {datetime.now().strftime('%Y-%m-%d')} in YYYY-MM-DD format."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    functions = [format_tool_to_openai_function(f) for f in tools]
    model = ChatOpenAI(temperature=0, verbose=True).bind(functions=functions)
    agent_chain = RunnablePassthrough.assign(
        agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
    ) | prompt | model | OpenAIFunctionsAgentOutputParser()
    from langchain.memory import ConversationBufferMemory
    memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")   
    # memory = StreamlitChatMessageHistory(key="chat_messages")
    def run_agent_chain(prompt, chat_history):
        history = ChatMessageHistory()
        history.messages = chat_history
        memory = ConversationBufferMemory(chat_memory=history, return_messages=True,memory_key="chat_history")
        agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, memory=memory)
        return agent_executor.invoke({"input": prompt})
    # for func in tools:
    #     print(func)
    # print("\n")
    for func in functions:
        print("\n")
        print(json.dumps(func,indent=2))
        print("\n")
    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_memory" not in st.session_state:
        st.session_state["agent_memory"] = []

    # Display chat messages from history on app rerun   
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # stream = client.chat.completions.create(
            #     model=st.session_state["openai_model"],
            #     messages=[
            #         {"role": m["role"], "content": m["content"]}
            #         for m in st.session_state.messages
            #     ],
            #     stream=True,
            # )
            response = run_agent_chain(prompt, st.session_state["agent_memory"])
            st.write(response['output'])
            st.session_state["agent_memory"] = response["chat_history"]
            if forecast_weather.fig is not None:
                fig = forecast_weather.fig
                st.plotly_chart(fig)
            if historical_temperature.fig is not None:
                fig = historical_temperature.fig
                st.plotly_chart(fig)
            print(st.session_state["agent_memory"])
        st.session_state.messages.append({"role": "assistant", "content": response['output']})
if __name__ == "__main__":
    main()