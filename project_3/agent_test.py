import traceback

try:
    from langchain_community.chat_models import ChatOllama
    from langchain.agents import initialize_agent, Tool, AgentType
    print('Imports OK')

    def get_current_weather(city: str, unit: str = 'celsius') -> str:
        symbol = '°C' if unit.lower().startswith('c') else '°F'
        temp = 37 if symbol == '°C' else 73
        return f"It is {temp}{symbol} and sunny in {city}."

    weather_tool = Tool.from_function(get_current_weather, name='get_weather', description='Get current weather for a city')
    print('Tool created:', weather_tool.name)

    llm = ChatOllama(model='gemma3:1b', temperature=0)
    print('ChatOllama created')

    try:
        agent_type = AgentType.REACT_DESCRIPTION
    except Exception:
        try:
            agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION
        except Exception:
            agent_type = list(AgentType)[0]
    print('Using agent type:', agent_type)

    agent = initialize_agent(tools=[weather_tool], llm=llm, agent=agent_type, verbose=False)
    print('Agent initialized OK')

    res = agent.run('Do I need an umbrella in San Francisco today?')
    print('Agent run result:', res)

except Exception as e:
    print('ERROR:', type(e).__name__, e)
    traceback.print_exc()
