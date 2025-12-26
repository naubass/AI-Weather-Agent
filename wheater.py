from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv
import requests
import os
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

def wheater_agent(city: str) -> str:
    """get wheater from openweathermap"""
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if api_key is None:
        raise ValueError("OPENWEATHER_API_KEY is not set")
    
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            humidity = data["main"]["humidity"]

            return f"Temperature: {temp}°C\nDescription: {desc}\nHumidity: {humidity}%"
        else:
            return f"Error: {data['message']}"
    except Exception as e:
        return f"Error: {str(e)}"

tools_map = {
    'wheater_agent': wheater_agent
}

agent = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_retries=5
)

llm_with_tools = agent.bind_tools([wheater_agent])

query = "What is the weather in Bandung right now?"
print(f"User Query: {query}")

messages = [HumanMessage(content=query)]

ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

if ai_msg.tool_calls:
    for tool_call in ai_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Jalankan fungsi python yang sebenarnya
        print(f"-> AI meminta tool '{tool_name}' untuk kota: {tool_args}")
        tool_function = tools_map[tool_name]
        tool_output = tool_function(**tool_args)
        
        print(f"Hasil Output: {tool_output}")

        tool_msg = ToolMessage(
            tool_call_id=tool_call["id"], 
            content=str(tool_output),
            name=tool_name
        )
        messages.append(tool_msg)

final_response = llm_with_tools.invoke(messages)

print("-" * 20)
print(f"Hasil Cuaca: {final_response.content}")
