from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_community.tools import DuckDuckGoSearchRun
from geopy.geocoders import Nominatim # Library baru untuk Map
from dotenv import load_dotenv
import requests
import os
import uvicorn

# Load Environment Variables
load_dotenv()

app = FastAPI(title="Free AI Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Weather Tools
def get_weather(city: str) -> str:
    """Cek cuaca saat ini berdasarkan nama kota."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key: return "Error: API Key Cuaca belum disetting."
    
    print(f"🌦️ Mengecek cuaca untuk: {city}")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            return (f"Laporan Cuaca {city}: {data['weather'][0]['description']}, "
                    f"Suhu: {data['main']['temp']}°C, Kelembaban: {data['main']['humidity']}%")
        return f"Gagal mengambil cuaca: {data.get('message')}"
    except Exception as e:
        return f"Error Cuaca: {str(e)}"

# DuckDuckGo Tools
ddg_search = DuckDuckGoSearchRun()
def internet_search(query: str) -> str:
    """Mencari informasi, tempat, berita, atau fakta di internet."""
    print(f"🌍 Searching DuckDuckGo: {query}")
    try:
        return ddg_search.run(query)
    except Exception as e:
        return f"Error Search: {str(e)}"

# Map Tools
geolocator = Nominatim(user_agent="my_ai_nexus_app")

def get_coordinates(location: str) -> str:
    """Mendapatkan koordinat peta (latitude/longitude) dari nama tempat/kota."""
    print(f"📍 Mencari Koordinat: {location}")
    try:
        loc = geolocator.geocode(location)
        if loc:
            return (f"Koordinat {location} ditemukan: Lat {loc.latitude}, Lon {loc.longitude}. "
                    f"INSTRUKSI PENTING: Di akhir jawabanmu, WAJIB sertakan tag ini persis: "
                    f"[MAP:{loc.latitude},{loc.longitude},{location}]")
        return "Lokasi tidak ditemukan di peta."
    except Exception as e:
        return f"Error Map: {str(e)}"

# Mapping Tools
tools_map = {
    'get_weather': get_weather,
    'internet_search': internet_search,
    'get_coordinates': get_coordinates # Daftarkan tool baru
}

# Inisialisasi Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Bind 3 Tools
llm_with_tools = llm.bind_tools([get_weather, internet_search, get_coordinates])

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

def parse_gemini_content(content) -> str:
    if isinstance(content, str): return content
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and 'text' in block:
                text_parts.append(block['text'])
            elif isinstance(block, str):
                text_parts.append(block)
        return " ".join(text_parts)
    else: return str(content)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        messages = [HumanMessage(content=request.message)]
        max_turns = 5
        
        for i in range(max_turns):
            print(f"🔄 Turn ke-{i+1}...")
            ai_msg = llm_with_tools.invoke(messages)
            
            if ai_msg.tool_calls:
                messages.append(ai_msg)
                print(f"🛠️ AI memanggil {len(ai_msg.tool_calls)} tools...")
                
                for tool_call in ai_msg.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    func = tools_map.get(tool_name)
                    output = "Tool tidak ditemukan."
                    
                    if func:
                        try:
                            if tool_name == 'get_weather':
                                output = func(**tool_args)
                            elif tool_name == 'get_coordinates': 
                                val = list(tool_args.values())[0] if tool_args else ""
                                output = func(val)
                            else:
                                val = list(tool_args.values())[0] if tool_args else ""
                                output = func(val)
                        except Exception as e:
                            output = f"Error Tool: {str(e)}"
                    
                    print(f"   -> Hasil {tool_name}: {str(output)[:50]}...")
                    
                    messages.append(ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=str(output),
                        name=tool_name
                    ))
            else:
                final_text = parse_gemini_content(ai_msg.content)
                print("✅ Jawaban Final Ditemukan.")
                return ChatResponse(reply=final_text)
        
        return ChatResponse(reply="Maaf, proses terlalu rumit.")

    except Exception as e:
        print(f"❌ Error System: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return FileResponse("index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)