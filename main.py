import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool
from agents.run import RunConfig
from pydantic import BaseModel

# class UserContext(BaseModel):
#     name: Optional[str] = None
#     is_premium_user: bool = False
#     issue_type: Optional[str] = None 

# ENVIRONMENT VARIABLES & CONFIGURATION

load_dotenv()

set_tracing_disabled(disabled = True)

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model_gemini = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model_gemini,
    model_provider=external_client,
    tracing_disabled=True
)