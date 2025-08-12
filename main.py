# IMPORTS
import os
import re
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool
from agents.run import RunConfig
from pydantic import BaseModel
from typing import Optional

# USER CONTEXT
class UserContext(BaseModel):
    name: str = None
    account_type: str = None
    issue_type: str = None
    balance: float = 0.0
    loan_status: str = None
    requested_amount: Optional[float] = None

# CONFIGURATION
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

# INPUT GUARDRAILS
BANNED_INPUT_PATTERNS = [
    r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
]

def validate_input(prompt: str) -> Optional[str]:
    """
    Return error message string if invalid, otherwise None.
    """
    if not prompt or not prompt.strip():
        return "Input is empty. Please type your request."
    if len(prompt) < 3:
        return "Please write a bit more — one-word inputs are usually unclear."
    if len(prompt) > 1000:
        return "Input too long. Keep it short and focused."
    for pat in BANNED_INPUT_PATTERNS:
        if re.search(pat, prompt):
            return "Input contains invalid characters."
    return None

def extract_amount(prompt: str) -> Optional[float]:
    """
    Try to extract the first monetary amount (e.g., 100, $100, 10.50).
    """
    m = re.search(r"\$?\s*([0-9]+(?:\.[0-9]{1,2})?)", prompt.replace(",", ""))
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None

# OUTPUT GUARDRAILS
BANNED_WORDS = ["sorry"]

def guard_output(text: str) -> str:
    """
    Remove/replace banned words and enforce a max length for safety.
    """
    out = text
    for w in BANNED_WORDS:
        out = re.sub(rf"(?i)\b{re.escape(w)}\b", "[redacted]", out)
    # enforce length limit
    if len(out) > 1200:
        out = out[:1196] + "..."
    return out.strip()

# FUNCTION TOOLS
@function_tool
async def check_balance_tool(context: UserContext):
    """Return the user's balance (always enabled)."""
    
    return f"✅ {context.name or 'User'}, your balance is ${context.balance:,.2f}."

check_balance_tool.is_enabled = lambda tool, context: True

@function_tool
async def request_loan_tool(context: UserContext):
    """Place a loan request. We'll use context.requested_amount for the requested loan amount."""

    amt = getattr(context, "requested_amount", None)
    if not amt:
        return "Please specify the loan amount."
    
    context.loan_status = "pending"
    return f"🔔 Loan request for ${amt:,.2f} recorded and set to pending."

request_loan_tool.is_enabled = lambda tool, context: getattr(context, "issue_type", None) == "loan"

@function_tool
async def approve_loan_tool(context: UserContext):
    """Approve the loan — this tool is gated: only premium users can auto-approve."""

    amt = getattr(context, "requested_amount", None)
    if not amt:
        return "No requested amount found."

    context.loan_status = "approved"

    context.balance += float(amt)
    return f"✅ Loan for ${amt:,.2f} approved and credited to your account."

approve_loan_tool.is_enabled = lambda tool, context: getattr(context, "account_type", "") == "premium" and getattr(context, "issue_type", "") == "loan"

# AGENTS
triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are a routing bot. Read the user message and return EXACTLY one word: "
        "'account', 'loan' or 'general'. Do not add anything else."
    ),
    model=model_gemini,
)

account_agent = Agent(
    name="Account Agent",
    instructions="You are an account specialist. Suggest balance checks, transfers, or bill pay. Use tools when appropriate.",
    model=model_gemini,
    tools=[check_balance_tool],
)

loan_agent = Agent(
    name="Loan Agent",
    instructions="You are a loan specialist. Help users request loans and advise on approval. Use tools when appropriate.",
    model=model_gemini,
    tools=[request_loan_tool, approve_loan_tool],
)

general_agent = Agent(
    name="General Agent",
    instructions="You handle general bank inquiries and small talk.",
    model=model_gemini,
)

# HANDLE QUERY
async def handle_query(prompt: str, context: UserContext):
    invalid = validate_input(prompt)
    if invalid:
        return type("R", (), {"final_output": invalid})

    amt = extract_amount(prompt)
    if amt:
        context.requested_amount = amt

    triage_result = await Runner.run(
        triage_agent,
        prompt,
        run_config=config,
        context=context
    )
    category = (triage_result.final_output or "").strip().lower()
    if category not in {"account", "loan", "general"}:
        category = "general"

    context.issue_type = category

    if category == "account":
        res = await Runner.run(account_agent, prompt, run_config=config, context=context)
    elif category == "loan":
        res = await Runner.run(loan_agent, prompt, run_config=config, context=context)
    else:
        res = await Runner.run(general_agent, prompt, run_config=config, context=context)

    out = getattr(res, "final_output", "") or ""
    out = guard_output(out)
    return type("R", (), {"final_output": out})


# MAIN FUNCTIONS
async def run_loop():
    user_context = UserContext(
        name="Omer",
        account_type="premium",
        balance=5400.00,
        loan_status="none"
    )

    print("AI Bot: 👋 Welcome to the Bank Agent System!")
    print("Type 'quit' or 'exit' to stop. Try: 'check balance', 'request loan 5000', 'how do i apply for loan?'")

    while True:
        try:
            prompt = input("\nPrompt: ").strip()
            if not prompt:
                continue
            if prompt.lower() in {"quit", "exit"}:
                print("AI Bot: 👋 Goodbye.")
                break

            result = await handle_query(prompt, user_context)
            print("\nAI Bot:", result.final_output)

            print(f"[context] issue_type={user_context.issue_type}, loan_status={user_context.loan_status}, balance={user_context.balance:,.2f}")

        except KeyboardInterrupt:
            print("\nAI Bot: 👋 Exiting.")
            break

        except Exception as e:
            print("⚠️ Error:", e)

if __name__ == "__main__":
    asyncio.run(run_loop())