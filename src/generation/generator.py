import json
import instructions

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

def generate_resolution_steps(complaint, retrieved_kbs,retrieved_tickets):
    kb_context = []

    for kb in retrieved_kbs:
        kb_context.append(
            f"""
KB
kb_id: {kb.metadata.get("kb_id")}
content: {kb.page_content}
resolution_steps: {kb.metadata.get("resolution_steps")}
""".strip()
        )

    ticket_context = []

    for retrieved_ticket in retrieved_tickets:
        ticket_context.append(
            f"""
TICKET
ticket_id: {retrieved_ticket.metadata.get("ticket_id")}
complaint: {retrieved_ticket.page_content}
resolution_steps: {retrieved_ticket.metadata.get("resolution_steps")}
""".strip()
        )

    human_message = f"""
Complaint:
{complaint}

Knowledge bases:
{kb_context}

Similar tickets:
{ticket_context}

Use only this evidence.
Follow the system prompt exactly.
""".strip()

    response = llm.invoke(
        [
            SystemMessage(
                content=instructions.SYSTEM_PROMPT_GENERATION
            ),
            HumanMessage(
                content=human_message
            ),
        ]
    )
    return json.loads(response.content)

