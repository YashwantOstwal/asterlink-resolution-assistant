import json

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import instructions

from src.ingestion.document_parser import parse_docx, parse_pdf
from vector_store import (
    create_kb_record_id,
    create_or_upsert_documents_to_vector_db,
)

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)


def upsert_knowledge_base(
    composites,
    file_name,
    dir_name,
    index_name,
):
    valid_kbs = []

    for composite in composites:
        text = composite.text
        page_number = composite.metadata.page_number

        response = llm.invoke([
            SystemMessage(
                content=instructions.SYSTEM_PROMPT_KB
            ),
            HumanMessage(
                content=f"""
PAGE NUMBER: {page_number}

{text}
""".strip()
            ),
        ])

        try:
            kb = json.loads(response.content)
        except:
            continue

        if kb.get("is_valid_kb") is True:
            valid_kbs.append(kb)

    kb_documents = []
    kb_ids = []

    for kb in valid_kbs:
        kb_id = kb["kb_id"]

        record_id = create_kb_record_id(
            file_name,
            kb_id,
        )

        kb_documents.append(
            Document(
                page_content=(
                    f"{kb_id} - {kb.get('title', '')}\n"
                    f"{kb.get('description', '')}"
                ),
                metadata={
                    "record_id": record_id,
                    "kb_id": kb_id,
                    "title": kb.get("title", ""),
                    "resolution_steps": json.dumps(
                        kb.get("resolution_steps", [])
                    ),
                    "file_name": file_name,
                    "dir_name": dir_name,
                    "page_number": kb.get("page_number"),
                },
            )
        )

        kb_ids.append(record_id)
    create_or_upsert_documents_to_vector_db(
        documents=kb_documents,
        ids=kb_ids,
        index_name=index_name,
    )


def upsert_ticket_archive(
    composites,
    file_name,
    dir_name,
    index_name,
):
    valid_tickets = []

    for composite in composites:
        text = composite.text

        response = llm.invoke([
            SystemMessage(
                content=instructions.SYSTEM_PROMPT_TICKET
            ),
            HumanMessage(content=text),
        ])

        try:
            ticket = json.loads(response.content)
        except:
            continue

        if ticket.get("is_valid_ticket") is True:
            valid_tickets.append(ticket)

    ticket_documents = []
    ticket_ids = []

    for ticket in valid_tickets:
        ticket_id = ticket["ticket_id"]

        ticket_documents.append(
            Document(
                page_content=(
                    f"{ticket_id} - "
                    f"{ticket.get('complaint', '')}"
                ),
                metadata={
                    "ticket_id": ticket_id,
                    "resolution_steps": json.dumps(
                        ticket.get("resolution_steps", [])
                    ),
                    "file_name": file_name,
                    "dir_name": dir_name,
                    "product": ticket.get("product", ""),
                    "category": ticket.get("category", ""),
                    "customer_sentiment": ticket.get(
                        "customer_sentiment", ""
                    ),
                    "severity": ticket.get("severity", ""),
                },
            )
        )

        ticket_ids.append(ticket_id)

    create_or_upsert_documents_to_vector_db(
        documents=ticket_documents,
        ids=ticket_ids,
        index_name=index_name,
    )


def ingest_documents():
    src_documents_dir = "docs"

    kb_file_name = "asterlink_knowledge_base.pdf"
    ticket_archive_file_name = "asterlink_ticket_archive.docx"

    kb_path = f"{src_documents_dir}/{kb_file_name}"
    ticket_archive_path = (
        f"{src_documents_dir}/{ticket_archive_file_name}"
    )

    kb_index_name = "asterlink-knowledge-base"
    ticket_index_name = "asterlink-ticket-archive"

    kb_composites = parse_pdf(
        file_path=kb_path
    )

    upsert_knowledge_base(
        composites=kb_composites,
        file_name=kb_file_name,
        dir_name=src_documents_dir,
        index_name=kb_index_name,
    )

    ticket_composites = parse_docx(
        file_path=ticket_archive_path
    )

    upsert_ticket_archive(
        composites=ticket_composites,
        file_name=ticket_archive_file_name,
        dir_name=src_documents_dir,
        index_name=ticket_index_name,
    )