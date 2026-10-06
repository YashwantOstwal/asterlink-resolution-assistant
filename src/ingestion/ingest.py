import json

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.ingestion.document_parser import parse_pdf, parse_docx
from vector_store import (
    create_kb_record_id,
    create_or_upsert_documents_to_vector_db,
)

import instructions

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

def ingest_documents():
   src_documents_dir = "docs"

   kb_file_name = "asterlink_knowledge_base.pdf"
   ticket_archive_file_name = "asterlink_ticket_archive.docx"

   kb_path = f"{src_documents_dir}/{kb_file_name}"
   ticket_archive_path = f"{src_documents_dir}/{ticket_archive_file_name}"

   kb_index_name = "asterlink-knowledge-base"
   ticket_index_name = "asterlink-ticket-archive"

   # Parse Knowledge Base
   kb_composites = parse_pdf(file_path=kb_path)

   # for i,composite in enumerate(kb_composites):
   #    print(composite.to_dict())
   valid_kbs = []

   for composite in kb_composites:
      text = composite.text
      page_number = composite.metadata.page_number
      response = llm.invoke([
            SystemMessage(content=instructions.SYSTEM_PROMPT_KB),
            HumanMessage(content=f"""
PAGE NUMBER: {page_number}

{text}
""")
      ])

      try:
         kb = json.loads(response.content)
      except:
         continue

      if (kb.get("is_valid_kb") is True):
            valid_kbs.append(kb)

   # Create KB Documents
   kb_documents = []
   kb_ids = []

   for kb in valid_kbs:
      kb_id = kb["kb_id"]

      record_id = create_kb_record_id(
            kb_file_name,
            kb_id
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
                  "file_name": kb_file_name,
                  "page_number": kb.get("page_number")
               }
            )
      )

      kb_ids.append(record_id)
   create_or_upsert_documents_to_vector_db(
      documents=kb_documents,
      ids=kb_ids,
      index_name=kb_index_name
   )
   
   # Parse Ticket Archive
   ticket_composites = parse_docx(file_path=ticket_archive_path)
   
   # for i,composite in enumerate(ticket_composites):
   #      print(composite.to_dict())
   valid_tickets = []
   for composite in ticket_composites[9:-1]:
      text = composite.text
      print(text)

      response = llm.invoke([
         SystemMessage(content=instructions.SYSTEM_PROMPT_TICKET),
         HumanMessage(content=text)
      ])
      print(response.content)
      try:
         ticket = json.loads(response.content)
      except:
         continue

      if (ticket.get("is_valid_ticket") is True):
         valid_tickets.append(ticket)

   # Create Ticket Documents  
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
                  "file_name": ticket_archive_file_name,
                  "product":ticket.get("product",""),
                  "category":ticket.get("category",""),
                  "customer_sentiment":ticket.get("customer_sentiment",""),
                  "severity":ticket.get("severity",""),
               }
         )
      )

      ticket_ids.append(ticket_id)

   create_or_upsert_documents_to_vector_db(
      documents=ticket_documents,
      ids=ticket_ids,
      index_name=ticket_index_name
   )
