from pathlib import Path
from typing import Any

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from src.classification.classifier import classify_complaint
from src.generation.generator import generate_resolution_steps
from src.retrieval.retriever import retrieve_relevant_records,lookup_ticket,lookup_knowledge_base

from src.ingestion.ingest import upsert_knowledge_base,upsert_ticket_archive
from src.ingestion.document_parser import parse_pdf,parse_docx

from vector_store import delete_from_vector_db, create_kb_record_id

def process_new_complaint(complaint:str):

    print("\nNew Complaint:")
    print(complaint)
    
    choices = classify_complaint(complaint)

    print(f"\nPRODUCT: {choices["product"].choice} (confidence: {choices['product'].confidence})")
    print(f"CATEGORY: {choices['category'].choice} (confidence: {choices['category'].confidence})")
    print(f"SEVERITY: {choices['severity'].choice} (confidence: {choices['severity'].confidence})")
    print(f"CUSTOMER SENTIMENT: {choices['customer_sentiment'].choice} (confidence: {choices['customer_sentiment'].confidence})")

    (retrieved_kbs, retrieved_tickets) = retrieve_relevant_records(complaint)

    kbs = []
    tickets = []

    for retrieved_kb in retrieved_kbs:
        kbs.append({
            "kb_id": retrieved_kb.metadata.get("kb_id"),
            "file_name": retrieved_kb.metadata.get("file_name"),
            "page_number": retrieved_kb.metadata.get("page_number"),
            "record_id": retrieved_kb.metadata.get("record_id"),
        })

    for retrieved_ticket in retrieved_tickets:
        tickets.append({
            "ticket_id": retrieved_ticket.metadata.get("ticket_id"),
            "file_name": retrieved_ticket.metadata.get("file_name"),
            "page_number": retrieved_ticket.metadata.get("page_number"),
            "record_id": retrieved_ticket.metadata.get("record_id"),
        })
    references = (
        [kb["kb_id"] for kb in kbs]
        + [ticket["ticket_id"] for ticket in tickets]
    )

    print(f"\nCiting : {', '.join(references)}")
    content = generate_resolution_steps(complaint, retrieved_kbs,retrieved_tickets)

    print("\nResolution Steps:")
    for i, step in enumerate(content["resolution_steps"], start=1):
        print(f"{i}. {step}")
    print()
    render_reference_select(retrieved_kbs, retrieved_tickets)


def handle_upsert_knowledge_base(filepath: str):
    path = Path(filepath)

    print("\nParsing the file...")
    composites = parse_pdf(file_path=filepath)

    print("\nAdding to the asterlink-knowledge-base vector index...")
    upsert_knowledge_base(
        composites=composites,
        file_name=path.name,
        dir_name=str(path.parent),
        index_name="asterlink-knowledge-base",
    )

def handle_upsert_ticket_archive(filepath: str):
    path = Path(filepath)

    print("\nParsing the file...")
    composites = parse_docx(file_path=filepath)
    print("\nAdding to the asterlink-ticket-archive vector index...")

    upsert_ticket_archive(
        composites=composites,
        file_name=path.name,
        dir_name=str(path.parent),
        index_name="asterlink-ticket-archive",
    )

def ask_select(message: str, choices: list[Choice]) -> Any | None:
    try:
        return inquirer.select(
            message=message,
            choices=choices,
            cycle=False,
        ).execute()
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled.")
        return None

def ask_filepath(message: str) -> str | None:
    try:
        return inquirer.filepath(
            message=message,
            only_files=True,
        ).execute()
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled.")
        return None

def ask_text(message: str) -> str | None:
    try:
        return inquirer.text(
            message=message,
            validate=lambda value: bool(value.strip()),
            invalid_message="A value is required.",
        ).execute().strip()
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled.")
        return None


def ask_confirm(message: str,default = False) -> bool | None:
    try:
        return inquirer.confirm(message=message, default=default).execute()
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled.")
        return None


def handle_new_complaint():
    complaint = ask_text("Enter customer complaint:")
    if complaint is None:
        return

    print("\nProcessing complaint...")

    try:
        process_new_complaint(complaint)
    except NotImplementedError as exc:
        print(f"\n{exc}")
    except Exception as exc:
        print(f"\nUnable to process complaint: {exc}")

def handle_record_lookup() -> None:
    source = ask_select(
        "Select record source:",
        [
            Choice(value="kb", name="Knowledge Base"),
            Choice(value="ticket", name="Ticket Archive"),
        ],
    )
    record_id = ask_text(
        "Enter Knowledge Base record ID:"
        if source == "kb"
        else "Enter Ticket ID:"
    )
    if record_id is None:
        return

    try:
        lookup_knowledge_base(record_id) if source == "kb" else lookup_ticket(record_id)
    except NotImplementedError as exc:
        print(f"\n{exc}")
    except Exception as exc:
        print(f"\nUnable to look up record: {exc}")


def handle_upsert() -> None:
    destination = ask_select(
        "Select destination:",
        [
            Choice(value="kb", name="Upsert to Knowledge Base"),
            Choice(value="ticket", name="Upsert to Ticket Archive"),
        ],
    )
    filepath = ask_filepath(f"Enter file path: {"(only .pdf)" if destination == "kb" else "(only .docx)"}")
    if filepath is None:
        return

    path = Path(filepath).expanduser()
    if not path.is_file():
        print("\nFile not found.")
        return

    handle_upsert_knowledge_base(str(path)) if destination == "kb" else handle_upsert_ticket_archive(str(path))

    print("\nIngestion successful.")


def render_reference_select(kbs: list[dict], tickets: list[dict]):

    choices = []

    for i, kb in enumerate(kbs):
        choices.append(
            Choice(
                name=f'{kb.metadata["kb_id"]}, {kb.metadata["file_name"]}, (pg no. {int(kb.metadata["page_number"])})',
                value={
                    "type": "kb",
                    "index":i,
                },
            )
        )

    for i,ticket in enumerate(tickets):
        choices.append(
            Choice(
                name=f'{ticket.metadata["ticket_id"]}, {ticket.metadata["file_name"]}',
                value={
                    "type": "ticket",
                    "index": i,
                },
            )
        )

    while True:
        try:
            selected = inquirer.select(
                message="View citation:",
                choices=choices,
                cycle=True,
            ).execute()

            if selected is None:
                return

            if selected["type"] == "kb":
                print(f"\n{kbs[selected["index"]].page_content}\n")

            elif selected["type"] == "ticket":
                print(f"\n{tickets[selected["index"]].page_content}\n")


        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            return

def handle_delete() -> None:
    source = ask_select(
        "Select record source:",
        [
            Choice(value="kb", name="Delete from Knowledge Base"),
            Choice(value="ticket", name="Delete from Ticket Archive"),
        ],
    )

    if source == "kb":
        filepath = ask_filepath("Select Knowledge Base file:")

        if filepath is None:
            return

        file_name = Path(filepath).name

        kb_id = ask_text("Enter Knowledge Base ID:")

        if kb_id is None:
            return

        confirmed = ask_confirm(
            f"Delete {kb_id} from {file_name}?"
        )

        if confirmed is not True:
            if confirmed is False:
                print("\nDeletion cancelled.")
            return

        delete_from_vector_db(
            "asterlink-knowledge-base",
            create_kb_record_id(file_name,kb_id)
        )

        print("Deletion successful")
        return

    ticket_id = ask_text("Enter Ticket ID:")

    if ticket_id is None:
        return

    confirmed = ask_confirm(
        f"Delete {ticket_id} from the Ticket Archive?"
    )

    if confirmed is not True:
        if confirmed is False:
            print("\nDeletion cancelled.")
        return

    delete_from_vector_db("asterlink-ticket-archive",ticket_id)
    print("Deletion successful")


def show_main_menu() -> str | None:
    return ask_select(
        "What would you like to do?",
        [
            Choice(value="new_complaint", name="New Complaint"),
            Choice(value="lookup", name="Look Up Record"),
            Choice(value="upsert", name="Upsert Records"),
            Choice(value="delete", name="Delete Record"),
            Choice(value="exit", name="Exit"),
        ],
    )


def run_app() -> None:
    print("\nAsterLink Resolution Assistant:\n")

    handlers = {
        "new_complaint": handle_new_complaint,
        "lookup": handle_record_lookup,
        "upsert": handle_upsert,
        "delete": handle_delete,
    }

    while True:
        try:
            action = show_main_menu()

            if action is None or action == "exit":
                print("\nGoodbye.")
                return

            handlers[action]()
            print()

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            return
        except Exception as exc:
            print(f"\nUnexpected error: {exc}\n")
