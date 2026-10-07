from src.classification.classifier import classify_complaint
from src.generation.generator import generate_resolution_steps
from src.retrieval.retriever import retrieve_relevant_records


def main():
    new_complaint = "I can receive calls and send texts, but mobile internet stopped immediately after an eSIM replacement. Connected hotspot devices cannot get reliable internet or are far slower than the phone itself. I'm frustrated that this problem is still interfering with normal use."

    print("\nNew Complaint:")
    print(new_complaint)
    
    choices = classify_complaint(new_complaint)

    print(f"\nPRODUCT: {choices["product"].choice} (confidence: {choices['product'].confidence})")
    print(f"CATEGORY: {choices['category'].choice} (confidence: {choices['category'].confidence})")
    print(f"SEVERITY: {choices['severity'].choice} (confidence: {choices['severity'].confidence})")
    print(f"CUSTOMER SENTIMENT: {choices['customer_sentiment'].choice} (confidence: {choices['customer_sentiment'].confidence})")

    (retrieved_kbs, retrieved_tickets) = retrieve_relevant_records(new_complaint)

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

    print(f"Referring to: {', '.join(references)}")
    content = generate_resolution_steps(new_complaint, retrieved_kbs,retrieved_tickets)

    print("\nResolution Steps:")
    for i, step in enumerate(content["resolution_steps"], start=1):
        print(f"{i}. {step}")

if __name__ == "__main__":
    main()