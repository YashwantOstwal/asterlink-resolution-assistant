from vector_store import pc, embeddings

from langchain_pinecone import PineconeVectorStore

kb_vector_store = PineconeVectorStore(
    index=pc.Index("asterlink-knowledge-base"),
    embedding=embeddings,
)

ticket_vector_store = PineconeVectorStore(
    index=pc.Index("asterlink-ticket-archive"),
    embedding=embeddings,
)

def retrieve_relevant_records(complaint: str):

    complaint_vector = embeddings.embed_query(complaint)
    retrieved_kbs = kb_vector_store.similarity_search_by_vector(complaint_vector, k=2)
    retrieved_tickets = ticket_vector_store.similarity_search_by_vector(complaint_vector, k=3)
    return (retrieved_kbs,retrieved_tickets)

def lookup_ticket(ticket_id):
    [doc] = ticket_vector_store.similarity_search("f{ticket_id}", k=1, filter = {"ticket_id":ticket_id })
    print(f"\n{doc.page_content}")

def lookup_knowledge_base(kb_id):
    [doc] = kb_vector_store.similarity_search("f{kb_id}", k=1, filter = {"kb_id":kb_id })
    print(f"\n{doc.page_content}")
