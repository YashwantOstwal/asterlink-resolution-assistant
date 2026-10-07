from vector_store import pc, embeddings

from langchain_pinecone import PineconeVectorStore

def retrieve_relevant_records(complaint: str):

    kb_vector_store = PineconeVectorStore(
        index=pc.Index("asterlink-knowledge-base"),
        embedding=embeddings,
    )

    ticket_vector_store = PineconeVectorStore(
        index=pc.Index("asterlink-ticket-archive"),
        embedding=embeddings,
    )

    complaint_vector = embeddings.embed_query(complaint)
    retrieved_kbs = kb_vector_store.similarity_search_by_vector(complaint_vector, k=2)
    retrieved_tickets = ticket_vector_store.similarity_search_by_vector(complaint_vector, k=3)
    return (retrieved_kbs,retrieved_tickets)
