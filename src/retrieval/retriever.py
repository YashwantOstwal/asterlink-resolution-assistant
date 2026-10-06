from vector_store import pc, embeddings

from langchain_pinecone import PineconeVectorStore

TOP_K = 3

KB_SCORE_THRESHOLD = 0.75
TICKET_SCORE_THRESHOLD = 0.77

def retrieve_relevant_records(complaint: str):

    kb_vector_store = PineconeVectorStore(
        index=pc.Index("asterlink-knowledge-base"),
        embedding=embeddings,
    )

    ticket_vector_store = PineconeVectorStore(
        index=pc.Index("asterlink-ticket-archive"),
        embedding=embeddings,
    )

    kb_retriever = kb_vector_store.as_retriever(
        # search_type="similarity_score_threshold",
        search_kwargs={
            "k": 2,
            # "score_threshold": KB_SCORE_THRESHOLD,
        },
    )

    ticket_retriever = ticket_vector_store.as_retriever(
        # search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            # "score_threshold": TICKET_SCORE_THRESHOLD,
        },
    )
    retrieved_kbs = kb_retriever.invoke(
        complaint
    )
    retrieved_tickets = ticket_retriever.invoke(
        complaint
    )
    return (retrieved_kbs,retrieved_tickets)
