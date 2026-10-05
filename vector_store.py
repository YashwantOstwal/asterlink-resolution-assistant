from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

pc = Pinecone()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

def create_or_upsert_documents_to_vector_db(
    documents: list[Document],
    ids: list[str],
    index_name: str,
):
    if len(documents) != len(ids):
        raise ValueError("documents and ids must have the same length")

    if not documents:
        return

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings
    )

    vector_store.add_documents(
        documents=documents,
        ids=ids
    )


def delete_from_vector_db(
    index_name: str,
    document_id: str,
):
    if not pc.has_index(index_name):
        raise ValueError(
            f"Index '{index_name}' does not exist"
        )

    pc.Index(index_name).delete(
        ids=[document_id]
    )

def create_kb_record_id(file_name: str, kb_id: str) -> str:
    return f"{file_name}-{kb_id}"