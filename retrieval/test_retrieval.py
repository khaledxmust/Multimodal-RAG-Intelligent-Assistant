from qdrant_client.models import PointStruct

from retrieval.embeddings import EmbeddingModel
from retrieval.vector_store import (
    COLLECTION_NAME,
    get_qdrant_client,
)


def main():
    client = get_qdrant_client()
    embedding_model = EmbeddingModel()

    document_text = "Telecom Egypt offers internet services for home customers."

    document_vector = embedding_model.encode_documents(
        [document_text]
    )[0]

    point = PointStruct(
        id=1,
        vector=document_vector.tolist(),
        payload={
            "text": document_text,
            "source": "test_document",
        },
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[point],
    )

    print("Document inserted successfully.")

    query = "خدمات الانترنت المنزلي"

    query_vector = embedding_model.encode_queries(
        [query]
    )[0]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=3,
    ).points

    print("\nQuery:", query)

    for result in results:
        print("\nScore:", result.score)
        print("Text:", result.payload["text"])
        print("Source:", result.payload["source"])


if __name__ == "__main__":
    main()

