from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
)


# ============================================================
# CONFIG
# ============================================================

from config.settings import QDRANT_URL, COLLECTION_NAME

# A dedicated demo collection is selected in config.settings.

VECTOR_SIZE = 384


# ============================================================
# CLIENT
# ============================================================

def get_qdrant_client() -> QdrantClient:

    return QdrantClient(
        url=QDRANT_URL
    )


# ============================================================
# COLLECTION HELPERS
# ============================================================

def collection_exists(
    client: QdrantClient,
) -> bool:

    existing_collections = {
        collection.name
        for collection
        in client.get_collections().collections
    }

    return (
        COLLECTION_NAME
        in existing_collections
    )


def create_collection(
    client: QdrantClient,
):

    if collection_exists(
        client
    ):

        print(
            f"Collection "
            f"'{COLLECTION_NAME}' "
            f"already exists."
        )

        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print(
        f"Collection "
        f"'{COLLECTION_NAME}' "
        f"created."
    )


def reset_collection(
    client: QdrantClient,
):
    """
    Delete and recreate the TE knowledge collection.

    Used when rebuilding the complete corpus so stale/test
    vectors cannot remain in Qdrant.
    """

    if collection_exists(
        client
    ):

        print(
            f"Deleting old collection "
            f"'{COLLECTION_NAME}'..."
        )

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print(
        f"Collection "
        f"'{COLLECTION_NAME}' "
        f"created."
    )


def get_point_count(
    client: QdrantClient,
) -> int:

    if not collection_exists(
        client
    ):
        return 0

    collection_info = (
        client.get_collection(
            collection_name=COLLECTION_NAME
        )
    )

    return (
        collection_info.points_count
        or 0
    )


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Connecting to Qdrant..."
    )

    client = (
        get_qdrant_client()
    )

    print(
        "Connected successfully."
    )

    create_collection(
        client
    )

    print(
        "Collection:",
        COLLECTION_NAME,
    )

    print(
        "Points:",
        get_point_count(
            client
        ),
    )