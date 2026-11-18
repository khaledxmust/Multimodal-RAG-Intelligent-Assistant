import gc
import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from config.hardware import choose_torch_device


load_dotenv()


from config.settings import EMBEDDING_MODEL, HF_LOCAL_FILES_ONLY

MODEL_NAME = EMBEDDING_MODEL

# This model is small enough that a modest GPU can be useful.
EMBEDDING_MIN_VRAM_GB = 1.0


class EmbeddingModel:
    def __init__(self):
        print("Initializing SentenceTransformer...")

        hardware = choose_torch_device(
            min_vram_gb=EMBEDDING_MIN_VRAM_GB,
        )

        preferred_device = hardware["device"]

        print(
            "Preferred embedding device:",
            preferred_device,
        )

        print(
            "Hardware decision:",
            hardware["reason"],
        )

        try:
            self.model = self._load_model(
                preferred_device
            )

            self.device = preferred_device

        except Exception as error:
            if preferred_device != "cuda":
                raise

            print()
            print(
                "Embedding GPU initialization failed:"
            )
            print(error)
            print(
                "Falling back to CPU..."
            )

            self.model = self._load_model(
                "cpu"
            )

            self.device = "cpu"

        print(
            "Embedding device:",
            self.device,
        )

        print(
            "SentenceTransformer initialized."
        )

    def _load_model(
        self,
        device: str,
    ) -> SentenceTransformer:

        return SentenceTransformer(
            MODEL_NAME,
            device=device,
            token=os.getenv("HF_TOKEN"),
            local_files_only=HF_LOCAL_FILES_ONLY,
        )

    def _fallback_to_cpu(
        self,
    ):
        """
        Replace the CUDA model with a CPU model after a
        runtime GPU failure.
        """

        print()
        print(
            "Embedding GPU inference failed."
        )

        print(
            "Reloading embedding model on CPU..."
        )

        old_model = self.model

        self.model = None

        del old_model

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception:
            pass

        self.model = self._load_model(
            "cpu"
        )

        self.device = "cpu"

        print(
            "Embedding CPU fallback ready."
        )

    def _encode(
        self,
        texts,
    ):
        try:
            return self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=True,
            )

        except Exception as error:
            if self.device != "cuda":
                raise

            print()
            print(
                "Embedding GPU inference error:"
            )
            print(error)

            self._fallback_to_cpu()

            return self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=True,
            )

    def encode_documents(
        self,
        texts,
    ):
        texts = [
            f"passage: {text}"
            for text in texts
        ]

        return self._encode(
            texts
        )

    def encode_queries(
        self,
        texts,
    ):
        texts = [
            f"query: {text}"
            for text in texts
        ]

        return self._encode(
            texts
        )


if __name__ == "__main__":
    model = EmbeddingModel()

    queries = [
        "How much does the internet package cost?",
        "سعر باقة الانترنت كام؟",
        "الباقة دي بكام؟",
    ]

    vectors = model.encode_queries(
        queries
    )

    print(
        "Shape:",
        vectors.shape,
    )

    similarities = (
        vectors @ vectors.T
    )

    print(
        "\nSimilarity matrix:"
    )

    print(
        similarities
    )