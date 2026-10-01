import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model(
    model_name: str | None = None
) -> HuggingFaceEmbeddings:
    """
    Create a Hugging Face embedding model.

    Args:
        model_name (str | None): Optional Hugging Face model name.
            If omitted, the value from EMBEDDING_MODEL in .env is used.

    Returns:
        HuggingFaceEmbeddings: Ready-to-use embedding model.

    Raises:
        RuntimeError: If the embedding model cannot be created.

    Example:
        embeddings = get_embedding_model()
        vector = embeddings.embed_query("What is the refund policy?")
        print(len(vector))
    """

    selected_model = model_name or os.getenv(
        "EMBEDDING_MODEL",
        DEFAULT_EMBEDDING_MODEL
    )

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=selected_model
        )

    except Exception as exc:
        raise RuntimeError(
            "Failed to create the Hugging Face embedding model. "
            "Check EMBEDDING_MODEL and your internet connection."
        ) from exc

    return embeddings


def get_embedding_model_name() -> str:
    """
    Return the configured embedding model name for UI display.

    Returns:
        str: Hugging Face embedding model name currently configured.

    Example:
        print(get_embedding_model_name())
    """

    return os.getenv(
        "EMBEDDING_MODEL",
        DEFAULT_EMBEDDING_MODEL
    )