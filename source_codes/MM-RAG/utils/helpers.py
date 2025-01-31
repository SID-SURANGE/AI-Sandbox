import numpy as np

# --- Utility Function to Adjust Embedding Dimensions ---
def adjust_embedding_dimension(embedding, target_dim):
    """
    Adjusts the dimensionality of an embedding by padding or truncating.

    Args:
        embedding (list or np.ndarray): The original embedding.
        target_dim (int): The target dimensionality.

    Returns:
        np.ndarray: The adjusted embedding.
    """
    embedding = np.array(embedding)
    current_dim = len(embedding)

    if current_dim > target_dim:
        # Truncate if the current dimension is greater than the target
        return embedding[:target_dim]
    elif current_dim < target_dim:
        # Pad with zeros if the current dimension is less than the target
        return np.pad(embedding, (0, target_dim - current_dim))
    return embedding
