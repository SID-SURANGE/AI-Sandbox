# local imports
from utils.config import collection, text_model, image_model, FIXED_DIMENSION
from utils.helpers import adjust_embedding_dimension

def query_db(query_text, n_results=10):
    """
    Enhanced query function handling all content types: text, image, frame, transcript, and pdf.

    Args:
        query_text (str): The user's query text
        n_results (int): Number of results to return

    Returns:
        List[Dict]: Ranked results with metadata
    """
    print(f"\nProcessing query: {query_text}")

    # 1. Determine query type and intent
    query_lower = query_text.lower()
    is_visual_query = any(
        term in query_lower
        for term in [
            "show",
            "look",
            "visual",
            "image",
            "video",
            "frame",
            "picture",
            "photo",
            "show me",
            "show me how to",
            "show me how to do",
            "show me how to make",
            "show me how to make a",
        ]
    )

    # 2. Generate query embedding based on type
    if is_visual_query:
        query_embedding = image_model.encode([query_text])[0]
    else:
        query_embedding = text_model.encode(query_text)

    adjusted_embedding = adjust_embedding_dimension(query_embedding, FIXED_DIMENSION)

    # 3. Query ChromaDB with increased initial results
    results = collection.query(
        query_embeddings=[adjusted_embedding],
        n_results=n_results * 5,
        include=["documents", "metadatas", "distances"],
    )

    print(f"\nRaw results count: {len(results['documents'][0])}")

    # 4. Process and format results with type-specific handling
    formatted_results = []
    for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        # Calculate normalized similarity score
        similarity_score = 1 - (distance / 2)

        # Get content type
        content_type = meta.get("type", "unknown")

        # Apply type-specific thresholds
        thresholds = {
            "text": 0.45,  # Lower threshold for text as it's more direct
            "image": 0.5,  # Medium threshold for images
            "frame": 0.5,  # Medium threshold for video frames
            "transcript": 0.4,  # Lower threshold for transcripts as they're more verbose
        }

        threshold = thresholds.get(content_type, 0.5)

        if similarity_score > threshold:
            result = {
                "document": doc,
                "type": content_type,
                "file_name": meta.get("file_name"),
                "similarity_score": similarity_score,
                "timestamp": meta.get("timestamp"),
            }

            # Add type-specific information
            if content_type == "frame":
                result["frame_info"] = f"Frame at {meta.get('timestamp', 0):.2f}s"
            elif content_type == "transcript":
                result["chunk_index"] = meta.get("chunk_index", 0)
            elif content_type == "image":
                result["image_path"] = (
                    doc  # For images, the document field contains the full file path that was stored during processing_image()
                )

            formatted_results.append(result)

    # 5. Sort results with type-specific boosting
    def sort_key(result):
        base_score = result["similarity_score"]
        content_type = result["type"]

        # Apply type-specific boosts based on query type
        if is_visual_query:
            # Boost visual content for visual queries
            if content_type in ["image", "frame"]:
                base_score *= 1.3  # Higher boost for visual content
            elif content_type in ["text", "transcript"]:
                base_score *= 0.9  # Slight penalty for text content
        else:
            # Boost text content for text queries
            if content_type in ["text", "transcript"]:
                base_score *= 1.3  # Higher boost for text content
            elif content_type in ["image", "frame"]:
                base_score *= 0.9  # Slight penalty for visual content

        return base_score

    print(f'Unsorted formatted results - {formatted_results}')
    formatted_results.sort(key=sort_key, reverse=True)
    print(f'Sorted formatted result - {formatted_results}')

    # 6. Ensure diverse results across all content types
    final_results = []
    type_counts = {
        'text': 0, 
        'image': 0, 
        'frame': 0, 
        'transcript': 0,
        'pdf': 0  # Added pdf type
    }
    
    # Calculate target count per type based on available types in results
    available_types = set(result['type'] for result in formatted_results)
    total_types = len(available_types)
    target_per_type = max(1, n_results // total_types) if total_types > 0 else n_results
    
    # First pass: try to get target number from each type
    for result in formatted_results:
        curr_type = result['type']
        # Initialize type count if not present
        if curr_type not in type_counts:
            type_counts[curr_type] = 0
            
        if type_counts[curr_type] < target_per_type:
            final_results.append(result)
            type_counts[curr_type] += 1
            
        if len(final_results) >= n_results:
            break
    
    # Second pass: fill remaining slots with best remaining results
    if len(final_results) < n_results:
        remaining = [r for r in formatted_results if r not in final_results]
        final_results.extend(remaining[: n_results - len(final_results)])

    print(f"\nReturning {len(final_results)} results")
    print("Result distribution:", type_counts)
    print(f'Final results {final_results}')
    return final_results