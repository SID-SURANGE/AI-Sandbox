
from chromadb import PersistentClient

client = PersistentClient(path="./chromadb")
collection = client.get_or_create_collection(name="multimodal_data_new")


def inspect_chromadb_contents():
    """Debug function to inspect ChromaDB contents."""
    print("\nInspecting ChromaDB contents:")
    try:
        # Get all items
        results = collection.get(include=["metadatas", "documents", "embeddings"])

        # Print summary
        print(f"\nTotal items in ChromaDB: {len(results['ids'])}")

        # Count by type
        type_counts = {}
        for metadata in results["metadatas"]:
            type_counts[metadata["type"]] = type_counts.get(metadata["type"], 0) + 1

        print("\nCounts by type:")
        for type_name, count in type_counts.items():
            print(f"{type_name}: {count}")

        # Print sample items
        print("\nSample items:")
        for i in range(min(3, len(results["ids"]))):
            print(f"\nItem {i+1}:")
            print(f"ID: {results['ids'][i]}")
            print(f"Type: {results['metadatas'][i]['type']}")
            print(f"Document: {results['documents'][i][:100]}...")

    except Exception as e:
        print(f"Error inspecting ChromaDB: {e}")

inspect_chromadb_contents()