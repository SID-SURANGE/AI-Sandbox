# local imports
from src.model import run_llm
from src.retriever import query_db

TOP_RESULTS = 5

def get_contextualized_llm_response(query: str):
    try:
        # STEP 1: Query the database and fetch relevant information
        print("\nStarting query...")
                
        results = query_db(query, n_results=TOP_RESULTS)
        context = ""
        print("\nQuery Results:")
        for result in results:
            print(f"\nSource Type: {result['type']}")
            print(f"File Name: {result['file_name']}")
            print(f"Similarity Score: {result['similarity_score']:.3f}")
            if result["type"] == "image":
                print("[Image content cannot be displayed as text]")
            elif result["type"] == "frame":
                print(f"Frame Timestamp: {result['timestamp']}s")
            else:
                print(f"Content: {result['document'][:500]}...")
                context += result['document']

        # STEP 2: Use the retrieved data and curate a contextualized response
        llm_resp = run_llm(query, context)
        return llm_resp
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        print(traceback.format_exc())