# Standard library imports
import logging
# Third party imports
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# Local application imports
from src.model import run_llm
from src.context_engine import get_contextualized_llm_response

# Create a router for the parser endpoints
router = APIRouter(tags=["barista"])


class Query(BaseModel):
    query: str


@router.post("/ask-barista-bot")
async def process_barista_query(user_query: Query) -> JSONResponse:
    """
    Process a user query and return a contextualized response.

    Args:
        user_query (Query): The query object containing the user's question.

    Returns:
        JSONResponse: The response containing the answer or error message.

    Raises:
        HTTPException: If the query is empty or invalid.
    """
    try:
        # Input validation
        query_text = user_query.query.strip()
        if not query_text:
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )

        # Log incoming query for debugging
        logging.info(f"Processing query: {query_text}")

        # Get response from LLM
        response = await get_contextualized_llm_response(query_text)
        
        return JSONResponse(
            content={
                "status": "success",
                "response": response,
                "data": {
                    "query": query_text,
                }
            },
            status_code=200
        )

    except HTTPException as he:
        # Re-raise HTTP exceptions for proper error handling
        raise he

    except Exception as e:
        # Log the full error for debugging
        logging.error(f"Error processing query: {str(e)}", exc_info=True)
        
        return JSONResponse(
            content={
                "status": "error",
                "response": "An error occurred while processing your request",
                "error_type": "internal_server_error"
            },
            status_code=500
        )
