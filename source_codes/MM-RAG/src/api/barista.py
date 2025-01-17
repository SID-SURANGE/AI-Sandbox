# Standard library imports

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
def process_barista_query(user_query: Query) -> JSONResponse:
    """
    print(f'user query: {uquery}')
        query (Query): The query object containing the user query string.

    Returns:
        JSONResponse: The response containing the result of the query.
    """
    uquery = user_query.query
    print(f"user query: {uquery}")
    if uquery is None:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # fetch relevant response for the query
    response = get_contextualized_llm_response(uquery)

    return JSONResponse(content={"response": response})
