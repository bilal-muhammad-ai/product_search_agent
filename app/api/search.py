from fastapi import APIRouter

from app.agents.search import format_labeled_text, rewrite_query
from app.database.retriever import retrieve, search_by_keyword, search_by_similarity
from app.models.search import SearchRequest, SearchResponse, SearchProductsResponse1
from logger import logger

router = APIRouter()


@router.post("/search", response_model=SearchProductsResponse1)
def search(request: SearchRequest) -> SearchProductsResponse1:
    message = request.message.strip()
    rewritten_query = rewrite_query(message)
    search_query = format_labeled_text(rewritten_query, message)
    logger.info(f"rewritten_query: {rewritten_query}")
    logger.info("search_query:\n%s", search_query)
    results = retrieve(search_query, rerank_text=message)
    #search_by_keyword(search_query)
    #search_by_similarity(search_query)

    return SearchProductsResponse1(
        count=len(results),
        next_after_id=None,
        products=results,
    )
