from fastapi import FastAPI

from app.api.search import router as search_router
from app.api.ingest_products import router as ingest_products_router
from app.api.ingest import router as ingest_router

app = FastAPI(title="Glancy Vector DB", version="0.1.0")
app.include_router(search_router)
app.include_router(ingest_products_router)
app.include_router(ingest_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
