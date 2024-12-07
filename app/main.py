from fastapi import FastAPI, HTTPException
from typing import List
from app.models import TableSearchResponse, TableSearchRequest
from app.services.es_service import ESService
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="表搜索服务")

# 创建ES服务实例
es_service = ESService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/tables/search", response_model=List[TableSearchResponse])
async def search_tables(search_request: TableSearchRequest):
    try:
        results = es_service.search_tables(search_request)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 