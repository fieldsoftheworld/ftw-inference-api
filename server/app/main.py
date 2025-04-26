import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router as api_router
from app.core.config import get_settings
from app.db.database import create_db_and_tables

app = FastAPI(
    title=get_settings().api_title,
    description=get_settings().api_description,
    version=get_settings().api_version,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)


@app.on_event("startup")
async def startup_event():
    create_db_and_tables()


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=get_settings().host,
        port=get_settings().port,
        reload=get_settings().debug,
    )
