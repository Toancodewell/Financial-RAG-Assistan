from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool
import logging

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Application Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Loads the RAG pipeline once during startup.
    """
    logger.info("Financial RAG API is starting up...")

    try:
        # Import and initialize RAG pipeline
        from rag_pipeline import ask_question
        app.state.ask_question = ask_question
        logger.info("RAG pipeline loaded successfully.")

    except Exception as e:
        logger.exception("Failed to load RAG pipeline.")
        raise e

    yield

    logger.info("Financial RAG API is shutting down...")


# FastAPI App Initialization
app = FastAPI(
    title="Financial RAG API",
    description="API for querying Samsung financial data using Llama 3 and RAG architecture.",
    version="1.0.0",
    lifespan=lifespan
)
# Request & Response Schemas
class QueryRequest(BaseModel):
    question: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What was Samsung's revenue in 2024?"
            }
        }
    }


class QueryResponse(BaseModel):
    answer: str

# API Endpoints
@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """
    Accepts a financial question and returns an answer
    generated using Retrieval-Augmented Generation (RAG).
    """

    try:
        # Run synchronous RAG pipeline in a threadpool
        answer = await run_in_threadpool(
            app.state.ask_question,
            request.question
        )

        return QueryResponse(answer=answer)

    except Exception:
        logger.exception("Error occurred while processing the request.")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify that the API is running.
    """
    return {"status": "healthy"}
