import os
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI
from src.serving.model_manager import get_model_info, is_loaded, load_model, predict
from src.serving.schemas import ModerationResult, TextInput


@asynccontextmanager
async def lifespan(app):
    load_model(os.getenv("MODEL_PATH", "models/text_toxicity/artifacts/model"))
    yield
    # Cleaning


app = FastAPI(lifespan=lifespan)


@app.get("/v1/health")
def health():
    return {"is_loaded": is_loaded()}


@app.get("/v1/model/info")
def model_info():
    return get_model_info()


@app.post("/v1/moderate/text")
def moderate(text: TextInput) -> ModerationResult:
    content = text.content

    start = perf_counter()
    result = predict(content)
    end = perf_counter()
    processing_time_ms = end - start

    # Mark it as unsafe if any of the labels is flagged
    safe = not (result[0].flagged or result[1].flagged)

    output = ModerationResult(
        id=text.id,  # re-using the same id?
        text=content,
        toxicity=result[0],
        hate=result[1],
        safe=safe,
        processing_time_ms=processing_time_ms,
    )

    return output


# @app.post("/v1/moderate/text/batch")
# TODO: implement
