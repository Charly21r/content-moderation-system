import os
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from src.serving.model_manager import get_model_info, is_loaded, load_model, predict
from src.serving.schemas import ModerationResult, TextInput
from src.serving.metrics import (
    INFERENCE_LATENCY, MODEL_CONFIDENCE,
    PREDICTION_DISTRIBUTION, REQUEST_COUNT, REQUEST_LATENCY,
)


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
    with REQUEST_LATENCY.labels(endpoint='/v1/moderate/text').time():
        start = perf_counter()
        result = predict(text.content)
        end = perf_counter()
        processing_time_ms = end - start
        INFERENCE_LATENCY.observe(processing_time_ms)

        # Mark it as unsafe if any of the labels is flagged
        safe = not (result[0].flagged or result[1].flagged)

        # Record prediction distribution
        if result[0].flagged:
            PREDICTION_DISTRIBUTION.labels(label="toxic").inc()
        if result[1].flagged:
            PREDICTION_DISTRIBUTION.labels(label="hate").inc()
        if safe:
            PREDICTION_DISTRIBUTION.labels(label="safe").inc()

        # Record confidence scores
        MODEL_CONFIDENCE.labels(label="toxicity").observe(result[0].prob)
        MODEL_CONFIDENCE.labels(label="hate").observe(result[1].prob)
    
    REQUEST_COUNT.labels(endpoint="/v1/moderate/text", status_code="200").inc()

    output = ModerationResult(
        id=text.id,  # re-using the same id?
        text=text.content,
        toxicity=result[0],
        hate=result[1],
        safe=safe,
        processing_time_ms=processing_time_ms,
    )

    return output


# @app.post("/v1/moderate/text/batch")
# TODO: implement

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


