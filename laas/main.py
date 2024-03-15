# API model ================
from enum import Enum
from typing import List

from pydantic import BaseModel


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"


class MessageType(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class HistoryRecord(BaseModel):
    role: MessageType
    content: str


class EngineInput(BaseModel):
    history: List[HistoryRecord]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "history": [
                        {"role": "system", "content": "You are nice assistant. Be nice."},
                        {"role": "user", "content": "How are you"},
                    ]
                }
            ]
        }
    }






# Config ================================================================================
CHECKPOINT_DIR = "/home/ec2-user/git/llama/llama-2-7b-chat/"
TOKENIZER_PATH = "/home/ec2-user/git/llama/tokenizer.model"
MAX_SEQ_LEN = 512
MAX_BATCH_SIZE = 8
MODEL_PARALLEL_SIZE = 1

TEMPERATURE = 0.6
TOP_P = 0.9
MAX_GEN_LEN = 64
HOST = "0.0.0.0"
PORT = 8080
    

# Engine ========================================================================
from typing import Annotated

from fastapi import Depends
from llama import Llama

MODEL = None


def init_model():
    global MODEL
    if not MODEL:
        print("loading model")
        MODEL = Llama.build(
            ckpt_dir=CHECKPOINT_DIR,
            tokenizer_path=TOKENIZER_PATH,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=MAX_BATCH_SIZE,
            model_parallel_size=MODEL_PARALLEL_SIZE,
        )
        print("model loaded")


def preprocess_message(engine_input: EngineInput) -> list[dict]:
    return engine_input.model_dump()["history"]


def process_message(dialog: Annotated[list[dict], Depends(preprocess_message)]) -> str:
    results = MODEL.chat_completion(
        [dialog],
        max_gen_len=MAX_GEN_LEN,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return results[0]["generation"]["content"].strip()


# Main ================================================================================

"""Main entrypoint for the app."""


from typing import Annotated

from fastapi import Depends, FastAPI, status

# from laas import api_models, config, engine

app = FastAPI()


@app.on_event("startup")
async def startup():
    init_model()


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    """
    ## Perform a Health Check
    Endpoint to perform a healthcheck on. This endpoint can primarily be used Docker
    to ensure a robust container orchestration and management is in place. Other
    services which rely on proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")


@app.post("/process_message")
async def process_message(result: Annotated[str, Depends(process_message)]) -> str:
    """
    ## Process message
    Endpoint to generate assistant answer to the chat history.
    Returns:
        str: Returns a string with generated answer by LLM
    """
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
