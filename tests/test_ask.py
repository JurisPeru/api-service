import json
import pytest
from typing import Any
from contextlib import contextmanager

from fastapi.testclient import TestClient

from app.deps import get_rag_service
from app.schemas.ask import AskResponse
from app.main import app


class FakeRag:
    def __init__(self):
        self.request = None
        self.retrieval_type = None

    def run_rag_pipeline_stream(self, ask_request, retrieval_type) -> Any:
        self.request = ask_request
        self.retrieval_type = retrieval_type

        def sse(data: AskResponse) -> bytes:
            return f"{json.dumps(data.model_dump())}\n\n".encode("utf-8")

        yield sse(AskResponse(stage="tok", data=""))
        yield sse(AskResponse(stage="tok", data="Hello"))
        yield sse(AskResponse(stage="tok", data="Hello World"))
        yield sse(
            AskResponse(
                stage="end",
                data="Hello World!",
                contexts=[
                    {
                        "document": "Document 1",
                        "source": "hello_world.pdf",
                        "page": 2,
                        "total_pages": 40,
                        "score": 0.9,
                        "text": "Hello World!",
                    },
                    {
                        "document": "Document 2",
                        "source": "hello_world.pdf",
                        "page": 9,
                        "total_pages": 40,
                        "score": 0.7,
                        "text": "Hello Mini World!",
                    },
                ],
            )
        )


class FakeRagValueError:
    def run_rag_pipeline_stream(self, ask_request, retrieval_type):
        _ = (ask_request, retrieval_type)
        raise ValueError("boom!")


class FakeRagException:
    def run_rag_pipeline_stream(self, ask_request, retrieval_type):
        _ = (ask_request, retrieval_type)
        raise RuntimeError("kaboom!")


@contextmanager
def override_rag_dependencie(fake):
    app.dependency_overrides[get_rag_service] = lambda: fake
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_rag_service, None)


def _iter_stream(resp):
    """
    Iterate over a streaming HTTP response, yielding parsed JSON objects.
    """
    try:
        raw_chunk = next(resp.iter_raw())
    except StopIteration:
        return
    except Exception as exc:
        raise RuntimeError(f"Failed to read from response stream: {exc}")

    try:
        decoded_data = raw_chunk.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Failed to decode response chunk: {exc}")

    for entry in decoded_data.split("\n\n"):
        entry = entry.strip()
        if not entry:
            continue
        try:
            yield json.loads(entry)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON from stream entry: {exc}\nEntry: {entry}")


##############################################
# Happy-path streaming tests
##############################################
@pytest.mark.parametrize(
    "payload",
    [
        {"query": "hi"},
    ],
)
def test_streaming_llm_happy_path(payload):
    client = TestClient(app)
    with client.stream("POST", "/api/ask", json=payload) as resp:
        assert resp.status_code == 200


@pytest.mark.parametrize(
    "payload,expect_k,expect_lang",
    [
        ({"query": "hola"}, 12, "spanish"),  # defaults
        (
            {"query": "hi", "k": 5, "temperature": 0.2, "language": "english"},
            5,
            "english",
        ),
        ({"query": "auto lang", "language": "auto"}, 12, "auto"),
    ],
)
def test_streaming_happy_path(payload, expect_k, expect_lang):
    fake_rag = FakeRag()
    with override_rag_dependencie(fake_rag):
        client = TestClient(app)
        with client.stream("POST", "/api/ask", json=payload) as resp:
            assert resp.status_code == 200
            events = [re for re in _iter_stream(resp)]

            assert events[0]["stage"] == "tok"
            assert events[1]["stage"] == "tok"
            assert events[2]["stage"] == "tok"
            assert events[3]["stage"] == "end"
            assert events[3]["data"] == "Hello World!"
            assert isinstance(events[3].get("contexts"), list)

            # Request validation
            assert fake_rag.retrieval_type == "mmr"
            assert getattr(fake_rag.request, "k", None) == expect_k
            assert getattr(fake_rag.request, "language", None) == expect_lang


##############################################
# Validation (422) tests
##############################################
@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing query
        {"query": None},  # invalid type
        {"query": "hi", "language": "german"},  # invalid name
        {"query": "hi", "k": "auto"},  # invalid k
    ],
)
def test_validation_errors_422(payload):
    fake_rag = FakeRag()
    with override_rag_dependencie(fake_rag):
        client = TestClient(app)
        resp = client.post("/api/ask", json=payload)
        assert resp.status_code == 422


##############################################
# Error handling (500) tests
##############################################
@pytest.mark.parametrize(
    "payload,fake,expected_detail",
    [
        ({"query": "hi"}, FakeRagValueError(), "boom!"),
        ({"query": "hi"}, FakeRagException(), "kaboom!"),
    ],
)
def test_internal_errors(payload, fake, expected_detail):
    with override_rag_dependencie(fake):
        client = TestClient(app)
        resp = client.post("/api/ask", json=payload)
        assert resp.status_code == 500
        data = resp.json()
        assert data.get("detail") == expected_detail
