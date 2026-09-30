"""Structured Gemini failures expose the same exceptions as ordinary streaming."""

from unittest.mock import AsyncMock, Mock

import pytest
from google import genai
from pydantic import BaseModel

from strands.models.gemini import GeminiModel
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException


class Output(BaseModel):
    """Requested structured response."""

    value: str


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "status", "message", "exception_type"),
    [
        (429, "RESOURCE_EXHAUSTED", "Quota exceeded", ModelThrottledException),
        (503, "UNAVAILABLE", "Temporarily unavailable", ModelThrottledException),
        (429, "RESOURCE_EXHAUSTED", None, ModelThrottledException),
        (
            400,
            "INVALID_ARGUMENT",
            "Input exceeds the maximum number of tokens",
            ContextWindowOverflowException,
        ),
    ],
)
async def test_structured_output_normalizes_provider_failure(code, status, message, exception_type):
    """Provider failures keep the standard Strands type and the original cause."""
    error = genai.errors.ClientError(code, {"error": {"status": status, "message": message}})
    client = Mock()
    client.aio = AsyncMock()
    client.aio.models.generate_content.side_effect = error
    model = GeminiModel(client=client, model_id="gemini-2.5-flash")

    with pytest.raises(exception_type) as raised:
        _ = [event async for event in model.structured_output(Output, [{"role": "user", "content": [{"text": "hi"}]}])]

    assert raised.value.__cause__ is error
    assert str(raised.value)
    client.aio.models.generate_content.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["INVALID_ARGUMENT", "PERMISSION_DENIED"])
async def test_structured_output_preserves_unrelated_provider_failure(status):
    """Errors unrelated to limits are not rewritten."""
    error = genai.errors.ClientError(400, {"error": {"status": status, "message": "Bad request"}})
    client = Mock()
    client.aio = AsyncMock()
    client.aio.models.generate_content.side_effect = error
    model = GeminiModel(client=client, model_id="gemini-2.5-flash")

    with pytest.raises(genai.errors.ClientError) as raised:
        _ = [event async for event in model.structured_output(Output, [{"role": "user", "content": [{"text": "hi"}]}])]

    assert raised.value is error
