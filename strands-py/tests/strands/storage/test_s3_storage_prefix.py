"""Storage prefixes identify the same S3 namespace with or without a trailing slash."""

import boto3
import pytest
from moto import mock_aws

from strands.storage import S3Storage


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["agents/", "agents//", "/agents///"])
async def test_trailing_slash_prefix_shares_s3_namespace(prefix):
    """Read, list and delete reach objects written with an unsuffixed prefix."""
    with mock_aws():
        session = boto3.Session(region_name="us-east-1")
        client = session.client("s3")
        client.create_bucket(Bucket="test-prefix")
        canonical = S3Storage("test-prefix", prefix="agents", boto_session=session)
        slash = S3Storage("test-prefix", prefix=prefix, boto_session=session)
        await canonical.write("first", b"one")

        assert await slash.read("first") == b"one"
        assert await slash.list("") == ["first"]
        await slash.write("second", b"two")
        assert client.get_object(Bucket="test-prefix", Key="agents/second")["Body"].read() == b"two"
        assert await canonical.list("") == ["first", "second"]
        await slash.delete("first")
        assert await canonical.read("first") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["", "/", "///"])
async def test_empty_prefix_uses_bucket_root(prefix):
    """An empty normalized prefix does not add a separator."""
    with mock_aws():
        session = boto3.Session(region_name="us-east-1")
        client = session.client("s3")
        client.create_bucket(Bucket="test-prefix")
        storage = S3Storage("test-prefix", prefix=prefix, boto_session=session)
        await storage.write("first", b"one")
        assert client.get_object(Bucket="test-prefix", Key="first")["Body"].read() == b"one"
        assert await storage.list("") == ["first"]
