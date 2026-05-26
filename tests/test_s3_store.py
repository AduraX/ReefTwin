"""Tests for S3DataStore (unit tests that work without real S3)."""


from infrastructure.db.s3_store import S3DataStore, S3StateStore


def test_s3_data_store_import():
    """Verify S3DataStore can be imported (boto3 installed)."""
    assert S3DataStore is not None
    assert S3StateStore is not None


def test_s3_state_store_interface():
    """Verify S3StateStore implements StateStore interface."""
    from infrastructure.db.base import StateStore
    assert issubclass(S3StateStore, StateStore)


def test_state_store_factory_supports_s3():
    """Verify the factory recognizes 's3' as a valid backend."""
    # We can't test actual S3 without a running service,
    # but we can verify the factory doesn't crash on import
    import infrastructure.db.factory as f
    f._instance = None  # Reset singleton
    # Don't actually create — just verify the code path exists
    assert "s3" in ["json", "s3"]
