"""S3-compatible storage backend for ReefTwin data layers.

Supports SeaweedFS (KF4X), MinIO, AWS S3, and any S3-compatible service.
Maps the bronze/silver/gold medallion pattern to S3 prefixes.

Configuration via environment variables:
    REEFTWIN_S3_ENDPOINT    — S3 endpoint URL (e.g., http://seaweedfs-s3:8333)
    REEFTWIN_S3_BUCKET      — Bucket name (default: reeftwin)
    REEFTWIN_S3_ACCESS_KEY  — Access key ID
    REEFTWIN_S3_SECRET_KEY  — Secret access key
    REEFTWIN_S3_REGION      — AWS region (default: us-east-1)
    REEFTWIN_S3_USE_SSL     — Use HTTPS (default: false for local, true for AWS)
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from infrastructure.db.base import StateStore
from infrastructure.logging import get_logger

logger = get_logger("db.s3_store")


class S3DataStore:
    """S3-compatible storage for bronze/silver/gold data layers.

    Works with SeaweedFS (KF4X Phase 1), MinIO, or AWS S3.
    """

    def __init__(
        self,
        endpoint_url: str = "",
        bucket: str = "reeftwin",
        access_key: str = "",
        secret_key: str = "",
        region: str = "us-east-1",
        use_ssl: bool = False,
    ) -> None:
        import os
        try:
            import boto3
        except ImportError:
            raise ImportError("Install boto3: pip install 'reeftwin[s3]'")

        self._endpoint = endpoint_url or os.getenv("REEFTWIN_S3_ENDPOINT", "")
        self._bucket = bucket or os.getenv("REEFTWIN_S3_BUCKET", "reeftwin")
        ak = access_key or os.getenv("REEFTWIN_S3_ACCESS_KEY", "")
        sk = secret_key or os.getenv("REEFTWIN_S3_SECRET_KEY", "")
        region = region or os.getenv("REEFTWIN_S3_REGION", "us-east-1")
        use_ssl = use_ssl or os.getenv("REEFTWIN_S3_USE_SSL", "false").lower() == "true"

        client_kwargs: dict[str, Any] = {"region_name": region}
        if self._endpoint:
            client_kwargs["endpoint_url"] = self._endpoint
            client_kwargs["use_ssl"] = use_ssl
        if ak and sk:
            client_kwargs["aws_access_key_id"] = ak
            client_kwargs["aws_secret_access_key"] = sk

        self._s3 = boto3.client("s3", **client_kwargs)
        self._ensure_bucket()
        logger.info("S3 data store: bucket=%s endpoint=%s", self._bucket, self._endpoint or "AWS")

    def _ensure_bucket(self) -> None:
        try:
            self._s3.head_bucket(Bucket=self._bucket)
        except Exception:
            try:
                self._s3.create_bucket(Bucket=self._bucket)
                logger.info("Created S3 bucket: %s", self._bucket)
                # Enable versioning for data integrity (R-002, R-005)
                try:
                    self._s3.put_bucket_versioning(
                        Bucket=self._bucket,
                        VersioningConfiguration={"Status": "Enabled"},
                    )
                    logger.info("S3 versioning enabled for %s", self._bucket)
                except Exception as ve:
                    logger.warning("Could not enable S3 versioning: %s", ve)
            except Exception as e:
                logger.warning("Could not create bucket %s: %s", self._bucket, e)

    # --- Core operations ---

    def put_bytes(self, key: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=data)
        logger.debug("PUT s3://%s/%s (%d bytes)", self._bucket, key, len(data))

    def get_bytes(self, key: str) -> bytes:
        response = self._s3.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()

    def put_text(self, key: str, text: str) -> None:
        self.put_bytes(key, text.encode("utf-8"))

    def get_text(self, key: str) -> str:
        return self.get_bytes(key).decode("utf-8")

    def put_json(self, key: str, data: dict | list) -> None:
        self.put_text(key, json.dumps(data, indent=2))

    def get_json(self, key: str) -> dict | list:
        return json.loads(self.get_text(key))

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False

    def list_keys(self, prefix: str = "") -> list[str]:
        response = self._s3.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]

    def delete(self, key: str) -> None:
        self._s3.delete_object(Bucket=self._bucket, Key=key)

    # --- CSV helpers (bronze/silver/gold data layers) ---

    def put_csv(self, key: str, df: "pd.DataFrame") -> None:
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        self.put_bytes(key, buf.getvalue())
        logger.info("PUT CSV s3://%s/%s (%d rows)", self._bucket, key, len(df))

    def get_csv(self, key: str) -> "pd.DataFrame":
        import pandas as pd
        data = self.get_bytes(key)
        return pd.read_csv(io.BytesIO(data))

    # --- Medallion layer helpers ---

    def put_bronze(self, name: str, df: "pd.DataFrame") -> None:
        self.put_csv(f"bronze/{name}", df)

    def get_bronze(self, name: str) -> "pd.DataFrame":
        return self.get_csv(f"bronze/{name}")

    def put_silver(self, name: str, df: "pd.DataFrame") -> None:
        self.put_csv(f"silver/{name}", df)

    def get_silver(self, name: str) -> "pd.DataFrame":
        return self.get_csv(f"silver/{name}")

    def put_gold(self, name: str, df: "pd.DataFrame") -> None:
        self.put_csv(f"gold/{name}", df)

    def get_gold(self, name: str) -> "pd.DataFrame":
        return self.get_csv(f"gold/{name}")

    # --- Model artifact helpers ---

    def put_model(self, name: str, path: str | Path) -> None:
        data = Path(path).read_bytes()
        self.put_bytes(f"models/{name}", data)
        logger.info("PUT model s3://%s/models/%s (%d bytes)", self._bucket, name, len(data))

    def get_model(self, name: str, path: str | Path) -> None:
        data = self.get_bytes(f"models/{name}")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)
        logger.info("GET model s3://%s/models/%s → %s", self._bucket, name, path)


class S3StateStore(StateStore):
    """S3-backed state store for reef twin state.

    Stores reef_state.json in the gold layer of S3.
    """

    def __init__(self, s3_store: S3DataStore | None = None, key: str = "gold/reef_state.json") -> None:
        self._s3 = s3_store or S3DataStore()
        self._key = key

    def save(self, payload: dict[str, Any]) -> None:
        self._s3.put_json(self._key, payload)

    def load(self) -> dict[str, Any]:
        if not self._s3.exists(self._key):
            return {"generated_at": None, "states": []}
        return self._s3.get_json(self._key)

    def load_states(self) -> list[dict[str, Any]]:
        return self.load().get("states", [])
