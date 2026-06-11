from __future__ import annotations

import os
from urllib.parse import urljoin

import pytest
import requests


TYPE1_SUBMISSION_REQUEST = {
    "query_id": "api-smoke-type1",
    "type": "type1",
    "query": "Does it follow that Alex can access the lab?",
    "premises": [
        "Every student who completed safety training can access the lab.",
        "Alex completed safety training.",
    ],
    "options": ["Yes", "No", "Uncertain"],
}

TYPE2_SUBMISSION_REQUEST = {
    "query_id": "api-smoke-type2",
    "type": "type2",
    "query": "Calculate the current when V = 12 V and R = 4 ohms.",
    "premises": [],
    "options": [],
}


def predict_url() -> str:
    raw_url = (
        os.getenv("SUBMISSION_API_URL")
        or os.getenv("PIPELINE_API_URL")
        or os.getenv("EXACT_API_URL")
    )
    if not raw_url:
        pytest.skip(
            "Set SUBMISSION_API_URL, PIPELINE_API_URL, or EXACT_API_URL to run "
            "the external API tests."
        )

    base_url = raw_url.rstrip("/")
    if base_url.endswith("/predict"):
        return base_url
    return urljoin(base_url + "/", "predict")


def request_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = os.getenv("SUBMISSION_API_TOKEN") or os.getenv("PIPELINE_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def assert_submission_response(result: dict, query_id: str) -> None:
    assert result["query_id"] == query_id
    assert isinstance(result["answer"], str)
    assert result["answer"]
    assert isinstance(result["unit"], str)
    assert isinstance(result["explanation"], str)
    assert result["explanation"]
    assert isinstance(result["premises_used"], list)
    assert "reasoning" in result


def test_external_api_single_submission_request() -> None:
    response = requests.post(
        predict_url(),
        json=TYPE2_SUBMISSION_REQUEST,
        headers=request_headers(),
        timeout=120,
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert_submission_response(payload[0], TYPE2_SUBMISSION_REQUEST["query_id"])


def test_external_api_batch_submission_request() -> None:
    request_body = [TYPE1_SUBMISSION_REQUEST, TYPE2_SUBMISSION_REQUEST]

    response = requests.post(
        predict_url(),
        json=request_body,
        headers=request_headers(),
        timeout=180,
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == len(request_body)
    assert [item["query_id"] for item in payload] == [
        item["query_id"] for item in request_body
    ]

    for result, request_item in zip(payload, request_body):
        assert_submission_response(result, request_item["query_id"])
