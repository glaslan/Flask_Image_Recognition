"""
This module contains integration tests for sad path scenarios.
"""

import pytest


@pytest.fixture
def client():
    """Fixture for the Flask test client."""
    from app import app
    with app.test_client() as test_client:
        yield test_client

def test_missing_file(client):
    """Test the prediction route with a missing file."""
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"File cannot be processed." in response.data  # Check if the error message is displayed
