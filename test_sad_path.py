# test_sad_path.py

import pytest
from io import BytesIO
from app import app

@pytest.fixture
def client():
    """
    Fixture for the Flask test client.
    - Purpose: Set up a test client for making requests to the Flask app during testing.
    - Usage: Provides a `client` object to use for HTTP request simulations.
    """
    with app.test_client() as client:
        yield client


def test_prediction_no_file_uploaded(client):
    """
    Test Case: No file uploaded to /prediction
    - Auther Navjot
    - Purpose: Ensure the app gracefully handles when no file is provided.
    - Expected Result: Should return 200 with an error message in the rendered page.
    """
    response = client.post(
        "/prediction", 
        data={}, 
        content_type="multipart/form-data"
    )
    
    assert response.status_code == 200
    assert b"Prediction" in response.data
    assert b"File cannot be processed" in response.data


def test_prediction_invalid_file_type(client):
    """
    Test Case: Invalid file type uploaded (text instead of image)
    - Auther Navjot
    - Purpose: Ensure the app catches processing errors gracefully.
    - Expected Result: Returns 200 with an error message rendered in the template.
    """
    fake_file = BytesIO(b"This is not an image")
    response = client.post(
        "/prediction",
        data={"file": (fake_file, "fake.txt")},
        content_type="multipart/form-data"
    )
    assert response.status_code == 200
    assert b"Prediction" in response.data
    assert b"File cannot be processed" in response.data