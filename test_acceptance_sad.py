'''
This is the acceptance test suite that tests for sad cases.
'''

from io import BytesIO
from unittest.mock import patch
import pytest


@pytest.fixture
def client():
    """
    Fixture for the Flask test client.
    - Purpose: Set up a test client for making requests to the Flask app during testing.
    - Usage: Provides a `client` object to use for HTTP request simulations.
    """
    from app import app
    with app.test_client() as test_client:
        yield test_client


def test_acceptance_missing_file(client):
    """
    Test Case: No File Uploaded
    - Purpose: Validate the application's behavior when no file is provided in the upload request.
    - Scenario:
        - Simulate a POST request to the `/prediction` route with no file data.
        - Assert the response status code is 200 (to indicate a valid request was processed).
        - Verify that the response includes an appropriate error message.
    """
    # Simulate a POST request with no file data
    response = client.post("/prediction", data={}, content_type="multipart/form-data")

    # Assertions:
    # 1. Ensure the response status code is 200, indicating the request was processed.
    assert response.status_code == 200

    # 2. Check for a meaningful error message in the response data.
    #    Modify the message check if your application uses a different error response text.
    assert b"server could not understand." in response.data  # Expected error message

def test_non_existent_file(client):
    """
    GIVEN the Flask_Image Recognition web application is running,
    WHEN the user uploads a non_existen file, or the system
    """
def test_upload_wrong_filetype_textfile(client):
    """
    GIVEN the Flask_Image_Recognition web application is running,
    WHEN the user uploads a text file
    THEN the app should not crash (status 200)
    """
    from io import BytesIO

    text_file = BytesIO(b"fake_large_image_data" * 1000)
    text_file.name = "invalid_file.txt"

    response = client.post(
        "/prediction",
        data={"file": (text_file, text_file.name)},
        content_type="multipart/form-data"
    )

    # Ensure the app doesn't crash
    assert response.status_code == 200

def test_upload_image_too_large(client):
    """
    GIVEN the Flask app is running,
    WHEN an excessively large image is uploaded,
    THEN the app should roload the main page, and respond with "cannot identify image file"
    """
    # Simulate a very large image (50Mb)
    large_image_data = BytesIO(b"large_image_data" * 5000000)
    large_image_data.name = "huge_image.jpg"

    # Mock the prediction function to raise a ValueError for large images
    with patch("model.predict_result", side_effect=ValueError("Image too large")):
        response = client.post(
            "/prediction",
            data={"file": (large_image_data, large_image_data.name)},
            content_type="multipart/form-data"
        )

    # App should return 200 (page rendered) but NOT show a prediction
    assert response.status_code == 200

    # The response should contain an error message about the large image
    assert b"cannot identify image file" in response.data
