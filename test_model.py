import os, io
import pytest
import numpy as np
from keras.models import load_model
from model import preprocess_img, predict_result  # Adjust based on your structure
from app import app
# Load the model before tests run
@pytest.fixture(scope="module")
def model():
    """Load the model once for all tests."""
    model = load_model("digit_model.h5")  # Adjust path as needed
    return model

# Basic Tests

def test_preprocess_img():
    """Test the preprocess_img function."""
    img_path = "test_images/2/Sign 2 (97).jpeg"
    processed_img = preprocess_img(img_path)

    # Check that the output shape is as expected
    assert processed_img.shape == (1, 224, 224, 3), "Processed image shape should be (1, 224, 224, 3)"

    # Check that values are normalized (between 0 and 1)
    assert np.min(processed_img) >= 0 and np.max(processed_img) <= 1, "Image pixel values should be normalized between 0 and 1"


def test_predict_result(model):
    """Test the predict_result function."""
    img_path = "test_images/4/Sign 4 (92).jpeg"  # Ensure the path is correct
    processed_img = preprocess_img(img_path)

    # Make a prediction
    prediction = predict_result(processed_img)

    # Print the prediction for debugging
    print(f"Prediction: {prediction} (Type: {type(prediction)})")

    # Check that the prediction is an integer (convert if necessary)
    assert isinstance(prediction, (int, np.integer)), "Prediction should be an integer class index"


# Advanced Tests

def test_invalid_image_path():
    """Test preprocess_img with an invalid image path."""
    with pytest.raises(FileNotFoundError):
        preprocess_img("invalid/path/to/image.jpeg")

def test_image_shape_on_prediction(model):
    """Test the prediction output shape."""
    img_path = "test_images/5/Sign 5 (86).jpeg"  # Ensure the path is correct
    processed_img = preprocess_img(img_path)

    # Ensure that the prediction output is an integer
    prediction = predict_result(processed_img)
    assert isinstance(prediction, (int, np.integer)), "The prediction should be an integer"

def test_predict_result_accurate(model):
    """Test the predict_result function and check if it is accurate."""
    img_path = "test_images/4/Sign 4 (92).jpeg"  # Ensure the path is correct
    processed_img = preprocess_img(img_path)

    # Make a prediction
    prediction = predict_result(processed_img)

    # Print the prediction for debugging
    print(f"Prediction: {prediction} (Type: {type(prediction)})")

    # Check that the prediction is an integer (convert if necessary)
    assert prediction == 4; "Prediction should be an integer class index"

def test_model_predictions_consistency(model):
    """Test that predictions for the same input are consistent."""
    img_path = "test_images/7/Sign 7 (54).jpeg"
    processed_img = preprocess_img(img_path)

    # Make multiple predictions
    predictions = [predict_result(processed_img) for _ in range(5)]

    # Check that all predictions are the same
    assert all(p == predictions[0] for p in predictions), "Predictions for the same input should be consistent"


# --- Test Flask main route (index page) ---
@pytest.fixture
def client():
    """
    Fixture for the Flask test client.
    Lighweight Test version of the Flask application
    Simulates HTTP rquests 
    """

    #Creates a temp test client 
    with app.test_client() as client:
        #Returns the test client to the test function 
        yield client


def test_main_route_renders_index_template(client):
    """
    Test Case: Validate that the main route renders the index.html template.
    
    """
    response = client.get("/")

    # Verify that the route executed successfully
    assert response.status_code == 200, "Expected status code 200 for the main route."

    # Verify that the rendered page contains the expected title text.
    # This ensures the correct template (index.html) was rendered successfully.
    assert b"Hand Sign Digit Language Detection" in response.data, \
    "Expected page title not found in index.html response."

def test_prediction_route_renders_result_template(client):
    """
    Test Case: Validate that the /prediction route renders the result.html template.

    This test simulates uploading a valid image file to the prediction endpoint,
    verifies the response status code, and checks that the response contains
    content from result.html (e.g., the prediction or error message).
    """

    # Set file path for image to load from
    img_path = "test_images/2/Sign 2 (97).jpeg"
    
    # Open the image file in binary mode to simulate web based file upload
    with open(img_path, "rb") as img:
        response = client.post(
        "/prediction",
        content_type="multipart/form-data",
        data={"file": (img, "Sign 2 (97).jpeg")}
    )

    # Create a dummy file to simulate image upload
    dummy_image = (io.BytesIO(b"fake image data"), "test_image.png")

    # Verify that the route executed successfully
    assert response.status_code == 200, "Expected status code 200 for the prediction route."

    # Verify that the result.html template was rendered by checking for expected content
    assert b"Prediction" in response.data and b"result" in response.data, \
        "Expected content from result.html not found in response."