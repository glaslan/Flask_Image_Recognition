'''
This module provides pytest fixtures for testing the Flask application.
'''

import pytest
from app import app  # This imports the Flask app for testing

@pytest.fixture
def client():
    """
    Create a test client for the Flask application.

    Yields:
        FlaskClient: A test client for making requests to the application.
    """
    with app.test_client() as test_client:
        yield test_client
