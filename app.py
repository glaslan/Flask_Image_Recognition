'''
This module is the main access point to the flask digit recognition program.
It sets routes for the user to access.
'''

# Importing required libs
from flask import Flask, render_template, request
from model import preprocess_img, predict_result

# Instantiating flask app
app = Flask(__name__)


# Home route
@app.route("/")
def main():
    """
    Render the home page.

    Returns:
        str: Rendered HTML template for the home page.
    """
    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    """
    Process uploaded image file and predict the digit.

    Accepts a POST request with an image file, preprocesses it,
    and returns the prediction result.

    Returns:
        str: Rendered HTML template with prediction results or error message.
    """
    try:
        if request.method == 'POST':
            img = preprocess_img(request.files['file'].stream)
            pred = predict_result(img)
            return render_template("result.html", predictions=str(pred))

    except Exception as e:
        error = "File cannot be processed."
        return render_template("result.html", err=error)


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
