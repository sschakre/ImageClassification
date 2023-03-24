# Load operating system library
import os

# website libraries
from flask import render_template
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

# Load math library
import numpy as np

# Load machine learning libraries
from keras.utils import load_img, img_to_array
from keras.models import load_model
import tensorflow as tf

from keras.backend import set_session

# sess = tf.Session()

# two categories
X = 'Clean Room'
Y = 'Messy Room'

sampleX = 'static/images/clean.png'
sampleY = 'static/images/messy.png'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ML_MODEL_FILENAME = 'saved_model.h5'


# Create the website object
app = Flask(__name__)


def load_model_from_file():
    # Set up the machine learning session
    mySession = tf.compat.v1.Session()
    set_session(mySession)
    myModel = load_model(ML_MODEL_FILENAME)
    myGraph = tf.compat.v1.get_default_graph()
    return mySession, myModel, myGraph


# Try to allow only images
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Define the view for the top level page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # Initial webpage load
    if request.method == 'GET':
        return render_template('index.html', myX=X, myY=Y, mySampleX=sampleX, mySampleY=sampleY)
    else:  # if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser may also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # If it doesn't look like an image file
        if not allowed_file(file.filename):
            flash('I only accept files of type' + str(ALLOWED_EXTENSIONS))
            return redirect(request.url)
        # When the user uploads a file with good parameters
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    test_image = load_img(UPLOAD_FOLDER + "/" + filename, target_size=(224, 224))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    mySession = app.config['SESSION']
    myModel = app.config['MODEL']
    myGraph = app.config['GRAPH']
    with myGraph.as_default():
        set_session(mySession)

        (mySession, myModel, myGraph) = load_model_from_file()
        mySession = tf.compat.v1.Session()
        set_session(mySession)
        myModel = load_model(ML_MODEL_FILENAME)

        result = myModel.predict(test_image)[0].round(1)
        image_src = "/" + UPLOAD_FOLDER + "/" + filename
        if result < 0.5:
            answer = "<div class='col text-center'><img width='150' height='150' src='" + image_src + "' class='img" \
                                                                                                      "-thumbnail' " \
                                                                                                      "/><h4>Guess is " \
                                                                                                      ": " + X + " " \
                     + "</h4></div><div class='col'></div><div class='w-100'></div> "
        else:
            answer = "<div class='col'></div><div class='col text-center'><img width='150' height='150' src='" + image_src + "' class='img-thumbnail' /><h4>Guess is : " + Y + " " + "</h4></div><div class='w-100'></div>"
        results.append(answer)
        return render_template('index.html', myX=X, myY=Y, mySampleX=sampleX, mySampleY=sampleY, len=len(results),
                               results=results)


def main():
    (mySession, myModel, myGraph) = load_model_from_file()

    app.config['SECRET_KEY'] = 'super secret key'

    app.config['SESSION'] = mySession
    app.config['MODEL'] = myModel
    app.config['GRAPH'] = myGraph

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit
    app.run()


# Create a running list of results
results = []

main()
