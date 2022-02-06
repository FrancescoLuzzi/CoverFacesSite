import os
import cv2 as cv
import numpy as np
from flask import Flask, render_template, request, make_response
from werkzeug.datastructures import FileStorage

from flaskr.flask_cv_factories import factories

model_factory = factories.modelFactory()
painter_factory = factories.painterFactory()


def create_app(test_config=None):
    # create and configure the app

    app = Flask(__name__, instance_relative_config=True)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route("/")
    def hello():
        return render_template("basePage.html", title="FaceUnderCover")

    def process_image(file: FileStorage, painter, model):
        img = cv.imdecode(np.frombuffer(file.read(), np.uint8), cv.IMREAD_UNCHANGED)
        height, width, _ = np.shape(img)
        detections = model.find_detections(
            frame=img, frame_width=width, frame_height=height
        )
        painter.paint_frame(img, detections=detections)
        output = cv.imencode(file.mimetype.replace("image/", "."), img)[1].tobytes()
        response = make_response(output)
        response.headers.set("Content-Type", file.mimetype)
        response.headers.set(
            "Content-Disposition", "attachment", filename=f"OUT_{file.filename}"
        )
        return response

    @app.route("/facedetection", methods=["POST"])
    def i_cant_see_you():
        global model_factory
        global painter_factory
        model = model_factory.aquire_model()
        painter = painter_factory.aquire_painter()

        file = request.files["file"]
        response = "OKOK"
        if "image" in file.mimetype:
            response = process_image(file, painter, model)
        elif "video" in file.mimetype:
            print("Video")
        else:
            print("ERROR")

        model_factory.release_model(model)
        painter_factory.release_painter(painter)

        return response

    return app
