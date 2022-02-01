import os
import cv2 as cv
import numpy as np
from flask import Flask, render_template, request, make_response

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

    @app.route("/facedetection", methods=["POST"])
    def i_cant_see_you():
        global model_factory
        global painter_factory
        model = model_factory.aquire_model()
        painter = painter_factory.aquire_painter()
        model.init_net()

        file = request.files["file"]

        if "image" in file.mimetype:
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
        elif "video" in file.mimetype:
            print("Video")
        else:
            print("ERROR")

        return "OKOK"

    return app
