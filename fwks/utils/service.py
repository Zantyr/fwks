import flask

import json
import numpy as np
import os
import scipy.io.wavfile as sio
import tempfile


class Service:
    """
    A basic abstract class for hosting speech processing applications.
    """
    def __init__(self, app=None, root="/"):
        self._app = app
        self._root = root
        if app:
            self.router = app.Namespace()
        else:
            self.router = flask.Flask()
        self.router.route("/predict", methods=["post"])(self._predict_callback)


class ASRService(Service):
    """
    A class created for hosting the speech recognition models. Wraps a speech recognition
    engine with .predict() method and directly returns the data produced by the model.
    Accepts .wav files.
    """

    _WIDGET = """
    <form action="/predict" method="post" enctype="multipart/form-data">
    	<input type="file" name="recording" />
    	<input type="submit" />
    </form>
    """

    _HTML_BASE = """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>ASRDemo</title>
        <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
      </head>
      <body>
        {widget}
      </body>
    </html>
    """.replace('{widget}', _WIDGET)

    def __init__(self, app=None, root="/", engine=None):
        super().__init__(app=app, root=root)
        self.router.route('/')(lambda: flask.render_template(self._HTML_BASE))
        self.engine = engine

    def _predict_callback(self):
        if 'recording' not in flask.request.files:
            json.dumps(['No file passed']), 400
        file = flask.request.files['recording']
        if file.filename == '':
            json.dumps(['No file passed']), 400
        if file:
            filename = tempfile.mktemp()
            try:
                file.save(filename)
                rec = sio.read(filename)[1].astype(np.float32) / 2**15
                content = self.engine.predict(rec, **flask.request.form)
                return content
            finally:
                os.remove(filename)
        else:
            return json.dumps(['No file passed']), 400 

