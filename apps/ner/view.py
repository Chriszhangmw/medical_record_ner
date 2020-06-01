

from flask_restful import Resource,reqparse

import tensorflow as tf

from keras.backend import set_session
import cherrypy

from apps.ner.combine import combine,sess
from flask import  current_app

graph = tf.get_default_graph()


class ExtAttrib(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

        self.parser.add_argument("texts",type=str,trim=True,default=[],action='append')

    def post(self):
        _args = self.parser.parse_args()
        texts = _args["texts"]
        if not texts:
            return {"code":400,"msg":"提交内容位空"}
        global graph
        with graph.as_default():
            set_session(sess)
            result = combine.predict(texts)
        return {"data":result},200



