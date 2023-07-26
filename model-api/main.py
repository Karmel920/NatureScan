from flask import Flask
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

class Prediction(Resource):
    def get(self):
        return {'pred1': 'rose', 'pred2': 'sunflower'}
    
api.add_resource(Prediction, '/')

if __name__ == '__main__':
    app.run(debug=True)
