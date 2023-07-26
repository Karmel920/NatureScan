from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from model import make_prediction

app = Flask(__name__)
api = Api(app)


class Index(Resource):
    def get(self):
        return {'description': 'CNN model API to recognize objects', 
                'routes': ['GET / - general info', 
                           'POST /predict - return predicted class']}


class ImagePrediction(Resource):
    def post(self):
        image = request.files['image']
        return make_prediction(image)
        
    
api.add_resource(Index, '/')    
api.add_resource(ImagePrediction, '/predict')

if __name__ == '__main__':
    app.run(debug=True)
