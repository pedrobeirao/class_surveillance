import class_surveillance
from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
	json_path = '/home/pi/object_detection/conf.json'
	classification_label, prob = class_surveillance.run_surveillance(json_path)
	return "Image Label is :"+ classification_label+ ", with Accuracy: "+ str(prob) + "%."

if __name__ == '__main__':
    app.run()
