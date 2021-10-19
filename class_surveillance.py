from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import warnings
import datetime

import imutils
import json
import time
import cv2
from tflite_runtime.interpreter import Interpreter 
from PIL import Image
import numpy as np
import time

def load_labels(path): # Read the labels from the text file as a Python list.
	with open(path, 'r') as f:
		return [line.strip() for i, line in enumerate(f.readlines())]

def set_input_tensor(interpreter, image):
	tensor_index = interpreter.get_input_details()[0]['index']
	input_tensor = interpreter.tensor(tensor_index)()[0]
	input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
	set_input_tensor(interpreter, image)

	interpreter.invoke()
	output_details = interpreter.get_output_details()[0]
	output = np.squeeze(interpreter.get_tensor(output_details['index']))

	scale, zero_point = output_details['quantization']
	output = scale * (output - zero_point)

	ordered = np.argpartition(-output, 1)
	return [(i, output[i]) for i in ordered[:top_k]][0]

def camera_init():
	camera = PiCamera()
	camera.resolution = tuple(conf["resolution"])
	camera.framerate = conf["fps"]
	return camera

def load_model():
	data_folder = "/home/pi/object_detection/class_surveillance/"
	model_path = data_folder + "model.tflite"
	label_path = data_folder + "labels.txt"

	with open(json_path) as f:
	  conf = json.load(f)
	client = None

def model_background():
	frame = f.array
	timestamp = datetime.datetime.now()
	text = "Empty"
	frame = imutils.resize(frame, width=512)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	if avg is None:
		print("Estimating background...")
		avg = gray.copy().astype("float")
		rawCapture.truncate(0)
		continue

	cv2.accumulateWeighted(gray, avg, 0.5)
	frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

	thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
		cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	return cnts

def save_and_classify_images(text):
	if text == "Occupied":
	if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
		motionCounter += 1

		if motionCounter >= conf["min_motion_frames"]:
				cv2.imwrite(tpath, frame)
				interpreter = Interpreter(model_path)
				print("Model Loaded Successfully.")
				interpreter.allocate_tensors()
				_, height, width, _ = interpreter.get_input_details()[0]['shape']
				print("Image Shape (", width, ",", height, ")")

				# Load an image to be classified.
				image = Image.open(t.path).convert('RGB').resize((width, height))

				# Classify the image.
				time1 = time.time()
				label_id, prob = classify_image(interpreter, image)
				time2 = time.time()
				classification_time = np.round(time2-time1, 3)
				print("Classificaiton Time =", classification_time, "seconds.")

				# Read class labels.
				labels = load_labels(label_path)

				# Return the classification label of the image.
				classification_label = labels[label_id]
				
									# Load an image to be classified.

			# update the last uploaded timestamp and reset the motion
			# counter
			lastUploaded = timestamp
			motionCounter = 0
	# otherwise, the room is not occupied
	else:
		motionCounter = 0
	return classification_label, np.round(prob*100, 2)

def run_surveillance(json_path):
	load_model()
	# initialize the camera and grab a reference to the raw camera capture
	camera = camera_init()
	rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))
	# check to see if the Dropbox should be used
	#if conf["use_dropbox"]:
		# connect to dropbox and start the session authorization process
	#	client = dropbox.Dropbox(conf["dropbox_access_token"])
	#	print("[SUCCESS] dropbox account linked")

	# allow the camera to warmup, then initialize the average frame, last
	# uploaded timestamp, and frame motion counter
	print("[INFO] warming up...")
	time.sleep(conf["camera_warmup_time"])
	avg = None
	lastUploaded = datetime.datetime.now()
	motionCounter = 0

	# capture frames from the camera
	for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
		# grab the raw NumPy array representing the image and initialize
		# the timestamp and occupied/unoccupied text
		# create model backgound contours
		cnts = model_background()
		# loop over the contours
		for c in cnts:
			# if the contour is too small, ignore it
			if cv2.contourArea(c) < conf["min_area"]:
				continue
			# compute the bounding box for the contour, draw it on the frame,
			# and update the text
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			text = "Occupied"
		# draw the text and timestamp on the frame
		ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
		cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
			0.35, (0, 0, 255), 1)

		# check to see if the room is occupied
		classification_label, np.round(prob*100, 2) = save_and_classify_images(text)

		# check to see if the frames should be displayed to screen
		if conf["show_video"]:
			# display the security feed
			cv2.imshow("Security Feed", frame)
			key = cv2.waitKey(1) & 0xFF
			# if the `q` key is pressed, break from the lop
			if key == ord("q"):
				break
		# clear the stream in preparation for the next frame
		rawCapture.truncate(0)
		return classification_label, np.round(prob*100, 2)
