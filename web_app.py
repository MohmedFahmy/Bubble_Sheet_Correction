import optik as ok
from contextlib import redirect_stdout
import requests
from flask import Response
from flask import Flask
from flask import render_template
from flask import redirect
from flask import url_for
from flask import request
import threading
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

bullbe_app=Flask(__name__)

UPLOAD_FOLDER = 'static\\uploads\\'
 
bullbe_app.secret_key = "secret key"
bullbe_app.config['UPLOAD_PATH'] = UPLOAD_FOLDER
bullbe_app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bullbe_app.route("/")
def index():
    path = 'static/uploads/'
    uploads = sorted(os.listdir(path), key=lambda x: os.path.getctime(path+x)) # Sorting as per image upload date and time

    return render_template("index.html")  

@bullbe_app.route("/",methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        print(f.filename)
        #f.save(secure_filename(f.filename))
        filename = secure_filename(f.filename)
        base_path = os.path.abspath(os.path.dirname(__file__))
        upload_path = os.path.join(base_path, bullbe_app.config['UPLOAD_PATH'])
        f.save(os.path.join(upload_path, filename))

        bubbleSheetScanner=ok.BubbleSheetScanner()

        image = cv2.imread('static/uploads/'+f.filename)
        h = int(round(600 * image.shape[0] / image.shape[1]))
        frame = cv2.resize(image, (600, h), interpolation=cv2.INTER_LANCZOS4)

        cannyFrame = bubbleSheetScanner.getCannyFrame(frame=frame)

        warpedFrame = bubbleSheetScanner.getWarpedFrame(cannyFrame, frame)

        adaptiveFrame = bubbleSheetScanner.getAdaptiveThresh(warpedFrame)

        ovalContours = bubbleSheetScanner.getOvalContours(adaptiveFrame)
        if (len(ovalContours) == bubbleSheetScanner.ovalCount):
            ovalContours = sorted(ovalContours, key=bubbleSheetScanner.y_cord_contour, reverse=False)
            score = 0
            for (q, i) in enumerate(np.arange(0, len(ovalContours), bubbleSheetScanner.bubbleCount)):
                bubbles = sorted(ovalContours[i:i + bubbleSheetScanner.bubbleCount], key=bubbleSheetScanner.x_cord_contour,reverse=False)
                for (j, c) in enumerate(bubbles):
                    area = cv2.contourArea(c)
                    mask = np.zeros(adaptiveFrame.shape, dtype="uint8")
                    cv2.drawContours(mask, [c], -1, 255, -1)
                    mask = cv2.bitwise_and(adaptiveFrame, adaptiveFrame, mask=mask)
                    total = cv2.countNonZero(mask)
                    answer = bubbleSheetScanner.ANSWER_KEY[q]
                    x, y, w, h = cv2.boundingRect(c)
                    isBubbleSigned = ((float)(total) / (float)(area)) > 1
                    if (isBubbleSigned):
                        if (answer == j):
                            # And calculate score
                            score += 100 / bubbleSheetScanner.questionCount
                            # Sign correct answered bubble with green circle
                            cv2.drawContours(warpedFrame, bubbles, j, (0, 255, 0), 2)
                        else:
                            # Sign wrong answered bubble with red circle
                            cv2.drawContours(warpedFrame, bubbles, j, (0, 0, 255), 1)
                    # Sign correct bubble
                    cv2.drawContours(warpedFrame, bubbles, answer, (255, 0, 0), 1)
            # Add score
            warpedFrame = cv2.putText(warpedFrame, 'Score:' + str(score),(100, 12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 255),1)
            # cv2.imshow('result', warpedFrame)
            print(f"score in exam ==  {score}")
        else:print('Invalid frame')
        return render_template("index.html",filename=f.filename,score=score)           # Redirect to route '/' for displaying images on fromt end

if __name__ == "__main__":
    bullbe_app.debug = True
    bullbe_app.run(port=9000)