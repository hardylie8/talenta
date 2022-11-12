#!/usr/bin/env python
# from flask_session import Session
from flask import Flask, render_template, Response
import io
import cv2
import mediapipe as mp
import numpy as np
import jyserver.Flask as jsf
from datetime import timedelta
from tensorflow import keras
from keras.models import load_model
# from flask_bootstrap import Bootstrap

vc = cv2.VideoCapture(0)
hand_found = True
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
app = Flask(__name__)

app.permanent_session_lifetime = timedelta(minutes=5)
savedModel = load_model('model/benar6_10.h5')
holModel = load_model('model/KATA_benar7_10.h5')


@app.route('/kata')
def kata():
    """Kata classification page."""
    return render_template('kata.html', hand_found=hand_found)


@app.route('/')
def home():
    """Video streaming home page."""
    return render_template('home.html', hand_found=hand_found)


@app.route('/huruf')
def index():
    return render_template('index.html', hand_found=hand_found)


def hol():
    while True:
        read_return_code, frame = vc.read()

        with mp_holistic.Holistic(min_detection_confidence=0.5) as hands:

            # Read image file with cv2 and process with face_mesh
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Define boolean corresponding to whether or not a face was detected in the image
            if results.pose_landmarks:
                # Create a copy of the image
                annotated_image = frame.copy()
                annotated_image = np.empty(annotated_image.shape)
                annotated_image.fill(255)

                # Draw landmarks on face
                mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(
                                              color=(0, 0, 255), thickness=4, circle_radius=3),
                                          mp_drawing.DrawingSpec(
                                              color=(0, 255, 0), thickness=2, circle_radius=2),
                                          )
                mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(
                                              color=(0, 0, 255), thickness=4, circle_radius=3),
                                          mp_drawing.DrawingSpec(
                                              color=(0, 255, 0), thickness=2, circle_radius=2),
                                          )
                mp_drawing.draw_landmarks(annotated_image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing_styles
                                          .get_default_face_mesh_tesselation_style()
                                          )
                mp_drawing.draw_landmarks(annotated_image,  results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(
                                              color=(0, 0, 255), thickness=4, circle_radius=3),
                                          mp_drawing.DrawingSpec(
                                              color=(0, 255, 0), thickness=2, circle_radius=2),
                                          )

                img_float32 = np.float32(annotated_image)
                image = cv2.cvtColor(img_float32, cv2.COLOR_BGR2GRAY)
                img_int8 = image.astype(np.uint8)
                equ = cv2.equalizeHist(img_int8)
                # x = image.img_to_array(equ)
                imgPredict = cv2.resize(
                    equ, (224, 224), interpolation=cv2.INTER_LINEAR)
                x = np.expand_dims(imgPredict, axis=0)
                images = np.vstack([x])
                classes = holModel.predict(images, batch_size=10)
                result = np.argmax(classes)
                # print(fn)
                arr = ["azan", "baik", 'bibir', "cinta", "dengar", "dia", "diam", "duduk", "guru", "ibu", "ini", "jam", "maaf", "mahasiswa", "perempuan", "pilih", "punya",
                       "salam", "saya", "sehat", "teman", "tidur", "tugas", "wajib", "zuhur"]
                if(classes[0][result] > 0.6):
                    equ = cv2.putText(
                        img=equ,
                        text=arr[result],
                        org=(100, 100),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=2.0,
                        color=(125, 246, 55),
                        thickness=3
                    )

                encode_return_code, image_buffer = cv2.imencode(
                    '.jpg', equ)
                io_buf = io.BytesIO(image_buffer)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')


def gen():
    """Video streaming generator function."""
    while True:
        read_return_code, frame = vc.read()

        with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:

            # Read image file with cv2 and process with face_mesh
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Define boolean corresponding to whether or not a face was detected in the image
            hand_found = bool(results.multi_hand_landmarks)
            if hand_found:
                # Create a copy of the image
                annotated_image = frame.copy()
                annotated_image = np.empty(annotated_image.shape)
                annotated_image.fill(255)

                # Draw landmarks on face
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(
                                                  color=(0, 0, 255), thickness=4, circle_radius=3),
                                              mp_drawing.DrawingSpec(
                                                  color=(0, 255, 0), thickness=4, circle_radius=2),
                                              )
                img_float32 = np.float32(annotated_image)
                image = cv2.cvtColor(img_float32, cv2.COLOR_BGR2GRAY)
                img_int8 = image.astype(np.uint8)
                equ = cv2.equalizeHist(img_int8)
                # x = image.img_to_array(equ)
                imgPredict = cv2.resize(
                    equ, (224, 224), interpolation=cv2.INTER_LINEAR)
                x = np.expand_dims(imgPredict, axis=0)
                images = np.vstack([x])
                classes = savedModel.predict(images, batch_size=10)
                result = np.argmax(classes)
                # print(fn)
                arr = [
                    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "L",
                    "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y"
                ]
                if(classes[0][result] > 0.6):
                    equ = cv2.putText(
                        img=equ,
                        text=arr[result],
                        org=(200, 200),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=3.0,
                        color=(125, 246, 55),
                        thickness=3
                    )

                encode_return_code, image_buffer = cv2.imencode(
                    '.jpg', equ)
                io_buf = io.BytesIO(image_buffer)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/hol_video_feed')
def hol_video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        hol(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    app.run(debug=True)
