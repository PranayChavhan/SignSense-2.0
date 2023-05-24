import cv2
import mediapipe as mp
from tensorflow import keras
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, Response

app = Flask(__name__)

def get_connections_list():
    return {
        "WRIST_TO_THUMB_MCP": (0, 2),
        "WRIST_TO_THUMB_IP": (0, 3),
        "WRIST_TO_THUMB_TIP": (0, 4),
        "WRIST_TO_INDEX_FINGER_PIP": (0, 6),
        "WRIST_TO_INDEX_FINGER_DIP": (0, 7),
        "WRIST_TO_INDEX_FINGER_TIP": (0, 8),
        "WRIST_TO_MIDDLE_FINGER_PIP": (0, 10),
        "WRIST_TO_MIDDLE_FINGER_DIP": (0, 11),
        "WRIST_TO_MIDDLE_FINGER_TIP": (0, 12),
        "WRIST_TO_RING_FINGER_PIP": (0, 14),
        "WRIST_TO_RING_FINGER_DIP": (0, 15),
        "WRIST_TO_RING_FINGER_TIP": (0, 16),
        "WRIST_TO_PINKY_PIP": (0, 18),
        "WRIST_TO_PINKY_DIP": (0, 19),
        "WRIST_TO_PINKY_TIP": (0, 20),
        "THUMB_MCP_TO_THUMB_TIP": (2, 4),
        "INDEX_FINGER_MCP_TO_INDEX_FINGER_TIP": (5, 8),
        "MIDDLE_FINGER_MCP_TO_MIDDLE_FINGER_TIP": (9, 12),
        "RING_FINGER_MCP_TO_RING_FINGER_TIP": (13, 16),
        "PINKY_MCP_TO_PINKY_TIP": (17, 20),
        "THUMB_TIP_TO_INDEX_FINGER_MCP": (4, 5),
        "THUMB_TIP_TO_INDEX_FINGER_PIP": (4, 6),
        "THUMB_TIP_TO_INDEX_FINGER_DIP": (4, 7),
        "THUMB_TIP_TO_INDEX_FINGER_TIP": (4, 8),
        "THUMB_TIP_TO_MIDDLE_FINGER_MCP": (4, 9),
        "THUMB_TIP_TO_MIDDLE_FINGER_PIP": (4, 10),
        "THUMB_TIP_TO_MIDDLE_FINGER_DIP": (4, 11),
        "THUMB_TIP_TO_MIDDLE_FINGER_TIP": (4, 12),
        "THUMB_TIP_TO_RING_FINGER_MCP": (4, 13),
        "THUMB_TIP_TO_RING_FINGER_PIP": (4, 14),
        "THUMB_TIP_TO_RING_FINGER_DIP": (4, 15),
        "THUMB_TIP_TO_RING_FINGER_TIP": (4, 16),
        "THUMB_TIP_TO_PINKY_MCP": (4, 17),
        "THUMB_TIP_TO_PINKY_PIP": (4, 18),
        "THUMB_TIP_TO_PINKY_DIP": (4, 19),
        "THUMB_TIP_TO_PINKY_TIP": (4, 20)
    }

def get_distance(first, second):
    return np.sqrt(
        (first.x - second.x) ** 2 
        + (first.y - second.y) ** 2 
        + (first.z - second.z) ** 2
    )

def get_sign_list():
    df = pd.read_csv('connections.csv', index_col=0)
    return df['SIGN'].unique()

def process_frames():
    global prediction 
    sign_list = get_sign_list()
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    connections_dict = get_connections_list()

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)

            results = hands.process(image)
            if not results.multi_hand_landmarks:
                prediction=""
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n\r\n')
            else:
                mp_drawing.draw_landmarks(
                    image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                )

                coordinates = results.multi_hand_landmarks[0].landmark
                data = []
                for _, values in connections_dict.items():
                    data.append(get_distance(coordinates[values[0]], coordinates[values[1]]))

                data = np.array([data])
                data[0] /= data[0].max()

                model = keras.models.load_model('ann_model.h5')

                pred = np.array(model(data))
                pred = sign_list[pred.argmax()]
                
                prediction = pred
                
              

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image = cv2.putText(
                    image, pred, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (255, 0, 0), 2
                )

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', image)[1].tobytes() + b'\r\n\r\n')

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


@app.route('/get_prediction')
def get_prediction():
    global prediction  # Access the global prediction variable
    return jsonify({'prediction': prediction})  # Return the prediction as a JSON response

@app.route('/prediction')
def index():
    return render_template('index.html')

@app.route('/')
def learning():
    return render_template('learning.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
   
