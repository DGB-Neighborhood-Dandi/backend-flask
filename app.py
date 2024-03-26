# app.py
from flask import Flask, jsonify, render_template, Response
import cv2
import pandas as pd
import torch


# socket
from flask_socketio import SocketIO, send

app = Flask(__name__)
app.secret_key = "mysecret"
# socket_io = SocketIO(app)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 후대폰 탐지여부
detect_cell_phone = False

# OpenCV를 사용하여 웹캠에 연결
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 가리킴


def findCellPhone(df):

    # 'name' 열에서 'cell phone'인 행이 있는지 확인
    if (df['name'] == 'cell phone').any():
        print("cell phone을 탐지했습니다.")
        detect_cell_phone = True
    else:
        print("cell phone을 탐지하지 못했습니다.")
        detect_cell_phone = False

        # 휴대폰 탐지 여부에 따라 메시지 보내기
    if detect_cell_phone:
        socket_io.emit('phone_detection', {'phone_detected': True})
    else:
        socket_io.emit('phone_detection', {'phone_detected': False})


@app.route('/')
def video_show():
    return render_template('video_show.html')


def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            results = model(frame)
            # render()의 반환 값은 리스트이므로 첫 번째 요소 사용
            annotated_frame = results.render()[0]
            annotated_frame = cv2.cvtColor(
                annotated_frame, cv2.COLOR_RGB2BGR)  # OpenCV 형식으로 변환

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            results.pandas().xyxy[0]
            # 데이터프레임 생성
            df = pd.DataFrame(results.pandas().xyxy[0])

            findCellPhone(df)

            print(df)

        # 프레임 전송
            # socket_io.emit('frame', {'frame': frame}, namespace='/video')

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    socket_io = SocketIO(app, cors_allowed_origins="http://localhost:3000")
    socket_io.run(app, debug=True, port=9999)
