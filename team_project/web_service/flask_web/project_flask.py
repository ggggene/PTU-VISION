from ultralytics import YOLO, solutions
import cv2
from flask import Flask, Response, render_template, jsonify
app = Flask(__name__)

# YOLO 모델 로드
model = YOLO("best.pt")

# 현재 상태 저장 변수
current_status = {
    "marker1": {"count": 0, "status": "Normal", "color": "black"},
    "marker2": {"count": 0, "status": "Normal", "color": "black"},
}

# 첫 번째 비디오 스트리밍 함수
def generate_frame_1():
    # video 경로 설정 필수!!!
    cap = cv2.VideoCapture("C:/Users/USER/Documents/PTU-VISION/team_project/web_service/flask_web/static/input1.mp4")
    return stream_video(cap, "marker1")

# 두 번째 비디오 스트리밍 함수
def generate_frame_2():
    # video 경로 설정 필수!!!
    cap = cv2.VideoCapture("C:/Users/USER/Documents/PTU-VISION/team_project/web_service/flask_web/static/input2.mp4")  # 다른 비디오 파일
    return stream_video(cap, "marker2")

# 공통 비디오 스트리밍 로직
def stream_video(cap, marker):
    global current_status
    while True:
        success, frame = cap.read()
        if not success:
            print("프레임 확인")
            break
        
        # 프레임 크기 줄이기
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        results = model(frame, conf=0.1)
        annotated_frame = results[0].plot()
        detected_objects_count = len(results[0].boxes)

        # 상태 메시지 정의
        status = f"COUNT: {detected_objects_count}"
        if detected_objects_count <= 1:
            status += " => Normal"
            color = "black"
        elif detected_objects_count <= 3:
            status += " => Warning"
            color = "blue"
        else:
            status += " => Danger"
            color = "red"

        # 상태 저장
        current_status[marker] = {"count": detected_objects_count, "status": status, "color": color}
        
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

# Flask 라우트 설정
@app.route('/')
def index():
    return render_template("page1.html")

@app.route('/video1')
def video_feed_1():
    return Response(generate_frame_1(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video2')
def video_feed_2():
    return Response(generate_frame_2(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 현재 상태를 반환하는 API
@app.route('/get_status')
def get_status():
    return jsonify(current_status)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)