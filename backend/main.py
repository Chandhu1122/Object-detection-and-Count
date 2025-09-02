import os
import sys
import time
import math
import smtplib
import ssl
import cv2
import torch
import numpy as np
import pathlib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List, Dict
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from queue import Queue

# Load environment variables
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StartRequest(BaseModel):
    video_url: str
    supervisor_name: str
    vehicle_number: str
    model_name: str

class StopRequest(BaseModel):
    supervisor_name: Optional[str] = None
    vehicle_number: Optional[str] = None

class DetectionState:
    is_running: bool = False
    vehicle_number: str = ""
    supervisor_name: str = ""
    start_time: float = 0
    frame_count: int = 0
    box_count: int = 0
    video_path: str = ""
    crossed_counts: Dict[str, int] = {}
    total_value: int = 0
    
state = DetectionState()
reports: List[Dict] = []
frame_queue = Queue(maxsize=1)
video_thread: Optional[threading.Thread] = None

# --- User Configurable Part ---
YOLOV5_PATH = "yolov5"
CONF_THRESH = 0.6
IOU_THRESH = 0.5
MODEL_WEIGHTS_PATHS = {
    "4,5,6 box": "bestb.pt",
    "single box": "best3.pt",
    "multiple box": "2best.pt",
}
LINE_POSITIONS = {
    "4,5,6 box": 0.30,
    "single box": 0.60,
    "multiple box": 0.10,
}
models_cache = {}

class PosixPathPatch(pathlib.PosixPath):
    def _new_(cls, *args, **kwargs):
        return pathlib.WindowsPath(*args, **kwargs)
pathlib.PosixPath = PosixPathPatch

FILE = os.path.abspath(_file_)
ROOT = os.path.dirname(FILE)
YOLOV5_FULLPATH = os.path.join(ROOT, YOLOV5_PATH)
if YOLOV5_FULLPATH not in sys.path:
    sys.path.append(YOLOV5_FULLPATH)

def get_model(model_name: str):
    if model_name in models_cache:
        return models_cache[model_name]
    weights_path = MODEL_WEIGHTS_PATHS.get(model_name)
    if not weights_path:
        raise HTTPException(status_code=400, detail=f"Invalid model name: {model_name}")

    try:
        model = torch.hub.load(YOLOV5_FULLPATH, 'custom', path=weights_path, source='local')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.conf = CONF_THRESH
        model.iou = IOU_THRESH
        models_cache[model_name] = model
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {model_name}. Error: {e}")

def send_email(to_email: str, subject: str, body: str):
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    smtp_server = os.getenv("SENDER_SMTP_SERVER")
    smtp_port = int(os.getenv("SENDER_SMTP_PORT", 587))
    receiver_email = os.getenv("RECEIVER_EMAIL")

    if not all([sender_email, sender_password, receiver_email]):
        print("Email configuration is missing. Cannot send email.")
        return
        
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = receiver_email
    part = MIMEText(body, "plain")
    message.attach(part)
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, message.as_string())
        print(f"Email sent successfully to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def run_video_processing(model_name: str):
    global state, frame_queue
    try:
        model = get_model(model_name)
    except HTTPException as e:
        state.is_running = False
        return

    cap = cv2.VideoCapture(state.video_path)
    if not cap.isOpened():
        state.is_running = False
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    counted_ids: Set[int] = set()
    state.total_value = 0
    state.crossed_counts = {}
    
    line_position_ratio = LINE_POSITIONS.get(model_name, 0.50)
    line_x = int(width * line_position_ratio)
    
    tracker = DeepSort()

    while state.is_running:
        ret, frame = cap.read()
        if not ret:
            state.is_running = False
            break
        
        state.frame_count += 1
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        dets = results.xyxy[0].cpu().numpy()
        
        deepsort_dets = []
        for *xyxy, conf, cls in dets:
            x1, y1, x2, y2 = map(float, xyxy)
            w = x2 - x1
            h = y2 - y1
            class_name = model.names[int(cls)]
            deepsort_dets.append(([x1, y1, w, h], float(conf), class_name))
        
        tracks = tracker.update_tracks(deepsort_dets, frame=frame)
        state.box_count = len(tracks)

        cv2.line(frame, (line_x, 0), (line_x, height), (150, 150, 150), 2)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cx = int((x1 + x2) / 2)
            label = track.get_det_class()
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id} {label}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if cx > line_x and track_id not in counted_ids:
                if label == 'box':
                    state.crossed_counts[label] = state.crossed_counts.get(label, 0) + 1
                    state.total_value += 1
                    counted_ids.add(track_id)
                elif label == '4box':
                    state.crossed_counts[label] = state.crossed_counts.get(label, 0) + 1
                    state.total_value += 4
                    counted_ids.add(track_id)
                elif label == '5box':
                    state.crossed_counts[label] = state.crossed_counts.get(label, 0) + 1
                    state.total_value += 5
                    counted_ids.add(track_id)
                elif label == '6box':
                    state.crossed_counts[label] = state.crossed_counts.get(label, 0) + 1
                    state.total_value += 6
                    counted_ids.add(track_id)
        
        y_offset = 50
        cv2.putText(frame, f"Total Value: {state.total_value}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        y_offset += 40
        for label, count in state.crossed_counts.items():
            cv2.putText(frame, f"{label}: {count}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 40

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            try:
                frame_queue.put_nowait(buffer.tobytes())
            except:
                pass
    
    cap.release()
    report_body = f"""
    Detection Report
    ----------------
    Supervisor: {state.supervisor_name}
    Vehicle Number: {state.vehicle_number}
    Frames Processed: {state.frame_count}
    Total Boxes Counted: {state.box_count}
    Total Value: {state.total_value}
    Individual Counts: {', '.join([f'{label}: {count}' for label, count in state.crossed_counts.items()])}
    """
    send_email("your_email_address@example.com", "Detection Complete", report_body)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Box Detection API"}

@app.post("/start-detection")
async def start_detection(request: StartRequest, background_tasks: BackgroundTasks):
    global state, video_thread
    if state.is_running:
        return {"message": "Detection is already in progress."}
    
    state.is_running = True
    state.video_path = request.video_url
    state.vehicle_number = request.vehicle_number
    state.supervisor_name = request.supervisor_name
    state.frame_count = 0
    state.box_count = 0
    state.total_value = 0
    state.crossed_counts = {}
    state.start_time = time.time()
    
    print(f"Starting detection on video: {state.video_path} with model: {request.model_name}")
    video_thread = threading.Thread(target=run_video_processing, args=(request.model_name,))
    video_thread.daemon = True
    video_thread.start()
    
    return {"message": "Detection started. A pop-up window should appear."}

@app.post("/stop-detection")
async def stop_detection(request: StopRequest):
    global state, video_thread
    if not state.is_running:
        return {"message": "No active processing to stop."}
    
    state.is_running = False
    if video_thread and video_thread.is_alive():
        video_thread.join()
        
    report_body = f"""
    Detection Report (Manual Stop)
    -----------------------------
    Supervisor: {state.supervisor_name}
    Vehicle Number: {state.vehicle_number}
    Frames Processed: {state.frame_count}
    Total Boxes Counted: {state.box_count}
    Total Value: {state.total_value}
    Individual Counts: {', '.join([f'{label}: {count}' for label, count in state.crossed_counts.items()])}
    """
    send_email("your_email_address@example.com", "Detection Manually Stopped", report_body)
    
    reports.append({
        "supervisor_name": state.supervisor_name,
        "vehicle_number": state.vehicle_number,
        "frames_processed": state.frame_count,
        "total_boxes_counted": state.box_count,
        "total_value": state.total_value,
        "individual_counts": state.crossed_counts,
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    })
    
    return {"message": "Processing stopped."}
    
@app.get("/reports")
async def get_reports():
    return {"reports": reports}

@app.get("/video_feed")
async def video_feed_endpoint():
    """Endpoint to stream the video frames."""
    async def generate_frames():
        while state.is_running:
            try:
                frame_data = frame_queue.get(timeout=1)
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            except:
                continue
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get("/status")
def get_status():
    return {
        "is_running": state.is_running,
        "total_value": state.total_value,
        "crossed_counts": state.crossed_counts
    }