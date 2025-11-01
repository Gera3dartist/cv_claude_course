import cv2
import numpy as np


video_path = ('/Users/agerasymchuk/private_repo/cv_claude_course/cv_course/videos/color.mp4')

lab_path = '/Users/agerasymchuk/private_repo/cv_claude_course/cv_course/videos/lab_video.mp4'


cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

lab_writer = cv2.VideoWriter(lab_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    lab_writer.write(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))

cap.release()
lab_writer.release()
print('saved rgb  and bgr video')