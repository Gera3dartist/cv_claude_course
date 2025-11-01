import cv2
import numpy as np


video_path = ('/Users/agerasymchuk/private_repo/cv_claude_course/cv_course/videos/color.mp4')
output_path = '/Users/agerasymchuk/private_repo/cv_claude_course/cv_course/videos/grayscale_video.mp4'


cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, isColor=False)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

cap.release()
out.release()
print('saved grayscale video')