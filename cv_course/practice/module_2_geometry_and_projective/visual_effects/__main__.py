import numpy as np

import cv2 as cv
from .transformers import AbstractTransformation

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
width, height = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
transformation = AbstractTransformation(width, height, density=1000)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here

    # rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # BGR → RGB
    # Display the resulting frame
    abstract = transformation.triangular_abstraction(frame)
    # abstract = transformation.create_voronoi_abstraction(rgb)
    # abstract = transformation.fractal_mirror_transform(frame)
    cv.imshow('frame', abstract)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()