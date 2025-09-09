import numpy as np
import cv2 as cv

drawing = False
mode = True  # True for rectangle, False for circle

def draw_shape(event,x,y,flags,param):
    print('inside a callback')
    global drawing, mode
    
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        if mode:
            cv.rectangle(img,(x-25,y-25),(x+25,y+25),(0,255,0),-1)
        else:
            cv.circle(img,(x,y),25,(0,0,255),-1)
        print(f'Picked point: x: {x}, y: {y}')
    
    elif event == cv.EVENT_RBUTTONDOWN:
        mode = not mode  # Switch between rectangle and circle
        print(f"Mode: {'Rectangle' if mode else 'Circle'}")

img = np.zeros((512,512,3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image',draw_shape)

print("Left click to draw, right click to switch modes, ESC to exit")

while True:
    cv.imshow('image',img)
    key = cv.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('c'):  # Clear screen
        img[:] = 0

cv.destroyAllWindows()


