import cv2 as cv
import numpy as np

box = [(0, 0)]
def on_mouse(event, x, y, flags, frame):

    if event == cv.EVENT_LBUTTONDOWN:
         print 'Start Mouse Position: '+str(x)+', '+str(y)
         box[0] = (x, y)

    elif event == cv.EVENT_LBUTTONUP:
        print 'End Mouse Position: '+str(x)+', '+str(y)
        # ebox = [x, y]
        # boxes.append(ebox)

        print((box[0], (x, y)))

cap = cv.VideoCapture('resources/current_training/three objects/amb 4/amb4_o3_re5_wv.webm')

cv.namedWindow("original")
# start = cv.getTrackbarPos('position', 'original')

ret, frame_t1 = cap.read()
f_shape = frame_t1.shape

# percentages of empirical cropped image
crops_y = int(f_shape[0] - np.ceil((f_shape[0] * 0.12037)))
crops_x1 = int(np.ceil((f_shape[1] * 0.3125)))
crops_x2 = int(f_shape[1] - np.ceil((f_shape[1] * 0.3457)))

def onChange(trackbarValue):
    print(trackbarValue)

cv.createTrackbar( 'start', 'original', 0, 600, onChange)

while cv.waitKey(30) & 0xFF != ord('q'):
    c_pos = cv.getTrackbarPos('start','original')
    cap.set(cv.CAP_PROP_POS_FRAMES, c_pos)
    print(cap.get(cv.CAP_PROP_POS_FRAMES))
    err, frame = cap.read()
    frame = frame[0:crops_y, crops_x1:crops_x2]
    frame = cv.resize(frame, ((crops_x2 - crops_x1) / 2, crops_y / 2))
    cv.setMouseCallback("original", on_mouse, frame)
    cv.imshow("original", frame)

# cap.release()
# cv.destroyAllWindows()
