import cv2
import numpy as np
import threading

_camera_shared = {

}

def on_trackbar_change_H(val):
    _camera_shared['h'] = val

def on_trackbar_change_S(val):
    _camera_shared['s'] = val

def on_trackbar_change_V(val):
    _camera_shared['v'] = val


def camera_thread(shared):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("HSV")
    cv2.createTrackbar("H", "HSV", shared['h'], 179, on_trackbar_change_H)
    cv2.createTrackbar("S", "HSV", shared['s'], 255, on_trackbar_change_S)
    cv2.createTrackbar("V", "HSV", shared['v'], 255, on_trackbar_change_V)

    while shared['running']:
        ret, frame = cap.read()
        if not ret:
            break

        frame_mirror = frame[:, ::-1, :]
        try:
            shared['frame_h'], shared['frame_w'] = frame_mirror.shape[:2]
        except Exception:
            pass

        image_HSV = cv2.cvtColor(frame_mirror, cv2.COLOR_BGR2HSV)

        h, s, v = shared['h'], shared['s'], shared['v']
        lower = np.array([max(0, h - 10), s, max(0, v - 50)])
        upper = np.array([min(179, h + 10), 255, min(255, v + 50)])

        mask = cv2.inRange(image_HSV, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output = frame_mirror.copy()
        if contours:
            maior_contorno = max(contours, key=cv2.contourArea)
            M = cv2.moments(maior_contorno)
            if M.get("m00", 0) != 0:
                cx = int(M.get("m10", 0) / M.get("m00", 1))
                cy = int(M.get("m01", 0) / M.get("m00", 1))
                shared['cx'], shared['cy'] = cx, cy
                cv2.circle(output, (cx, cy), 10, (0, 0, 255), -1)
                cv2.drawContours(output, [maior_contorno], -1, (0, 255, 0), 2)
            else:
                shared['cx'], shared['cy'] = None, None
        else:
            shared['cx'], shared['cy'] = None, None

        shared['mask'] = mask
        cv2.imshow("HSV", output)
        cv2.imshow("Mascara", mask)
        if cv2.waitKey(1) & 0xFF == 27:
            shared['running'] = False
            break

    cap.release()
    cv2.destroyAllWindows()
    shared['running'] = False


def should_shoot (cy):
    frame_h = _camera_shared.get('frame_h', 480)
    print(cy)
    if cy is not None and frame_h:
        try:
            top_third = float(frame_h) / 3.0
        except Exception:
            top_third = 160.0
        if cy < top_third:
            return True
    return None

