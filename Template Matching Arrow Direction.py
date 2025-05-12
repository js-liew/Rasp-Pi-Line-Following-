import cv2
import numpy as np
import math
from picamera2 import Picamera2
import time
import os

os.makedirs("debug_images", exist_ok=True)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main = {"format" : "RGB888"}))
picam2.start()
time.sleep(2)

template_paths = {
    'up':    "arrow_up.jpg",
    'down':  "arrow_down.jpg",
    'left':  "arrow_left.jpg",
    'right': "arrow_right.jpg"
}

edge_templates = {}
for direction, path in template_paths.items():
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading template for {direction}: {path}")
        continue

    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(binary, 50, 150)

    cv2.imwrite(f"debug_images/{direction}_binary.jpg", binary)
    cv2.imwrite(f"debug_images/{direction}_edges.jpg", edges)

    edge_templates[direction] = edges

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def detect_by_template(roi, scales=np.linspace(0.8, 1.2, 10), min_confidence=0.65):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 50, 150)

    cv2.imwrite("debug_images/debug_edge.jpg", edge)

    h_roi, w_roi = edge.shape
    best_dir, best_score = None, -1

    for d, tpl in edge_templates.items():
        h_t, w_t = tpl.shape

        for s in scales:
            sw, sh = int(w_t * s), int(h_t * s)
            if sw < 10 or sh < 10 or sw > w_roi or sh > h_roi:
                continue

            tpl_rs = cv2.resize(tpl, (sw, sh), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(edge, tpl_rs, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, _ = cv2.minMaxLoc(res)

            if maxv > best_score:
                best_score = maxv
                best_dir = d
                best_tpl = tpl_rs

    if best_score >= min_confidence:
        cv2.imwrite(f"debug_images/best_template_{best_dir}.jpg", best_tpl)
        return best_dir, best_score
    else:
        return None, best_score

def detect_by_angle(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, KERNEL, iterations=2)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    arrow = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    M = cv2.moments(arrow)
    if M['m00'] == 0:
        return None

    cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
    d2 = (arrow[:, 0] - cx) ** 2 + (arrow[:, 1] - cy) ** 2
    tip = arrow[np.argmax(d2)]
    ang = math.degrees(math.atan2(tip[1] - cy, tip[0] - cx))

    if -45 <= ang <= 45:
        return 'right'
    elif 45 < ang <= 135:
        return 'down'
    elif ang > 135 or ang <= -135:
        return 'left'
    else:
        return 'up'

try:
    while True:
        frame = picam2.capture_array()
        h, w = frame.shape[:2]
        roi = frame[h//4:h*3//4, w//4:w*3//4]

        cv2.imwrite("debug_images/raw_roi_1.jpg", roi)

        direction, score = detect_by_template(roi)

        if direction is None:
            direction = detect_by_angle(roi)
            if direction is None:
                print("No arrow detected")
            else:
                print(f"Detected Direction: {direction.upper()}")
                time.sleep(0.5)
                annotated = roi.copy()
                cv2.putText(annotated, direction.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2)
                cv2.imwrite("debug_images/detected_direction.jpg", annotated)
        else:
            print(f"Detected Direction: {direction.upper()}")
            annotated = roi.copy()
            cv2.putText(annotated, direction.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            cv2.imwrite("debug_images/detected_direction.jpg", annotated)


except KeyboardInterrupt:
    print("Stopped")
    picam2.stop()
