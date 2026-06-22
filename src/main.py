import cv2
import mediapipe as mp
import pyautogui
import math
import sys


CAMERA_DEVICE = "/dev/video10"
cap = cv2.VideoCapture(CAMERA_DEVICE)

if not cap.isOpened():
    print(f"Error: Could not open camera {CAMERA_DEVICE}. Exiting.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False 
SCREEN_W, SCREEN_H = pyautogui.size()


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils


SMOOTHING = 5
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0


FRAME_R = 200
CAM_W, CAM_H = 1280, 720


is_dragging = False
left_click_latch = False
right_click_latch = False

def calc_distance(p1, p2):
    """Calculate Euclidean distance between two MediaPipe landmarks."""
    return math.hypot(p2.x - p1.x, p2.y - p1.y)


while True:
    success, frame = cap.read()
    if not success:
        print("Error: Failed to read frame from camera. Exiting.")
        sys.exit(1)


    frame = cv2.flip(frame, 1)
    

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)


    cv2.rectangle(frame, (FRAME_R, FRAME_R), (CAM_W - FRAME_R, CAM_H - FRAME_R), (255, 0, 255), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


            lm = hand_landmarks.landmark
            thumb_tip = lm[4]
            index_tip = lm[8]
            middle_tip = lm[12]


            x3 = pyautogui.translate(index_tip.x * CAM_W, FRAME_R, CAM_W - FRAME_R, 0, SCREEN_W)
            y3 = pyautogui.translate(index_tip.y * CAM_H, FRAME_R, CAM_H - FRAME_R, 0, SCREEN_H)


            x3 = max(0, min(SCREEN_W, x3))
            y3 = max(0, min(SCREEN_H, y3))


            curr_x = prev_x + (x3 - prev_x) / SMOOTHING
            curr_y = prev_y + (y3 - prev_y) / SMOOTHING
            
            pyautogui.moveTo(curr_x, curr_y)
            
            prev_x, prev_y = curr_x, curr_y


            dist_thumb_index = calc_distance(thumb_tip, index_tip)
            dist_thumb_middle = calc_distance(thumb_tip, middle_tip)
            
            PINCH_THRESH = 0.05


            if dist_thumb_index < PINCH_THRESH and dist_thumb_middle < PINCH_THRESH:
                if not is_dragging:
                    pyautogui.mouseDown()
                    is_dragging = True
                    cv2.putText(frame, "DRAGGING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:

                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False


                if dist_thumb_index < PINCH_THRESH and dist_thumb_middle > PINCH_THRESH:
                    if not left_click_latch:
                        pyautogui.click()
                        left_click_latch = True
                        cv2.putText(frame, "LEFT CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                else:
                    left_click_latch = False


                if dist_thumb_middle < PINCH_THRESH and dist_thumb_index > PINCH_THRESH:
                    if not right_click_latch:
                        pyautogui.rightClick()
                        right_click_latch = True
                        cv2.putText(frame, "RIGHT CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                else:
                    right_click_latch = False


    cv2.imshow("Hand Mouse Control", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()