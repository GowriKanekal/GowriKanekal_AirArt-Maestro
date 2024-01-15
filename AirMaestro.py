import cv2 
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

#initialising the videocapture object 
v=cv2.VideoCapture(0)

#stores the co-ordinate of the tip of the index finger from the previous frame
prev_index_tip = None

# Initializing a blank canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
# creating a white background in the canvas 
canvas.fill(255)

while True:
    ret,frame=v.read()

    #flipping the frame in such a way that we directly see what we write (selfie mode)
    frame=cv2.flip(frame,1)

    #converting BGR into RGB fromat 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #analyses and obtains hand landmarks
    result = hands.process(rgb_frame)

    #checks if there is more than one hand
    if result.multi_hand_landmarks:

        #only taking the first hand into account if there are multiple hands detected
        landmarks = result.multi_hand_landmarks[0].landmark

        #x and y co-ordinates of the position of the tip of the index finger 
        index_tip = (int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]),
                     int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]))
        
        if prev_index_tip is not None:
            #specifiying the location of the index finger tip in the video frame through a green dot/line
            cv2.line(frame, prev_index_tip, index_tip,(0,255,0),2)

            #drawing a blue line on the canvas 
            cv2.line(canvas, prev_index_tip, index_tip,(255,0,0),5)

        #updating the value of the location of the index finger tip
        prev_index_tip = index_tip
    
    cv2.imshow('Video frame',frame)
    cv2.imshow('Canvas',canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#releasing the camera 
v.release()

#closing all windows 
cv2.destroyAllWindows()