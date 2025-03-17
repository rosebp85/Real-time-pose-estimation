import cv2
import mediapipe as mp
import numpy as np
import pygame
import pyttsx3
import pickle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

sound_play = False

engine = pyttsx3.init()
voices = engine.getProperty('voices') 
engine.setProperty('voice', voices[1].id)

try:
    with open('barfix_model.pkl', 'rb') as file:
        model = pickle.load(file)
        print("model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")


with open("scaler_barfix.pkl", "rb") as f:
    scaler = pickle.load(f)


def barfix_section():

    def calculate_angle(a,b,c):
        a= np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle > 180.0:
            angle = 360-angle

        return angle



    cap = cv2.VideoCapture(0)

    pygame.mixer.init()
    sound = pygame.mixer.Sound('whoosh-ding-gfx-sounds-1-00-02.mp3')

    barfix_couter = 0
    barfix_stage = None

    engine.say('Your exercise will start')
    engine.runAndWait()
    

    #countdown on screen
    for i in range(10, -1, -1):  

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  
        sentence = "Your exercise will start!"
        cv2.putText(frame, sentence, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(i), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 100, 255), 3, cv2.LINE_AA)
        
        cv2.imshow("brfix", frame)
        cv2.waitKey(1000)

    sound.play() 


    with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:

        while cap.isOpened():

            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            #make detection
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #extract landmarks
            if results.pose_landmarks:
                landmarks =  results.pose_landmarks.landmark

                #get coordinates
                leftShoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                leftElbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                leftWrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y] 
                leftshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                leftelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y] 
                lefthip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                rightelbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y] 

                rightShoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                rightElbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                rightWrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y] 
                rightshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                righthip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
   
                #caculate angle
                angleOFelbow_left = calculate_angle(leftShoulder, leftElbow, leftWrist)
                angleOFelbow_right = calculate_angle(rightShoulder, rightElbow, rightWrist)
                angleOFshoulder_left = calculate_angle(leftelbow, leftshoulder,lefthip)
                angleOFshoulder_right = calculate_angle(rightelbow, rightshoulder, righthip)
     
                
                #barfix counter logic
                if angleOFelbow_left> 120 and angleOFelbow_right >120:
                    barfix_stage = 'down'
                if angleOFelbow_left< 90 and angleOFelbow_right < 90 and barfix_stage == 'down':
                    barfix_stage = 'up'
                    barfix_couter +=1
                

                #visualize angle
                min_radius_left = 10
                max_radius_left = 30
                radius1 = int(np.interp(angleOFelbow_left, [0, 180], [max_radius_left, min_radius_left])) 
                
                min_radius_right = 10
                max_radius_right = 30
                radius2 = int(np.interp(angleOFelbow_right, [0, 180], [max_radius_right, min_radius_right]))

                cv2.circle(image, tuple(np.multiply(leftElbow, [640, 480]).astype(int)),
                            radius1,(0, 255, 0) if angleOFelbow_left < 80 else (0, 0, 255), -1)
                cv2.circle(image, tuple(np.multiply(rightElbow, [640, 480]).astype(int)),
                            radius2,(0, 255, 0) if angleOFelbow_right < 80 else (0, 0, 255), -1)

                input_data = np.array([[angleOFelbow_left, angleOFelbow_right, angleOFshoulder_left, angleOFshoulder_right]])
                scaler.feature_names_in_ = None
                input_data_scaled = scaler.transform(input_data)
                prediction = model.predict(input_data_scaled)
                probabilities = model.predict_proba(input_data_scaled)

                correct_prob = probabilities[0][0]
                correct_percentage = int(correct_prob*100)
                cv2.putText(image, f'Accuracy: {correct_percentage}%', (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (120,0,0),3)
            

                if prediction[0] == 0 and barfix_stage == 'up':
                    cv2.putText(image, 'Correct', (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),3)

                    if not sound_play:
                        engine.say('Well done')
                        engine.runAndWait()
                        sound_play = True

                elif barfix_stage == 'down':
                    sound_play = False



            #render curl counter
            #status box
            cv2.rectangle(image, (0, 0), (280, 73), (0, 165, 255), -1)
            cv2.putText(image, 'PERS', (15,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(barfix_couter), (8,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2, cv2.LINE_AA) 
            #stage data
            cv2.putText(image, 'STAGE', (120,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, barfix_stage, (100,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2, cv2.LINE_AA) 

            
            #render detections
            mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(0,165,255), thickness=2, circle_radius=2)
                                    )
 

            cv2.imshow('barfix', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release() 
        cv2.destroyAllWindows()


#barfix_section()