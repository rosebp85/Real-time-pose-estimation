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
    with open('pushup_model.pkl', 'rb') as file:
        model = pickle.load(file)
        print("model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")


with open('scaler_pushup.pkl', 'rb') as f:
    scaler = pickle.load(f)


def pushup_section():

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
    worng_sound = pygame.mixer.Sound('buzzer-buzzing-single-fascinatedsound-2-00-01.mp3')


    pushup_counter = 0
    pushup_stage = None


    engine.say('Your exercise will start')
    engine.runAndWait()

    #countdown on screen
    for i in range(10, -1, -1):  

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1) 

        sentence = "Your exercise will start!"
        cv2.putText(frame, sentence, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(i), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 100, 255), 3, cv2.LINE_AA)
      
        cv2.imshow("pushup", frame)
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
                lefthip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                leftknee= [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                leftelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y] 
                leftshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                leftWrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y] 
                leftankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y] 

                righthip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                rightknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                rightelbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y] 
                rightshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                rightWrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y] 
                rightankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y] 


                #caculate angle
                angleOFknee_left = calculate_angle(lefthip, leftknee, leftankle)
                angleOFknee_right = calculate_angle(righthip, rightknee, rightankle)
                angleOFhip_left = calculate_angle(leftshoulder, lefthip, leftknee)
                angleOFhip_right = calculate_angle(rightshoulder, righthip, rightknee)
                angleOFshoulder_left = calculate_angle(leftelbow, leftshoulder,lefthip)
                angleOFshoulder_right = calculate_angle(rightelbow, rightshoulder, righthip)
                angleOFelbow_left = calculate_angle(leftshoulder, leftelbow, leftWrist)
                angleOFelbow_right =  calculate_angle(rightshoulder, rightelbow, rightWrist)
                
                #pushup counter logic
                if (angleOFelbow_left > 170 and angleOFelbow_right > 170):
                    pushup_stage = 'up'

                if (angleOFelbow_left < 95 and angleOFelbow_right < 95) and (angleOFhip_left >= 150 and angleOFhip_right >= 150)and (150 < angleOFknee_left < 180 and 150 < angleOFknee_right < 180) and pushup_stage == 'up':
                    pushup_stage = 'down'
                    pushup_counter += 1

                #incorrecr error
                if not (angleOFhip_left >= 150 and angleOFhip_right >= 150):
                    cv2.putText(image, "Keep your hips straight", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    #engine.say('Keep your hips straight')
                    #engine.runAndWait()   # به دلیل ضعیف شدن اجرای برنامه کامنت میشه
                                           # در صورت داشتن پردازنده ی قوی تر این مورد باز میشه

                if not (150 < angleOFknee_left < 180 and 150 < angleOFknee_right < 180):
                    cv2.putText(image, "Keep your knees straight", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    #engine.say('Keep your knees straight')
                    #engine.runAndWait()

                #visualize knee angle 
                min_radius_left1 = 12
                max_radius_left2 = 28
                knee_radius_left = int(np.interp(angleOFknee_left, [0, 180], [max_radius_left2, min_radius_left1])) 
                
                min_radius_right = 12
                max_radius_right = 28
                knee_radius_right = int(np.interp(angleOFknee_right, [0, 180], [max_radius_right, min_radius_right]))

                cv2.circle(image, tuple(np.multiply(leftknee, [640, 480]).astype(int)),
                            knee_radius_left,(0, 255, 0) if 150 < angleOFknee_left <= 180 else (0, 0, 255), -1)
                cv2.circle(image, tuple(np.multiply(rightknee, [640, 480]).astype(int)),
                            knee_radius_right,(0, 255, 0) if 150 <angleOFknee_right <= 180 else (0, 0, 255), -1)
                
                #visualize hips angle 
                min_radius_left3 = 12
                max_radius_left4 = 28
                hip_radius_left = int(np.interp(angleOFhip_left, [0, 180], [max_radius_left4, min_radius_left3])) 
                
                min_radius_right5 = 12
                max_radius_right6 = 28
                hip_radius_right = int(np.interp(angleOFhip_right, [0, 180], [max_radius_right6, min_radius_right5]))

                cv2.circle(image, tuple(np.multiply(lefthip, [640, 480]).astype(int)),
                            hip_radius_left,(0, 255, 0) if 150 < angleOFhip_left <= 180 else (0, 0, 255), -1)
                cv2.circle(image, tuple(np.multiply(righthip, [640, 480]).astype(int)),
                            hip_radius_right,(0, 255, 0) if 150 < angleOFhip_right <= 180 else (0, 0, 255), -1)
                
                #visualize shoulder angle 
                #min_radius_left7 = 12
                #max_radius_left8 = 28
                #shoulder_radius_left = int(np.interp(angleOFshoulder_left, [0, 180], [max_radius_left8, min_radius_left7])) 
                
                #min_radius_right9 = 12
                #max_radius_right10 = 28
                #shoulder_radius_right = int(np.interp(angleOFshoulder_right, [0, 180], [max_radius_right10, min_radius_right9]))

                #cv2.circle(image, tuple(np.multiply(leftshoulder, [640, 480]).astype(int)),
                           # shoulder_radius_left,(0, 255, 0) if 15 < angleOFshoulder_left < 85 else (0, 0, 255), -1)
                #cv2.circle(image, tuple(np.multiply(rightshoulder, [640, 480]).astype(int)),
                           # shoulder_radius_right,(0, 255, 0) if 15 < angleOFshoulder_right < 85 else (0, 0, 255), -1)
                
                #visualize elbow angle 
                min_radius_left11 = 12
                max_radius_left12 = 28
                elbow_radius_left = int(np.interp(angleOFelbow_left, [0, 180], [max_radius_left12, min_radius_left11])) 
                
                min_radius_right13 = 12
                max_radius_right14 = 28
                elbow_radius_right = int(np.interp(angleOFelbow_right, [0, 180], [max_radius_right14, min_radius_right13]))

                cv2.circle(image, tuple(np.multiply(leftelbow, [640, 480]).astype(int)),
                            elbow_radius_left,(0, 255, 0) if  angleOFelbow_left < 95 else (0, 0, 255), -1)
                cv2.circle(image, tuple(np.multiply(rightelbow, [640, 480]).astype(int)),
                            elbow_radius_right,(0, 255, 0) if angleOFelbow_right < 95 else (0, 0, 255), -1)


                input_data = np.array([[angleOFknee_left, angleOFknee_right, angleOFhip_left ,angleOFhip_right ,angleOFelbow_left ,angleOFelbow_right ,angleOFshoulder_left ,angleOFshoulder_right]])
                scaler.feature_names_in_ = None
                input_data_scaled = scaler.transform(input_data)
                prediction = model.predict(input_data_scaled)
                probabilities = model.predict_proba(input_data_scaled)
 
                correct_prob = probabilities[0][0]
                correct_percentage = int(correct_prob*100)
                cv2.putText(image, f'Accuracy: {correct_percentage}%', (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (120,0,0),3)
            

                if prediction[0] == 0 and pushup_stage == 'down':
                    cv2.putText(image, 'Correct', (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),3)

                    if not sound_play:
                        engine.say('Well done')
                        engine.runAndWait()
                        sound_play = True

                elif pushup_stage == 'up':
                    sound_play = False


            #render curl counter
            #status box
            cv2.rectangle(image, (0, 0), (280, 73), (0, 165, 255), -1)
            cv2.putText(image, 'PERS', (15,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(pushup_counter), (8,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2, cv2.LINE_AA) 
            #stage data
            cv2.putText(image, 'STAGE', (120,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, pushup_stage, (100,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2, cv2.LINE_AA) 

            
            #render detections
            mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(0,165,255), thickness=2, circle_radius=2)
                                      )
                        

            cv2.imshow('pushup', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()



#pushup_section()