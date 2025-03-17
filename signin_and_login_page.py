import cv2
import os
import numpy as np
from pathlib import Path
from customtkinter import *
import face_recognition as fr
import pyttsx3


engine = pyttsx3.init()
voices = engine.getProperty('voices')  
rate = engine.getProperty('rate')
engine.setProperty('rate', 125)
engine.setProperty('voice', voices[1].id)


def opening_page():

    global app, register_label, register_frame ,tabview ,name_entry  
    app = CTk()
    app.geometry('600x600')
    set_default_color_theme("blue")

    tabview = CTkTabview(app)
    tabview.pack(padx=40, pady=40)

    tabview.add('ورود')
    tabview.add('ورزش‌ها')
    login_tab = tabview.tab("ورود")
    exercise_tab = tabview.tab("ورزش‌ها")

    title_label = CTkLabel(login_tab, text="در صورت نیاز ثبت نام کنید و اگر در سیستم ثبت هستید دکمه ی ورود را انتخاب کنید" ,font=("Arial", 15, "bold"))
    title_label.pack(pady=30,padx=10)

    login_button = CTkButton(login_tab, text="ورود", command=lambda: login_and_proceed(tabview, exercise_tab),corner_radius=32, border_width=2, fg_color='#4158D0', hover_color='#C850C0')
    login_button.pack(pady=10)

   
    register_button = CTkButton(login_tab, text="نام ثبت", command=register, corner_radius=32, border_width=2, fg_color='#4158D0', hover_color='#C850C0')
    register_button.pack(pady=10)

    register_label = CTkLabel(login_tab, text="", font=("Arial", 20, "bold"))
    register_label.pack(pady=10)

    register_frame = CTkFrame(login_tab, fg_color='#C850C0')
    register_frame.pack(pady=15)
    register_frame.pack_forget()  

    name_label = CTkLabel(register_frame, text="کنید وارد انگلیسی به را خود نام")
    name_label.pack(pady=10, padx=30)

    name_entry = CTkEntry(register_frame)
    name_entry.pack(pady=10, padx=30)

    submit_button = CTkButton(register_frame, text="ثبت", command=lambda: save_name(name_entry), corner_radius=32, border_width=2, fg_color='#4158D0', hover_color='#C850C0')
    submit_button.pack(pady=10, padx=30)


    app.mainloop()



def register():
    register_frame.pack(pady=15) 

def save_name(name_entry):
    name = name_entry.get().strip()
    if name:
        success = register_face(name, register_label) 
        if success:
            register_label.configure(text='اطلاعات با موفقیت ثبت شد!', font=("Arial", 20, "bold"))
        else:
            register_label.configure(text="چهره‌ای شناسایی نشد! دوباره امتحان کنید.")
        register_frame.pack_forget()  
    else:
        register_label.configure(text="لطفاً نام خود را وارد کنید!")



def register_face(name, register_label):

    if not os.path.exists("images"):
        os.makedirs("images")

    Path(f"images/{name}").mkdir(parents=True, exist_ok=True)

    cam = cv2.VideoCapture(0)
    

    while True:

        ret, frame = cam.read()
        if not ret:
            break
        
        cv2.putText(frame, "Press -space-  to register a new face ", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("registeration", frame)


        k = cv2.waitKey(1)
        if k % 256 == 27:
            cam.release()
            cv2.destroyAllWindows()
            return

        elif k % 256 == 32: 
            img_name = f"images/{name}/{name}.png"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} saved!")

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = fr.face_encodings(img_rgb)

            if encodings:
                np.save(f"images/{name}/{name}_encodings.npy", encodings[0])
                print("face features saved!")
                cam.release()
                cv2.destroyAllWindows()
                return True
            else:
                register_label.configure(text="نشد شناسایی چهره ای")

                cam.release()
                cv2.destroyAllWindows()
                return False


def login():
    global name
    face_detected = False
    
    known_encodings = {}
    known_names = []

    for person in os.listdir("images"):
        encoding_path = f"images/{person}/{person}_encodings.npy"
        if os.path.exists(encoding_path):
            known_encodings[person] = np.load(encoding_path)
            known_names.append(person)

    cam = cv2.VideoCapture(0)
   
    while True:
        ret, frame = cam.read()
        
        if not ret:
            print("Error: Failed to read frame from camera.")
            break

        cv2.putText(frame, "Press q to enter", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = fr.face_locations(img_rgb)
        encodings = fr.face_encodings(img_rgb, locations)

        if not locations:  
            cv2.putText(frame, "No faces detected", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("ورود", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue 

        
        for i, face_encoding in enumerate(encodings):
            matches = fr.compare_faces(list(known_encodings.values()), face_encoding)

            name = 'Unknown'

            face_distances = fr.face_distance(list(known_encodings.values()), face_encoding)
            
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances) 
                if matches[best_match_index]: 
                    name = known_names[best_match_index]  
                    
                    top, right, bottom, left = locations[i]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3) 
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    face_detected = True  

                else:
                    top, right, bottom, left = locations[i]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)  
                    cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    face_detected = False

                    
        cv2.imshow("ورود", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            if face_detected:
                engine.say(f"welcome {name}")
                engine.runAndWait()
            else:
                engine.say("Access Denied")
                engine.runAndWait()
            break

        

    cam.release()
    cv2.destroyAllWindows()

    return face_detected



def login_and_proceed(tabview, exercise_tab):
      
    if login():

        tabview.set("ورزش‌ها")

        def open_squat_butt():
            from exercises.squat_func import squat_section
            squat_section()
            

        def open_pushup_butt():
            from exercises.pushup_func import pushup_section
            pushup_section()

        def open_barfix_butt():
            from exercises.barfix_func import barfix_section
            barfix_section()


        welcome_label = CTkLabel(exercise_tab, text=f"Welcome {name}", font=("Arial", 20))
        welcome_label.pack(pady=10)
       
        button_squat = CTkButton(exercise_tab, text="اسکوات", command=open_squat_butt ,corner_radius=32, border_width=2, fg_color='#4158D0', hover_color='#C850C0')
        button_squat.pack(pady=5)

        button_pushup = CTkButton(exercise_tab, text="شنا", command=open_pushup_butt, corner_radius=32, border_width=2, fg_color='#4158D0', hover_color='#C850C0')
        button_pushup.pack(pady=5)

        button_barfix = CTkButton(exercise_tab, text="بارفیکس", command=open_barfix_butt, corner_radius=32, border_width=2, fg_color='#4158D0', hover_color='#C850C0')
        button_barfix.pack(pady=5)

    

        app.mainloop()


#opening_page()





# Sources


# tkinter page : https://youtu.be/Miydkti_QVE?si=LGhIO0YHuxHgLovv   یوتیوب  

#source of functions: https://youtu.be/9lHaIFDIKtE?si=Af0WzVUqMg1D_vTV   یوتیوب  
#https://github.com/RadiantCoding/Code  گیت هاب فایل

#aiolearn face_recognition sources (ده پروژه ی عملی ماشین لرنینگ)