import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

def main():
    def set_background_image(image_url):
        st.markdown(
            f"""
               <style>
               .stApp {{
                   background-image: url("{image_url}");
                   background-size: cover;
                   background-position: center;
                   background-repeat: no-repeat;
                   background-attachment: fixed;
               
               }}
               </style>
               """,
            unsafe_allow_html=True)

    set_background_image('https://i.pinimg.com/736x/80/72/30/80723069e7f8de1ceace8909ef9bad92.jpg')

    st.markdown(
        """
        <style>
        /* Set all text to deep wine red */
        * {
        color: #dea7a6 !important;
        }
        </style>
        """,
    unsafe_allow_html=True
    )       

        
    model=load_model("model/emotion_model.h5")
    face_cascade=cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    emotions=['angry','disgust','fear','happy','neutral','sad','surprise']

    def detect_emotion(frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            face=gray[y:y+h,x:x+w]
            face=cv2.resize(face,(48,48))
            face=face.astype("float32")/255.0
            face=img_to_array(face)
            face=np.expand_dims(face,axis=0)

            prediction=model.predict(face)[0]
            emotion_label=emotions[np.argmax(prediction)]

            cv2.putText(frame,emotion_label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

        return frame    



    st.title("REAL-TIME HUMAN EMOTION DETECTION")

    st.sidebar.title("Select Page:")
    page = st.sidebar.radio("Go to", ["Home", "Live Detection", "Contact"])

    if page=="Home":
        st.markdown("""
                    Welcome to the **Human Emotion Detection App**!  This app uses a deep learning model trained on the FER2013 dataset to detect emotions from facial expressions in real-time. 
                    Whether you're curious about how it works or want to try it out yourself, you're in the right place!
                    """)
        
        st.header(" Key Features")

        st.markdown("""
                    - **Real-Time Emotion Detection**: Detect emotions like 😠 **Angry**, 😄 **Happy**, 😢 **Sad**, 😲 **Surprise**, 😨 **Fear**, 😐 **Neutral**, and 🤢 **Disgust**.
                    - **Live Webcam Integration**: Use your webcam to see the app in action.
                    - **Powered by Deep Learning**: Built with Convolutional Neural Network(CNN) and OpenCV for accurate and fast predictions.
                    - **User-Friendly Interface**: Simple and intuitive design for seamless interaction.
                    """)
        
        st.header(" How It Works")

        st.markdown("""
                    1. **Open the Webcam**: Click on the **Live Detection** page to start your webcam.
                    2. **Face Detection**: The app will detect your face in real-time using OpenCV.
                    3. **Emotion Prediction**: The deep learning model will analyze your facial expression and predict your emotion.
                    4. **See the Results**: The predicted emotion will be displayed on the screen along with a bounding box around your face.
                    """)
    elif page=="Live Detection":
        st.title("Real Time Emotion Detection")
        st.write("Press the button to open webcam and detect your emotions!")
        run=st.button("Open Webcam")
        

        if run:
            video=cv2.VideoCapture(0)
            stframe=st.empty()
            stop=st.button("Stop Webcam")

            while True:
                suc,frame=video.read()
                if not suc:
                    st.write("Failed to capture image.")
                    break

                frame=detect_emotion(frame)
                frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb,channels="RGB")

                # 
                if stop:
                    break

            video.release()
            #cv2.destroyAllWindows()    

        

        
    elif page == "Contact":
        st.title("Contact Me 📧")
        st.markdown("""
                    Have questions or feedback? Feel free to reach out!
                    """)
        st.write("Email: shahabasali751@gmail.com.com")
        st.write("GitHub: [GitHub](https://github.com/shahabas123)")
        st.write("LinkedIn: [LinkedIn](https://www.linkedin.com/in/shahabas-ali-8-/)")    
        




    


main()