'''A Flask application to run the face recognition app using dlib and OpenCV on stored video or
IP cam stream

'''

import os.path
import validators #for validating the ip cam url
from flask import Flask, render_template, request, Response
from dlib_face_recognition import multi_frame_face_recognition
from database_pandas import store_dataframe_in_csv
from send_mail import prepare_and_send_email
from threading import Thread
from parameters import EMAIL_SENDER, \
                       EMAIL_SUBJECT,\
                       EMAIL_BODY,\
                       REPORT_PATH, \
                       VIDEO_UPLOAD_PATH, \
                       VIDEO_PATH, \
                       FACE_MATCHING_TOLERANCE, \
                       FACE_RECOGNITION_MODEL

import cv2

# Initialize the Flask application
app = Flask(__name__)
app.config["VIDEO_UPLOADS"] = VIDEO_UPLOAD_PATH
app.config["ALLOWED_VIDEO_EXTENSIONS"] = ["MP4", "MOV", "AVI", "WMV"]

#secret key for the session
app.config['SECRET_KEY'] = 'face_recognition'

#global variables
frames_buffer = [] #buffer to store frames from a stream
vid_path = app.config["VIDEO_UPLOADS"] #it can be path to stored video or IP cam stream
video_frames = cv2.VideoCapture(vid_path) #video capture object


def allowed_video(filename: str):
    '''A function to check if the uploaded file is a video
    
    Args:
        filename (str): name of the uploaded file

    Returns:
        bool: True if the file is a video, False otherwise
    '''
    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1]

    if extension.upper() in app.config["ALLOWED_VIDEO_EXTENSIONS"]:
        return True
    else:
        return False


def generate_raw_frames():
    '''A function to yield unprocessed frames from stored video file or ip cam stream
    
    Args:
        None
    
    Yields:
        bytes: a frame from the video file or ip cam stream
    '''
    global video_frames

    while True:            
        # Keep reading the frames from the video file or ip cam stream
        success, frame = video_frames.read()

        if success:
            # create a copy of the frame to store in the buffer
            frame_copy = frame.copy()

            #store the frame in the buffer for violation detection
            frames_buffer.append(frame_copy) 
            
            #compress the frame and store it in the memory buffer
            _, buffer = cv2.imencode('.jpg', frame) 
            #convert the buffer to bytes
            frame = buffer.tobytes() 
            #yield the frame to the browser
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n') 


def generate_processed_frames(face_matching_tolerance= 0.45):
    '''A function to yield processed frames from stored video file or ip cam stream after face recognition
    
    Args:
        face_matching_tolerance (float): tolerance for face matching
    
    Yields:
        bytes: a processed frame from the video file or ip cam stream
    '''
    #call the face recognition function which yields a list of processed frames
    fr_output = multi_frame_face_recognition(frames_buffer,
                                             model= FACE_RECOGNITION_MODEL,
                                             face_matching_tolerance=face_matching_tolerance)

    #iterate through the list of processed frames
    for detection_ in fr_output:
        #The function imencode compresses the image and stores it in the memory buffer 
        _,buffer=cv2.imencode('.jpg',detection_)
        #convert the buffer to bytes
        frame=buffer.tobytes()
        #yield the processed frame to the browser
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')


@app.route('/video_raw')
def video_raw():
    '''A function to handle the requests for the raw video stream
    
    Args:
        None

    Returns:
        Response: a response object containing the raw video stream
    '''

    return Response(generate_raw_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_processed')
def video_processed():
    '''A function to handle the requests for the processed video stream after face recognition

    Args:
        None
    
    Returns:
        Response: a response object containing the processed video stream
    '''
    return Response(generate_processed_frames(face_matching_tolerance=FACE_MATCHING_TOLERANCE),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=["GET", "POST"])
def index():
    '''A function to handle the requests from the web page

    Args:
        None
    
    Returns:
        render_template: the index.html page (home page)
    '''
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit_form():
    '''A function to handle the requests from the HTML form on the web page

    Args:
        None
    
    Returns:
        str: a string containing the response message
    '''
    # global variables
    global vid_path, video_frames, frames_buffer

    #if the request is a POST request made by user interaction with the HTML form
    if request.method == "POST":
        
        # handle video upload request
        if request.files:
            video = request.files['video']
            
            #check if video file is uploaded or not
            if video.filename == '':
                # display a flash alert message on the web page
                return "That video must have a file name"

            #check if the uploaded file is a video
            elif not allowed_video(video.filename):
                # display a flash alert message on the web page
                return "Unsupported video. The video file must be in MP4, MOV, AVI or WMV format."
            
            else:
                # ensure video size is less than 200MB
                if video.content_length > 200*1024*1024:
                    return "Error! That video is too large"
                else:
                    try:
                        video.save(vid_path)
                        return "That video is successfully uploaded"
                    except Exception as e:
                        print(e)
                        return "Error! The video could not be saved"
        
        # handle download request for the detections summary report
        if 'download_button' in request.form:            
            # store the face recognition summary report in a csv file
            store_dataframe_in_csv()

            #To do - add a check to see if the file exists
            return Response(open(REPORT_PATH, 'rb').read(),
                        mimetype='text/plain',
                        headers={"Content-Disposition":"attachment;filename=inferred_faces.csv"})

        # handle alert email request
        elif 'alert_email_checkbox' in request.form:
            email_checkbox_value = request.form['alert_email_checkbox']
            if email_checkbox_value == 'false':
                return "Alert email is disabled"    
            else: 
                report_recipient = request.form['alert_email_textbox']

                # create a thread that will send the email with attached report
                t = Thread(target=prepare_and_send_email, args=(EMAIL_SENDER, 
                                                                report_recipient, 
                                                                EMAIL_SUBJECT, 
                                                                EMAIL_BODY, 
                                                                REPORT_PATH))
                t.start()
                return f"The face recognition report is sent to {report_recipient}"
        
        # handle inference request for a video file
        elif 'inference_video_button' in request.form:
            vid_path = app.config["VIDEO_UPLOADS"]
            video_frames = cv2.VideoCapture(vid_path)
            frames_buffer.clear()
            # check if the video is opened
            if not video_frames.isOpened():
                return 'Error in opening video',500
            else:
                frames_buffer.clear()
                return 'success'
        
        # handle inference request for a live stream via IP camera
        elif 'live_inference_button' in request.form:
            #read ip cam url from the text box
            vid_ip_path = request.form['live_inference_textbox']
            #check if vid_ip_path is a valid url
            if validators.url(vid_ip_path):
                vid_path = vid_ip_path.strip()
                video_frames = cv2.VideoCapture(vid_path)
                #Set the buffer size to 1 for the IP camera stream to avoid lagging
                # Reference: https://devpress.csdn.net/python/6304568e7e6682346619a208.html
                video_frames.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                #check connection to the ip cam stream
                if not video_frames.isOpened():
                    # display a flash alert message on the web page
                    return 'Error: Cannot connect to live stream',500
                else:
                    frames_buffer.clear()
                    return 'success'
            else:
                # the url is not valid
                return 'Error: Entered URL is invalid',500


if __name__ == "__main__":
    #copy video from default dir to video uploads directory
    os.system(f'cp {VIDEO_PATH} {app.config["VIDEO_UPLOADS"]}')
    app.run(host='0.0.0.0', debug=False)
