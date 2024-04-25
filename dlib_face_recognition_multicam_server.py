'''
This module is a server that performs face recognition (using face_recognition library) 
on multiple video streams

'''

import time
import cv2
import threading

# Custom modules
from locks import lock
from dlib_face_recognition import batched_frame_face_recognition
from parameters import FRAME_HEIGHT, \
                       FRAME_WIDTH, \
                       LIVE_STREAM_BUFFER_SIZE, \
                       IP_CAMS, \
                       EMAIL_SENDER, \
                       EMAIL_RECEIVER, \
                       EMAIL_SUBJECT, \
                       EMAIL_BODY, \
                       REPORT_PATH, \
                       IP_CAM_REINIT_WAIT_DURATION
from custom_logging import logger
from database_pandas import store_dataframe_in_csv
from schedule_send_report import schedule_send_mail
from send_mail import prepare_and_send_email

# Create a buffer to store the frames (optionally from multiple IP Cams)
buffer = []


class IPCamera:
    '''A class to represent an IP camera'''

    def __init__(self, name, url, username=None, password=None):
        '''Initialize the camera'''
        self.name = name
        self.url = url
        self.username = username
        self.password = password 
        # initialize the video camera stream and read the first frame
        self.stream = cv2.VideoCapture(self.url)
        # set the buffer size to BUFFER_SIZE
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, LIVE_STREAM_BUFFER_SIZE)
        # we need to read the first frame to initialize the stream
        self.grabbed, _ = self.stream.read()
        # store whether the camera stream was initialized successfully
        self.is_initialized = self.grabbed
        # set the flag to process the frame
        self.process_this_frame = True
        if not self.grabbed:
            logger.error(f'Camera stream from {self.name} (url: {self.url})) unable to initialize')
        else:
            logger.info(f'Camera stream from {self.name} (url: {self.url}) initialized')

    
    def _read_one_frame(self):
        '''Reads a frame from the camera'''
        self.grabbed, self.frame = self.stream.read()

    
    def _read_and_discard_frame(self):
        '''Reads and discards one frame'''
        _, _ = self.stream.read()

    
    def place_frame_in_buffer(self):
        '''Places the frame in the buffer'''
        global buffer, lock
        if self.process_this_frame:
            self._read_one_frame()
            if not self.grabbed:
                # if the frame was not grabbed, then we have reached the end of the stream
                logger.error(f'Could not read a frame from the camera stream from {self.name} (url: {self.url})). Releasing the stream...')
                self.release()
                self.is_initialized = False
            else:
                # resize the frame if the frame size is larger than the frame size specified in parameters.py
                #if self.frame.shape[0] > FRAME_HEIGHT or self.frame.shape[1] > FRAME_WIDTH:
                self.frame = cv2.resize(self.frame, (FRAME_WIDTH, FRAME_HEIGHT))
                # Use lock to prevent multiple threads from accessing the buffer at the same time
                with lock:
                    # append the frame to the buffer along with the name and url of the camera
                    buffer.append([self.frame, self.name, self.url])
        else:
            self._read_and_discard_frame()
        
        # toggle the flag to process alternate frames to improve the performance
        self.process_this_frame = not self.process_this_frame

    
    def release(self):
        '''Releases the camera stream'''
        self.stream.release()


def create_camera(name, url, username=None, password=None): 
    '''Creates a camera object and places the frames in the buffer
    
    Args:
        name (str): name of the camera
        url (str): url of the camera
        username (str): username for the camera
        password (str): password for the camera

    Returns:
        None
    '''
    cam = IPCamera(name, url)
    # Place the frames in the buffer until the end of camera stream is reached
    while(True):
        if cam.is_initialized:
            try:
                cam.place_frame_in_buffer()
            except Exception as e:
                # if exception is raised, then release the camera stream and set the flag to False
                logger.error(f'Exception raised while placing the frame in the buffer from {cam.name} (url: {cam.url})). Releasing the stream...')
                cam.stream.release()
                cam.is_initialized = False
        else:
            # destroy the camera object since the camera stream was not initialized
            logger.error(f'Camera stream from {cam.name} (url: {cam.url})) is not accessible. Destroying the camera object...')
            del cam
            # put the thread to sleep for  second
            logger.info(f'Putting the thread to sleep for {name} (url: {url})) for {IP_CAM_REINIT_WAIT_DURATION} seconds...')
            time.sleep(IP_CAM_REINIT_WAIT_DURATION)
            # again try to recreate a new camera object
            logger.info(f'Creating a new camera object for {name} (url: {url}))...')
            cam = IPCamera(name, url)
            

# Create a thread for each camera and start the thread
for cam_name, cam_url in IP_CAMS.items():
    cam = threading.Thread(target=create_camera, args=(cam_name, cam_url))
    cam.start()

# Start a thread to schedule send email to the client at EMAIL_SEND_TIME
send_email_thread = threading.Thread(target=schedule_send_mail)
# start the thread
send_email_thread.start()

try:
    # Perform batched face recognition on the frames in the buffer
    batched_frame_face_recognition(buffer)
except KeyboardInterrupt:
    # Release the camera streams, store the dataframe in csv and exit the program
    for cam_name, cam_url in IP_CAMS.items():
        # To do - release the camera streams
        pass
    logger.info('Camera streams released')
    store_dataframe_in_csv()
    logger.info('Dataframe stored in csv')
    #Finally also send report to the client 
    prepare_and_send_email(EMAIL_SENDER,
                           EMAIL_RECEIVER, 
                           EMAIL_SUBJECT, 
                           EMAIL_BODY, 
                           REPORT_PATH)
    logger.info('Exiting the program')
