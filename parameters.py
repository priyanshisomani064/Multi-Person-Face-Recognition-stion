'''
This module contain the default parameters used by the app 
'''

#The default sender of emails
EMAIL_SENDER = 'support.ai@giindia.com'

#The default receiver of emails
EMAIL_RECEIVER = 'sugandh.gupta@giindia.com'

# The default email subject
EMAIL_SUBJECT = 'Face Recognition Summary Report'

# The default email body
EMAIL_BODY = 'Please find the attached Face Recognition Summary Report'

#Path where dataset is stored (used for creating face embedding)
DATASET_PATH = 'dataset/train/pics_dlib_abesit/'

# The path where dlib face encodings are stored
DLIB_FACE_ENCODING_PATH = 'dataset/dlib_face_encoding_9thApril.pkl'

# The path where face recognition report will be stored
REPORT_PATH = 'reports/inferred_faces.csv'

# Fles path where various supplementary files are stored
FILES_PATH = 'data/files/'

# The path of the demo video
VIDEO_PATH = 'data/files/vid.mp4'

# The path where the user video will be uploaded
VIDEO_UPLOAD_PATH = 'src/static/video/vid.mp4'

# The path where logs will be stored
LOG_FILE_PATH = 'logs/multicam_server.log'

# face matching tolerance (distance -> less the distance, more the similarity)
FACE_MATCHING_TOLERANCE = 0.4

#face recognition model
FACE_RECOGNITION_MODEL = 'hog' #hog -> for CPU or cnn -> for GPU (DGX)

# Number of times to upsample the image looking for faces
NUMBER_OF_TIMES_TO_UPSAMPLE = 1 # for realtime keep it to 1

# Set video frame height and width
FRAME_HEIGHT = 720 #576
FRAME_WIDTH = 1280 #10https://github.com/anubhavpatrick/Multi-Person-Face-Recognition.git24

# set BATCH_SIZE for face detection
BATCH_SIZE = 1 #for DGX 32, 1 for CPU

# buffer size for video streaming to minimize inconsistent network conditions
LIVE_STREAM_BUFFER_SIZE = 256 #single camera

# buffer size for frames on which face recognition will be performed
INFERENCE_BUFFER_SIZE = 256 #256 for DGX

# IP Camera Details
IP_CAMS = {'cam1': 'http://192.168.12.10:4747/video',
           #'cam2': 'http://192.168.1.16:4747/video',
          }

# Set wait duration for IP cam re initialization if we are not able to initialize the cam
IP_CAM_REINIT_WAIT_DURATION = 30 #seconds

# Time of the day when the email will be sent
EMAIL_SEND_TIME = '18:00' # 6 PM

# Wait duration for the email report to be sent
EMAIL_SEND_WAIT_DURATION = 15 #seconds
