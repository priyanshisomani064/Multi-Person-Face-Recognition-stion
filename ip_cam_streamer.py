'''This module acts as relay streamer that broadcasts the video stream from the local IP camera 
to the client over the internet

ngrok utility is used to map the local IP camera to a public URL

For more information on ngrok, visit https://ngrok.com/

'''

import argparse
import cv2
from flask import Flask, Response

from parameters import LIVE_STREAM_BUFFER_SIZE

app = Flask(__name__)
#secret key for the session
app.config['SECRET_KEY'] = 'ip_cam_streamer'

#Set the video path (ip camera url)
VIDEO_PATH = None


def video_feed():

    # Open the video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, LIVE_STREAM_BUFFER_SIZE)

    print('Initialising camera stream...')
    while True:
        ret, _ = cap.read()
        #find resolution of the video
        if ret:
            print('Initialised!!!')
            break 

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if ret:
            # Yield the frame as a response to the client
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n')


@app.route('/video')
def video():
    return Response(video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Run the Flask app
if __name__ == '__main__':

    #Read run time parameters from the command line - host and port
    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--app', type=str, default='0.0.0.0', help='Host app address')
    parser.add_argument('-p','--port', type=int, default=5000, help='Port number')
    parser.add_argument('-c','--cam_ip', type=str, help='(HTTP) IP address of the IP camera', required=True)
    args = parser.parse_args()

    #Update VIDEO_PATH
    VIDEO_PATH = args.cam_ip 

    app.run(host=args.app, port=args.port)
