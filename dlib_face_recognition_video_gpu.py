# -*- coding: utf-8 -*-


#importing the required libraries
import pickle
import time
import cv2
import face_recognition
import numpy as np

from parameters import VIDEO_PATH, \
     NUMBER_OF_TIMES_TO_UPSAMPLE, \
     FRAME_WIDTH, \
     FRAME_HEIGHT, \
     FILES_PATH, \
     BATCH_SIZE, \
     DLIB_FACE_ENCODING_PATH,\
     FACE_MATCHING_TOLERANCE

# Find start time
tick = time.time()

#capture the video from stored video file
vid_capture = cv2.VideoCapture(VIDEO_PATH)
#print frame count
print("Total frames: ", int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT)))

# grab and initialize the stream
vid_capture.grab() 

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(DLIB_FACE_ENCODING_PATH,"rb").read())

#save the encodings and the corresponding labels in seperate arrays in the same order
known_face_encodings = data["encodings"]
known_face_names = data["names"]

#initialize the array variable to hold all face locations, encodings and names 
all_face_locations = []
all_face_encodings = []
all_face_names = []
all_processed_frames = []

frames_processed = 0
frames_buffer = []

# we will process only alternate frames to speed up the process
process_this_frame = False

#loop through every frame in the video
while True:
    #get the current frame from the video stream as an image
    ret,current_frame = vid_capture.read()

    # process only alternate frames to speed up the process
    process_this_frame = not process_this_frame
    if not process_this_frame:
        continue

    if ret:
        frames_processed += 1

        #resize frame to FRAME_WIDTHxFRAME_HEIGHT to display the video if frame is too big
        if current_frame.shape[1] > FRAME_WIDTH or current_frame.shape[0] > FRAME_HEIGHT:
            current_frame_small = cv2.resize(current_frame, (FRAME_WIDTH, FRAME_HEIGHT))
        else:
            current_frame_small = current_frame

        #append the current frame to the buffer
        frames_buffer.append(current_frame_small)

        if len(frames_buffer) == BATCH_SIZE:

            batch_of_face_locations = face_recognition.batch_face_locations(frames_buffer, 
                                                                            number_of_times_to_upsample=NUMBER_OF_TIMES_TO_UPSAMPLE,
                                                                            batch_size=BATCH_SIZE)
        
            for current_frame_small,all_face_locations in zip(frames_buffer,batch_of_face_locations):
                # uncomment the below line to see the face locations
                #for all_face_locations in batch_of_face_locations:
                #    print(all_face_locations)

                #detect face encodings for all the faces detected
                all_face_encodings = face_recognition.face_encodings(current_frame_small,all_face_locations)

                #looping through the face locations and the face embeddings
                for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
                    #splitting the tuple to get the four position values of current face
                    top_pos,right_pos,bottom_pos,left_pos = current_face_location
                    
                    #find all the matches and get the list of matches
                    all_matches = face_recognition.face_distance(known_face_encodings, current_face_encoding)
                
                    # Find the best match (smallest distance to a known face)
                    best_match_index = np.argmin(all_matches)
                    
                    # If the best match is within tolerance, use the name of the known face
                    if all_matches[best_match_index] <= FACE_MATCHING_TOLERANCE:
                        name_of_person = known_face_names[best_match_index]
                        color = (0,255,0) #Green
                    else:
                        name_of_person = 'Unknown'
                        color = (0,0,255) #Red

                    #draw rectangle around the face    
                    cv2.rectangle(current_frame_small,(left_pos,top_pos),(right_pos,bottom_pos),color, 2)
                    
                    #display the name as text in the image
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(current_frame_small, name_of_person, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)

                #add current processed frame to the all_processed_frames array
                all_processed_frames.append(current_frame_small)
        
            # clear the frames buffer after processing the batch
            frames_buffer.clear()
        
    else:
        break

# Find end time
tock = time.time()

# Calculate total time
total_time = tock - tick
print("Total time taken: ", total_time)

# Calculate FPS
fps = frames_processed / total_time
print("FPS: ", fps)

#release the stream and cam
#close all opencv windows open
vid_capture.release()
#cv2.destroyAllWindows()

# Ensure all frames are processed
if all_processed_frames != []:  
    frame_height, frame_width, _ = all_processed_frames[0].shape
    #save the processed video frames as an output video file
    output_video = cv2.VideoWriter(FILES_PATH+'output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (frame_width,frame_height))
    for i in range(len(all_processed_frames)):
        output_video.write(all_processed_frames[i])
    output_video.release()
else:
    print("No frames processed")
