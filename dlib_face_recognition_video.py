# -*- coding: utf-8 -*-


#importing the required libraries
import pickle
import time
import cv2
import face_recognition

from parameters import FILES_PATH, NUMBER_OF_TIMES_TO_UPSAMPLE, FACE_RECOGNITION_MODEL, FRAME_WIDTH, FRAME_HEIGHT

# Find start time
tick = time.time()

#capture the video from stored video file
upload_folder = FILES_PATH
vid_capture = cv2.VideoCapture(upload_folder + 'vid1.mp4')

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open('dataset/dlib_face_encoding.pkl',"rb").read())

#save the encodings and the corresponding labels in seperate arrays in the same order
known_face_encodings = data["encodings"]
known_face_names = data["names"]

#initialize the array variable to hold all face locations, encodings and names 
all_face_locations = []
all_face_encodings = []
all_face_names = []
all_processed_frames = []

frames_processed = 0

#loop through every frame in the video
while True:
    #get the current frame from the video stream as an image
    ret,current_frame = vid_capture.read()

    if ret:
        frames_processed += 1

        #resize the current frame to 1/4 size to proces faster
        #current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)

        #resize frame to FRAME_WIDTHxFRAME_HEIGHT to display the video if frame is too big
        if current_frame.shape[1] > FRAME_WIDTH or current_frame.shape[0] > FRAME_HEIGHT:
            current_frame_small = cv2.resize(current_frame, (FRAME_WIDTH, FRAME_HEIGHT))
        else:
            current_frame_small = current_frame
        #detect all faces in the image
        #arguments are image,no_of_times_to_upsample, model
        all_face_locations = face_recognition.face_locations(current_frame_small,
                                                             number_of_times_to_upsample=NUMBER_OF_TIMES_TO_UPSAMPLE,
                                                             model= FACE_RECOGNITION_MODEL)
        
        print(all_face_locations)

        #detect face encodings for all the faces detected
        all_face_encodings = face_recognition.face_encodings(current_frame_small,all_face_locations)


        #looping through the face locations and the face embeddings
        for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
            #splitting the tuple to get the four position values of current face
            top_pos,right_pos,bottom_pos,left_pos = current_face_location
            
            #change the position maginitude to fit the actual size video frame
            '''top_pos = top_pos*4
            right_pos = right_pos*4
            bottom_pos = bottom_pos*4
            left_pos = left_pos*4'''
            
            #find all the matches and get the list of matches
            all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding, tolerance=0.40)
        
            #string to hold the label
            name_of_person = 'Unknown face'
            
            #check if the all_matches have at least one item
            #if yes, get the index number of face that is located in the first index of all_matches
            #get the name corresponding to the index number and save it in name_of_person
            if True in all_matches:
                first_match_index = all_matches.index(True)
                name_of_person = known_face_names[first_match_index]
            
            if name_of_person == 'Unknown face':
                color = (0,0,255) #Red
            else:
                color = (0,255,0) #Green

            #draw rectangle around the face    
            cv2.rectangle(current_frame_small,(left_pos,top_pos),(right_pos,bottom_pos),color, 2)
            
            #display the name as text in the image
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(current_frame_small, name_of_person, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
        
        #display the video
        #cv2.imshow("Webcam Video",current_frame)

        #add current processed frame to the all_processed_frames array
        all_processed_frames.append(current_frame_small)
        
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
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

frame_height, frame_width, _ = all_processed_frames[0].shape
#save the processed video frames as an output video file
output_video = cv2.VideoWriter(FILES_PATH+'output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (frame_width,frame_height))
for i in range(len(all_processed_frames)):
    output_video.write(all_processed_frames[i])
output_video.release()
