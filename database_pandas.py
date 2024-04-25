''' This module contain code to store inferred faces in a pandas dataframe

    The dataframe will have the following columns:
    1. name/id
    2. Time
    3. Date
    4. Cam Name
    5. Cam IP
    6. Face Distance - the distance between the face and the known face (which can be considered
    as inverse of confidence)

    The maximum number of rows in the dataframe will be equal to 1 million

    The dataframe will be stored in a csv file

'''

import time
import pandas as pd

from parameters import REPORT_PATH
from custom_logging import logger
from util.generic_utilities import check_for_directory

df_inferred_faces = pd.DataFrame(columns=['name/id','time','date','cam name', 'cam_ip','face_distance'])


def store_inferred_face_in_dataframe(name_of_person, face_distance, cam_name = 'unknown', cam_ip = 'unknown'):
    '''
        This function will store the name of the person and the face distance in the dataframe

        Arguments:
            name_of_person {string} -- name of the person
            face_distance {float} -- face distance
            cam_name {string} -- name of the camera
            cam_ip {string} -- ip address of the camera
        
        Returns:
            None
    '''

    #current time in HH:MM:SS format
    current_time = time.strftime("%H:%M:%S")
    #current date in DD/MM/YYYY format
    current_date = time.strftime("%d/%m/%Y")

    # We want to store the name of the person only if it is not already present in the dataframe
    if name_of_person not in df_inferred_faces['name/id'].values:        
        #store the name of the person in the dataframe
        df_inferred_faces.loc[len(df_inferred_faces)] = [name_of_person, current_time, current_date, cam_name, cam_ip, face_distance]

    #if the number of rows in the dataframe is greater than 1 million, then save the dataframe in a csv file
    #if len(df_inferred_faces) > 1000000:
    #    df_inferred_faces.to_csv('inferred_faces.csv')
    #    df_inferred_faces = pd.DataFrame(columns=['name/id','time','date','location','face_distance'])


def store_dataframe_in_csv():
    '''
        This function will store the dataframe in a csv file

        Arguments:
            None
        
        Returns:
            True if the dataframe is stored in the csv file, False otherwise
    '''
    global df_inferred_faces

    try:

        # remove the last element from the string as it is the file name
        report_path_dir = REPORT_PATH.rsplit('/', maxsplit=1)[0]
        # create a directory for logs if it does not exist
        check_for_directory(report_path_dir)

        df_inferred_faces.to_csv(REPORT_PATH, encoding='utf-8')
        logger.info('Dataframe stored in csv file')
        return True
    except Exception as e:
        logger.error(f'Exception occured while storing dataframe in csv file: {e}')
        return False
    finally:
        #reset the dataframe
        #df_inferred_faces = pd.DataFrame(columns=['name/id','time','date','location','face_distance'])
        pass

def purge_dataframe():
    '''
        This function will purge the dataframe

        Arguments:
            None
        
        Returns:
            None
    '''
    global df_inferred_faces

    #delete the dataframe
    del df_inferred_faces
    
    #reset the dataframe
    df_inferred_faces = pd.DataFrame(columns=['name/id','time','date','cam name', 'cam_ip','face_distance'])
