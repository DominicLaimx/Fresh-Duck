import os
import time
from pathlib import Path

import cv2
import csv
# import easyocr
import pandas as pd
import numpy as np
import tensorflow as tf
from peekingduck.pipeline.nodes.model import yolo
from peekingduck.pipeline.nodes.dabble import bbox_to_btm_midpoint
from peekingduck.pipeline.nodes.dabble import zone_count
# from matplotlib import pyplot as plt

def get_batch_time(zone):
  '''
  left: 0 centre: 1 right: 2
  prints out how many days, hours and minutes since the batch was first detected
  '''
  if zone==0:
    if data["start_time_left_zone"]>0:
      batchtime = time.time()-data["start_time_left_zone"]
      print("Left zone: " + str(batchtime//86400) + " Days " + str(batchtime//3600) + "hours " + str(batchtime//60) + "minutes" )
      return batchtime
    else:
      return 0
  if zone==1:
    if data["start_time_centre_zone"]>0:
      batchtime = time.time()-data["start_time_centre_zone"]
      print("Centre zone: " + str(batchtime//86400) + " Days " + str(batchtime//3600) + "hours " + str(batchtime//60) + "minutes" )
      return batchtime
    else:
      return 0
  if zone==2:
    if data["start_time_right_zone"]>0:
      batchtime = time.time()-data["start_time_right_zone"]
      print("Right zone: " + str(batchtime//86400) + " Days " + str(batchtime//3600) + "hours " + str(batchtime//60) + "minutes" )
      return batchtime
    else:
      return 0
  

def updatedata(zone_count,current_time):
  '''
  Updates data dictionary 
  if zone_count goes from 0 to >0 , new batch received, record time
  if zone_count goes from >0 to 0 , batch is removed, erase time
  '''
  if(data["prev_left_zone_count"]==0 and zone_count[0]>0):
    data["start_time_left_zone"] = current_time
  elif (data["prev_left_zone_count"]>0 and zone_count[0]==0):
    data["start_time_left_zone"] = 0
  data["prev_left_zone_count"] = zone_count[0]

  if(data["prev_centre_zone_count"]==0 and zone_count[1]>0):
    data["start_time_centre_zone"] = current_time
  elif (data["prev_centre_zone_count"]>0 and zone_count[1]==0):
    data["start_time_centre_zone"] = 0
  data["prev_centre_zone_count"] = zone_count[1]

  if(data["prev_right_zone_count"]==0 and zone_count[2]>0):
    data["start_time_right_zone"] = current_time
  elif (data["prev_right_zone_count"]>0 and zone_count[2]==0):
    data["start_time_right_zone"] = 0
  data["prev_right_zone_count"] = zone_count[2]

def take_frame():
    '''
    Returns one frame from webcam
    '''
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    # cv2.imwrite('webcamphoto.jpg', frame)

    '''
    cv2.imwrite('webcamphoto.jpg', frame) 
    if need to output the image ^^
    to change data type of the image file - change .jpg to .png etc
    ''' 
    cap.release()
    return frame

def export_csv(outputfile,x, y, z):
    #x, y, z is the time
    # add headers + replace rows
    f = open(outputfile, "w", newline = "")
    header = ["Zone","Time"]
    left = ["Left", x]
    centre = ["Centre", y]
    right = ["Right", z]
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerow(left)
    writer.writerow(centre)
    writer.writerow(right)
    f.close()
def update_csv(inputfile):
  df = pd.read_csv(inputfile)
  df_time = pd.read_csv("ExpiryDate.csv")# The CV output file

# merge data
  df = pd.merge(df,df_time,on=["Zone"])

# changing time format
  df["Days"] = df.Time//86400
  df["Hours"] = ((df.Time) - (df.Days*86400))//(3600)
  df["Minutes"] = ((df.Time) - (df.Days*86400)-(df.Hours*3600))//(60)
  df.drop(columns = "Time")

# status of item in each zone
  conditions = [
    (df.Days < 0.8 * df.Fresh),
    (df.Days >= 0.8 * df.Fresh) & (df.Days < df.Fresh),
    (df.Days >= df.Fresh)
]

  values = ["Fresh","Expiring","Expired"]
  df["Status"] = np.select(conditions,values)
  df.drop(columns=["Fresh","Time"],inplace=True)

# output to csv
  df.to_csv('output_1.csv',index=False)

yolo_model_node = yolo.Node(detect = ["apple","orange","banana","broccoli","carrot"]) #input: image data output: bboxdata + bboxlabels + bbox score data
dabble_btm_midpoint_node = bbox_to_btm_midpoint.Node() #input: image data + bbox data output: btm midpoint data
dabble_zone_count_node = zone_count.Node(
    zones= [[[0, 0], [0.33, 0], [0.33, 1], [0, 1]],
            [[0.33, 0], [0.66, 0], [0.66, 1], [0.33, 1]],
            [[0.66, 0], [1, 0], [1, 1], [0.66, 1]]],
    resolution = [640,480]
    )

data = {"start_time_left_zone": 0, "prev_left_zone_count": 0,"start_time_centre_zone": 0, "prev_centre_zone_count": 0, 
        "start_time_right_zone": 0, "prev_right_zone_count": 0}
        
prev_time = 0
while(True):
  current_time = time.time()
  if(current_time-prev_time)>30: #run updates every X seconds
    path = r'D:/PeekingDuck/Images/AppleBanana.png'
    outputfile="output.csv"
    image = cv2.imread(path)
    # image = take_frame()

    yolo_input = {"img": image}
    yolo_output = yolo_model_node.run(yolo_input)

    bbox_midpoint_input = {
      "img": image,
      "bboxes": yolo_output["bboxes"]
    }
    bbox_midpoint_output = dabble_btm_midpoint_node.run(bbox_midpoint_input)

    zone_count_input = {"btm_midpoint": bbox_midpoint_output["btm_midpoint"]}
    zone_count_output = dabble_zone_count_node.run(zone_count_input)

    print(zone_count_output["zone_count"])
    prev_time=current_time
    updatedata(zone_count_output["zone_count"],current_time)
    print(data)
    export_csv(outputfile,get_batch_time(0),get_batch_time(1),get_batch_time(2))

    update_csv(outputfile)



    
