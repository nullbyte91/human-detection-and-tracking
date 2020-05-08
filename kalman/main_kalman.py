"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import dlib

import logging as log
import paho.mqtt.client as mqtt

from fysom import *
from imutils.video import FPS
from sklearn.utils.linear_assignment_ import linear_assignment
from collections import deque

from argparse import ArgumentParser
from inference import Network

import utils
import tracker

from multiprocessing import Process, Queue
import multiprocessing
import threading
import queue

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# Browser and OpenCV Window toggle
Browser_ON = False
tracker_list =[] # list for trackers
track_id_list= deque(['1', '2', '3', '4', '5', '6', '7', '7', '8', '9', '10'])
min_hits =1  # no. of consecutive matches needed to establish a track
max_age = 15  # no.of consecutive unmatched detection before 
             # a track is deleted

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

fsm = Fysom({'initial': 'empty',
             'events': [
                 {'name': 'enter', 'src': 'empty', 'dst': 'standing'},
                 {'name': 'exit',  'src': 'standing',   'dst': 'empty'}]})

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = None

    return client

def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.6):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''    
    
    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk) 
        for d,det in enumerate(detections):
         #   det = convert_to_cv2bbox(det)
            IOU_mat[t,d] = utils.box_iou2(trk,det) 
    
    # Produces matches       
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)
    
    matched_idx = linear_assignment(-IOU_mat)        

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []
   
    # For creating trackers we consider any detection with an 
    # overlap less than iou_thrd to signifiy the existence of 
    # an untracked object
    
    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers) 

def image_process_worker(cap, frame_queue, image_queue, in_n, in_c, in_h, in_w):
    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            frame_queue.put(None)
            image_queue.put(None)
            break

        # Pre-process the frame
        image_resize = cv2.resize(frame, (in_w, in_h))
        image = image_resize.transpose((2,0,1))
        image = image.reshape(in_n, in_c, in_h, in_w)
        
        frame_queue.put(frame)
        image_queue.put(image)

def network_inference(infer_network, frame_queue, image_queue, 
                        fw, fh, prob_threshold, fps):
    global tracker_list
    global track_id_list
    global min_hits
    global max_age
    current_inference, next_inference = 0, 1
    while True:
        image = image_queue.get()
        if image is None:
            break
        frame = frame_queue.get()
        # Perform inference on the frame
        infer_network.exec_net_async(image, request_id=current_inference)

        # Get the output of inference
        if infer_network.wait(next_inference) == 0:
            result = infer_network.get_output(next_inference)
            z_box = []
            for box in result[0][0]: # Output shape is 1x1x100x7
                conf = box[2]
                if conf >= prob_threshold:
                    #xmin = int(box[3] * fw)
                    #ymin = int(box[4] * fh)
                    #xmax = int(box[5] * fw)
                    #ymax = int(box[6] * fh)
                    print("box[0] {} box[1] {} box[2] {} box[3] {}".format(box[3], box[4], box[5], box[6]))
                    # Check this logic later if the output is different
                    box_pixel = [int(box[4]*fh), int(box[3]*fw), int(box[6]*fh), int(box[5]*fw)]
                    #box_pixel = [int(box[3]*height), int(box[4]*width), int(box[5]*height), int(box[6]*width)]
                    #box_pixel = [int(box[4]*in_h), int(box[3]*in_w), int(box[5]*in_h), int(box[6]*in_w)]
                    z_box.append(np.array(box_pixel))
        
                x_box =[]
                if len(tracker_list) > 0:
                    for trk in tracker_list:
                        x_box.append(trk.box)
                
                matched, unmatched_dets, unmatched_trks \
                = assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.3) 
                print('Detection: ', z_box)
                print('x_box: ', x_box)
                print('matched:', matched)
                print('unmatched_det:', unmatched_dets)
                print('unmatched_trks:', unmatched_trks)

                # Deal with matched detections     
                if matched.size >0:
                    print('Deal with matched detections')
                    for trk_idx, det_idx in matched:
                        z = z_box[det_idx]
                        z = np.expand_dims(z, axis=0).T
                        tmp_trk= tracker_list[trk_idx]
                        tmp_trk.kalman_filter(z)
                        xx = tmp_trk.x_state.T[0].tolist()
                        xx =[xx[0], xx[2], xx[4], xx[6]]
                        x_box[trk_idx] = xx
                        tmp_trk.box =xx
                        tmp_trk.hits += 1
                
                # Deal with unmatched detections      
                if len(unmatched_dets)>0:
                    print('Deal with unmatched detections')
                    for idx in unmatched_dets:
                        z = z_box[idx]
                        z = np.expand_dims(z, axis=0).T
                        tmp_trk = tracker.Tracker() # Create a new tracker
                        x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
                        tmp_trk.x_state = x
                        tmp_trk.predict_only()
                        xx = tmp_trk.x_state
                        xx = xx.T[0].tolist()
                        xx =[xx[0], xx[2], xx[4], xx[6]]
                        tmp_trk.box = xx
                        tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
                        print(tmp_trk.id)
                        tracker_list.append(tmp_trk)
                        x_box.append(xx)
                
                # Deal with unmatched tracks       
                if len(unmatched_trks)>0:
                    print('Deal with unmatched tracks')
                    for trk_idx in unmatched_trks:
                        tmp_trk = tracker_list[trk_idx]
                        tmp_trk.no_losses += 1
                        tmp_trk.predict_only()
                        xx = tmp_trk.x_state
                        xx = xx.T[0].tolist()
                        xx =[xx[0], xx[2], xx[4], xx[6]]
                        tmp_trk.box =xx
                        x_box[trk_idx] = xx
                
                # The list of tracks to be annotated  
                good_tracker_list =[]
                for trk in tracker_list:
                    if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
                        good_tracker_list.append(trk)
                        x_cv2 = trk.box
                        frame = utils.draw_box_label(trk.id,frame, x_cv2) # Draw the bounding boxes on the 

                # Book keeping
                deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)  

                for trk in deleted_tracks:
                    track_id_list.append(trk.id)
        
                tracker_list = [x for x in tracker_list if x.no_losses<=max_age]

        current_inference, next_inference = next_inference, current_inference

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps.update()

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # global tracker_list
    # global track_id_list
    # global min_hits
    # global max_age

    frame_queue = queue.Queue(maxsize= 4)
    image_queue = queue.Queue(maxsize= 4)
    
    # Initialize the Inference Engine
    infer_network = Network()

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # Load the model through `infer_network`
    infer_network.load_model(args.model, args.device, CPU_EXTENSION, num_requests=2)

    # Get a Input blob shape
    in_n, in_c, in_h, in_w = infer_network.get_input_shape()

    # Get a output blob name
    _ = infer_network.get_output_name()
    
    # Handle the input stream
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    _, frame = cap.read()

    fps = FPS().start()
    _, frame = cap.read()
    fh = frame.shape[0]
    fw = frame.shape[1]
    
    preprocess_thread = None

    preprocess_thread = threading.Thread(target=image_process_worker, 
                    args=(cap, frame_queue, image_queue, in_n, in_c, in_h, in_w))
    
    preprocess_thread.start()

    network_inference(infer_network, frame_queue, image_queue, fw, fh, prob_threshold, fps)

    preprocess_thread.join()
    # Process frames until the video ends, or process is exited
    # while cap.isOpened():
    #     # Read the next frame
    #     flag, frame = cap.read()
    #     if not flag:
    #         break
        
    #     fh = frame.shape[0]
    #     fw = frame.shape[1]
    #     key_pressed = cv2.waitKey(60)

    #     # Pre-process the frame
    #     image_resize = cv2.resize(frame, (in_w, in_h))
    #     image = image_resize.transpose((2,0,1))
    #     image = image.reshape(in_n, in_c, in_h, in_w)
        
    #     # Perform inference on the frame
    #     infer_network.exec_net(image)
    #     # dim = image_resize.shape[0:2]
    #     # height, width = dim[0], dim[1]
    #     # Get the output of inference
    #     if infer_network.wait() == 0:
            
    #         result = infer_network.get_output()
    #         z_box = []
    #         for box in result[0][0]: # Output shape is 1x1x100x7
    #             conf = box[2]
                
    #             if conf >= prob_threshold:
    #                 #xmin = int(box[3] * fw)
    #                 #ymin = int(box[4] * fh)
    #                 #xmax = int(box[5] * fw)
    #                 #ymax = int(box[6] * fh)
    #                 print("box[0] {} box[1] {} box[2] {} box[3] {}".format(box[3], box[4], box[5], box[6]))
    #                 # Check this logic later if the output is different
    #                 box_pixel = [int(box[4]*fh), int(box[3]*fw), int(box[6]*fh), int(box[5]*fw)]
    #                 #box_pixel = [int(box[3]*height), int(box[4]*width), int(box[5]*height), int(box[6]*width)]
    #                 #box_pixel = [int(box[4]*in_h), int(box[3]*in_w), int(box[5]*in_h), int(box[6]*in_w)]
    #                 z_box.append(np.array(box_pixel))

    #         x_box =[]
    #         if len(tracker_list) > 0:
    #             for trk in tracker_list:
    #                 x_box.append(trk.box)
            
    #         matched, unmatched_dets, unmatched_trks \
    #         = assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.3  ) 
    #         print('Detection: ', z_box)
    #         print('x_box: ', x_box)
    #         print('matched:', matched)
    #         print('unmatched_det:', unmatched_dets)
    #         print('unmatched_trks:', unmatched_trks)

    #         # Deal with matched detections     
    #         if matched.size >0:
    #             print('Deal with matched detections')
    #             for trk_idx, det_idx in matched:
    #                 z = z_box[det_idx]
    #                 z = np.expand_dims(z, axis=0).T
    #                 tmp_trk= tracker_list[trk_idx]
    #                 tmp_trk.kalman_filter(z)
    #                 xx = tmp_trk.x_state.T[0].tolist()
    #                 xx =[xx[0], xx[2], xx[4], xx[6]]
    #                 x_box[trk_idx] = xx
    #                 tmp_trk.box =xx
    #                 tmp_trk.hits += 1
            
    #         # Deal with unmatched detections      
    #         if len(unmatched_dets)>0:
    #             print('Deal with unmatched detections')
    #             for idx in unmatched_dets:
    #                 z = z_box[idx]
    #                 z = np.expand_dims(z, axis=0).T
    #                 tmp_trk = tracker.Tracker() # Create a new tracker
    #                 x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
    #                 tmp_trk.x_state = x
    #                 tmp_trk.predict_only()
    #                 xx = tmp_trk.x_state
    #                 xx = xx.T[0].tolist()
    #                 xx =[xx[0], xx[2], xx[4], xx[6]]
    #                 tmp_trk.box = xx
    #                 tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
    #                 print(tmp_trk.id)
    #                 tracker_list.append(tmp_trk)
    #                 x_box.append(xx)
            
    #         # Deal with unmatched tracks       
    #         if len(unmatched_trks)>0:
    #             print('Deal with unmatched tracks')
    #             for trk_idx in unmatched_trks:
    #                 tmp_trk = tracker_list[trk_idx]
    #                 tmp_trk.no_losses += 1
    #                 tmp_trk.predict_only()
    #                 xx = tmp_trk.x_state
    #                 xx = xx.T[0].tolist()
    #                 xx =[xx[0], xx[2], xx[4], xx[6]]
    #                 tmp_trk.box =xx
    #                 x_box[trk_idx] = xx
            
    #         # The list of tracks to be annotated  
    #         good_tracker_list =[]
    #         for trk in tracker_list:
    #             if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
    #                 good_tracker_list.append(trk)
    #                 x_cv2 = trk.box
    #                 frame = utils.draw_box_label(trk.id,frame, x_cv2) # Draw the bounding boxes on the 

    #         # Book keeping
    #         deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)  

    #         for trk in deleted_tracks:
    #             track_id_list.append(trk.id)
    
    #         tracker_list = [x for x in tracker_list if x.no_losses<=max_age]

    #         cv2.imshow("frame",frame)
    #     fps.update()
    
    # Release the out writer, capture, and destroy any OpenCV windows
    cap.release()
    
    if Browser_ON != True:
        cv2.destroyAllWindows()
    
    cv2.destroyAllWindows()
    fps.stop()
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # set log level
    log.basicConfig(filename='example.log',level=log.CRITICAL)
    # Grab command line args
    args = build_argparser().parse_args()
    
    if Browser_ON == True:
        # Connect to the MQTT server
        client = connect_mqtt()
    else:
        # Perform inference on the input stream
        client = None 
    
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
