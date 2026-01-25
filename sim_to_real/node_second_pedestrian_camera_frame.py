#!/usr/bin/env python

import pyzed.sl as sl
import numpy as np
from float_array_timestamp import FloatArrayTimestamp
import cv2 
import time
from math import hypot
from datetime import datetime
import rospy
import heapq
import psutil
import os

class Pedestrian:
    def __init__(self, visualization=True):
        self.visualization = visualization
        
        self.pedestrian_num_max = 7
        self.pedestrian_dim = 4 # id, x, y, moving
        
        self.pedestrian_offset = np.array([0.1, 0.06], dtype=np.float32)
        self.current_time = time.time()
        self.detection_period = 0.1
        
        self.zed = sl.Camera()
        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.coordinate_units = sl.UNIT.METER        
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Camera Open : " + repr(err) + ". Exit program.")
            exit()
        
        self._init_zed_modules() # positional tracking, body tracking
        
        self.runtime_parameters = sl.RuntimeParameters()
        self.runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA # WORLD or CAMERA
        # Configure object detection runtime parameters
        self.obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
        self.obj_runtime_param.detection_confidence_threshold = 60
        self.obj_runtime_param.object_class_filter = [sl.OBJECT_CLASS.PERSON]    # Only detect Persons

        self.camera_pose = sl.Pose()
        self.objects = sl.Objects()
        self.image = sl.Mat()
        
        self.py_translation = sl.Translation()
        
        self.pub_pedestrian = rospy.Publisher('/pedestrians', FloatArrayTimestamp, queue_size=10)
        
    def _init_zed_modules(self):
        # At first, enable positional tracking before body tracking
        # Set positional tracking parameters
        py_transform = sl.Transform()  # First create a Transform object for TrackingParameters object
        tracking_params = sl.PositionalTrackingParameters(_init_pos=py_transform) #set parameters for Positional Tracking
        tracking_params.enable_imu_fusion = True
        # tracking_params.mode = sl.POSITIONAL_TRACKING_MODE.GEN_1
        zed_error = self.zed.enable_positional_tracking(tracking_params)
        if zed_error != sl.ERROR_CODE.SUCCESS:
            print("enable_positional_tracking: " + repr(zed_error) + ". Exit program.")
            self.zed.close()
            exit(-1)
        
        # Enable object detection module
        obj_param = sl.ObjectDetectionParameters()
        # Defines if the object detection will track objects across images flow.
        obj_param.enable_tracking = True       # if True, enable positional tracking
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM
        zed_error = self.zed.enable_object_detection(obj_param)
        if zed_error != sl.ERROR_CODE.SUCCESS:
            print("enable_object_detection: " + repr(zed_error) + ". Exit program.")
            self.zed.close()
            exit(-1)
            
        # Define video parameters
        frame_width = 1280  # Set width of the frames
        frame_height = 720  # Set height of the frames
        fps = 1.0 / self.detection_period  # Frames per second
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
        output_file = './evaluation_episodes/zed_video_' + str(timestamp) + '.mp4'
        codec = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for AVI format

        # Initialize VideoWriter
        # self.out_video = cv2.VideoWriter(output_file, codec, fps, (frame_width, frame_height))

    def pedestrian_thread(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        while not rospy.is_shutdown():
            t_start = time.time()
            if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
               
                # Retrieve left image
                self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
                # Convert to OpenCV Mat
                frame = sl.Mat.get_data(self.image)[:, :, :-1]
                frame_for_show = frame.copy()
            
                # Retrieve objects
                self.zed.retrieve_objects(self.objects, self.obj_runtime_param)
                all_pedestrians = [] 

                if self.objects.is_new:
                    
                    for object_i in self.objects.object_list:
                        bounding_box_2d = object_i.bounding_box_2d
                        # Draw each box as a polygon
                        points = np.array(bounding_box_2d, np.int32)
                        # Convert bounding box to rectangle
                        x_coords = [point[0] for point in points]
                        y_coords = [point[1] for point in points]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        
                        pedestrian_x = round(object_i.position[2] + self.pedestrian_offset[0], 3)
                        pedestrian_y = round(-object_i.position[0] + self.pedestrian_offset[1], 3)
                        
                        moving = object_i.action_state == sl.OBJECT_ACTION_STATE.MOVING

                        if moving:
                            text = str(object_i.id) + ": moving"
                        else:
                            text = str(object_i.id) + ": static"
                        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                        # Position the text (center it within the box)
                        text_x = x_min + (x_max - x_min - text_size[0]) // 2
                        text_y = y_min + (y_max - y_min + text_size[1]) // 2
                        cv2.putText(frame_for_show, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)
                        cv2.rectangle(frame_for_show, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        all_pedestrians.append(np.array([object_i.id, pedestrian_x, pedestrian_y, moving], dtype=np.float32))  
                
                used_pedestrians = []
                if len(all_pedestrians) > self.pedestrian_num_max:
                    dis_all = []
                    for ped in all_pedestrians:
                        dis = hypot(ped[1], ped[2])
                        dis_all.append(dis)
                    # Find the self.pedestrian_num_max smallest values and their indexes
                    smallest = heapq.nsmallest(self.pedestrian_num_max, enumerate(dis_all), key=lambda x: x[1])
                    # Separate indexes and values
                    indexes = [i for i, v in smallest]
                    used_pedestrians= [all_pedestrians[j] for j in indexes]
                else:
                    used_pedestrians = all_pedestrians
                        
                print('pedestrian: id, x, y, moving')   
                pedestrians_for_pub = -100.0 * np.ones((self.pedestrian_num_max, self.pedestrian_dim), dtype=np.float32)
                for index, item in enumerate(used_pedestrians):
                    pedestrians_for_pub[index] = item
                    print(item)
                    
                pedestrians_for_pub = pedestrians_for_pub.flatten()
                pedestrians_for_pub = pedestrians_for_pub.tolist()
                pedestrians_msg = FloatArrayTimestamp()
                pedestrians_msg.header.stamp = rospy.Time.now()
                pedestrians_msg.header.frame_id = "camera_init"  
                pedestrians_msg.data = pedestrians_for_pub
                self.pub_pedestrian.publish(pedestrians_msg)
                
                print('*********************************************************')
                # if self.visualization:
                #     cv2.imshow("Image with Bounding Boxes", frame_for_show)
                #     cv2.waitKey(1)
                # else:
                #     self.out_video.write(frame_for_show)
            t_end = time.time()
            print('total time to pedestrian detection: ', round(time.time() - t_start, 4))
            if t_end - t_start < self.detection_period:
                time.sleep(self.detection_period - (t_end - t_start)) 
                
    def close_zed(self):
        # self.out_video.release()
        cv2.destroyAllWindows()
        self.image.free(memory_type=sl.MEM.CPU)
        self.zed.disable_object_detection()
        self.zed.disable_positional_tracking()
        self.zed.close()
        
def main(): 
    p = psutil.Process(os.getpid())
    p.cpu_affinity([6, 5])
    print("Using CPU cores:", p.cpu_affinity())

    print('start pedestrian detection node')
    rospy.init_node("pedestrian", anonymous=True)
    pedestrian_node = Pedestrian()
    try:
        pedestrian_node.pedestrian_thread()
    except KeyboardInterrupt:
        pass
    finally:
        pedestrian_node.close_zed()
        rospy.signal_shutdown("Condition met, shutting down.")

if __name__ == "__main__":
    main()