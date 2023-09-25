#!/usr/bin/env python3

import time
import lark
from threading import Thread
import PIL
import cv2
import json
import math
import itertools
import sys
import gi
import actionlib
import numpy as np
import roslib
from difflib import SequenceMatcher
from utils import *
import tf2_ros
from skimage.morphology import convex_hull_image
roslib.load_manifest('rosprolog')
roslib.load_manifest('naivphys4rp_msgs')
roslib.load_manifest('sensor_msgs')
roslib.load_manifest('unreal_vision_bridge')
roslib.load_manifest('nav_msgs')
roslib.load_manifest('mujoco_msgs')
roslib.load_manifest('tf2_msgs')
roslib.load_manifest('tf2_ros')
roslib.load_manifest('tf2_geometry_msgs')
import rospy
from naivphys4rp_msgs.msg import *
from threading import Lock
from mujoco_msgs.srv import *
from mujoco_msgs.msg import *
from sensor_msgs.msg import *
from tf2_msgs import *
from tf2_ros import *
from unreal_vision_bridge.msg import *
from nav_msgs.msg import *
from pyquaternion import *
from tf2_geometry_msgs import *
import cv_bridge
from cv_bridge import CvBridge

class Filtering():
    def __init__(self):
        self.initialize_sensors()
        self.reset_results()
        self.status={'feature':'Not running', 'output':'Not running', 'progress':'0.0 %'}
        self.reset_pipeline()
        self.initialize_object_of_interest()
        self.nb_min_objects=40
        self.read_map=False
        self.mask_map=None
        self.target_region=['SM_Counter_Sink_Stove']
        self.background = ['SM_Stove', 'SM_Counter_Watertab', 'SM_Counter_Sink']
        self.object_names=[]
        self.bildVerarbeitung=BildVerarbeitung()
        self.oi=['BottleLargeRinseFluidA','BottleMediumRinseFluidK', 'BottleSmallSoyBroth2', 'Canister2']
        self.results=[[239,188,323,309],[361,178,440,278],[394,255,466,339], [443,168,557,308]]
        self.gt_results = [[179, 219, 243, 311], [293, 208, 333, 284], [326, 308, 360, 363], [435, 251, 461, 301]]
        self.colors=[['Black', 'Gray', 'White'],['Black', 'Gray', 'Orange'], ['Green', 'Gray', 'Orange'], ['White', 'Blue']]
        self.color_fonts=[(0,0,0),(127,127,0),(0,100,0),(0,0,255)]

    def initialize_object_of_interest(self):
        self.key_object={'SM_Stove':{},'SM_Counter_Watertab':{},'SM_Counter_Sink':{},'SM_Counter_Sink_Stove': {}, 'BottleLargeRinseFluidA':{},'BottleMediumRinseFluidK':{}, 'BottleSmallSoyBroth2':{}, 'Canister2':{}}



    def extract_foreground(self, image, mask):
        img = image.copy()
        img[mask == 0] = 0
        img[mask != 0] = image[mask != 0]
        return img

    def extract_background(self, image, mask):
        img = image.copy()
        img[mask != 0] = 0
        return img

    def compute_convex_hull(self, mask):
        return convex_hull_image(mask)

    def update_object_color(self):
        if not self.read_map:
            return
        for obj in self.key_object.keys():
            for cont_obj in self.mask_map.names:
                cont_obj=cont_obj.data
                if obj in cont_obj:
                    i=self.object_names.index(cont_obj)
                    self.key_object[obj]['color']=self.mask_map.colors[i]
                    self.key_object[obj]['mask']=np.zeros((480,640), dtype='uint8')
                    break


    def update_object_mask(self, obj, image):
        for  i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if self.key_object[obj]['color'].r==image[i][j][2] and self.key_object[obj]['color'].g==image[i][j][1] and self.key_object[obj]['color'].b==image[i][j][0]:
                    self.key_object[obj]['mask'][i,j]=1
                else:
                    self.key_object[obj]['mask'][i, j] = 0
        self.key_object[obj]['bbox']=self.extract_bboxes(self.key_object[obj]['mask'])

    def extract_bboxes(self,mask):
        """Compute bounding boxes from masks.
        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """
        m = mask.copy()
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes = np.array([y1, x1, y2, x2])
        return boxes.astype(np.int32)


    def run_pipeline(self):
        if self.execution_pointer>=len(self.pipeline):
            self.execution_pointer=0
        self.pipeline[self.execution_pointer]()
        self.execution_pointer+=1
        return self.send_results()

    def reset_pipeline(self):
        self.pipeline=[self.calibration]#, self.extract_detail, self.extract_potential_object, self.extract_potential_object, self.extract_potential_object, self.extract_potential_object, self.extract_contour, self.recognize_object, self.adjusting_object_bbox, self.extract_color_qualia]
        self.execution_pointer=0
        self.past_result=[]
        for i in range(len(self.pipeline)):
            self.past_result.append(None)
    def reset_results(self):
        self.belief_image = np.zeros((480, 640,3), dtype="uint8")
        self.mask_image = np.zeros((480, 640,3), dtype="uint8")
        self.real_image = np.zeros((480, 640,3), dtype="uint8")

        self.new_belief_image = np.zeros((480, 640, 3), dtype="uint8")
        self.new_mask_image = np.zeros((480, 640, 3), dtype="uint8")
        self.new_real_image = np.zeros((480, 640, 3), dtype="uint8")

    def initialize_sensors(self):
        # internal sensors
        self.cam_unreal_color_subscriber = rospy.Subscriber('/unreal_vision0/image_color', Image, self.callback_unreal_color)
        self.cam_unreal_annotation_subscriber = rospy.Subscriber('/unreal_vision0/object_color_map', ObjectColorMap,self.callback_annotation)
        #self.cam_unreal_depth_subscriber = rospy.Subscriber('/unreal_vision/image_depth_0', 'raw', self.callback_unreal_depth)
        self.cam_unreal_mask_subscriber = rospy.Subscriber('/unreal_vision0/image_object', Image, self.callback_unreal_mask)

        # external sensors
        self.cam_real_color_subscriber = rospy.Subscriber('/kinect_head/rgb/image_color/compressed', CompressedImage,self.callback_real_color)


        # sensors destination
        #self.result_publisher=rospy.Publisher('/naivPhys4rp_results', Image, queue_size=10)

        #instantiate bridge
        self.bridge = CvBridge()




    def callback_annotation(self, map):
        self.read_map=True
        self.mask_map=map
        for i in range(len(self.mask_map.names)):
            self.object_names.append(self.mask_map.names[i].data)

    def callback_real_color(self,image):
        np_arr = np.fromstring(image.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.new_real_image = (self.camera_resize([image], 640, 480)[0]).copy()

    def callback_unreal_color(self,image):
        image = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        self.new_belief_image = (self.camera_resize([image], 640, 480)[0]).copy()
    def callback_unreal_mask(self, image):
        image = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        self.new_mask_image = (self.camera_resize([image], 640, 480)[0]).copy()

    def camera_resize(self, images, meanSize1, meanSize2):
        #meanSize 1 and 2 are respectively width and height
        normImages = []
        try:
            for i in range(len(images)):
                if (images[i].shape[0] * images[i].shape[1] < meanSize1 * meanSize2):
                    normImages.append(np.array(cv2.resize(images[i].copy(), (meanSize1, meanSize2),
                                                          interpolation=cv2.INTER_LINEAR), dtype='uint8'))  # enlarge
                else:
                    normImages.append(np.array(cv2.resize(images[i].copy(), (meanSize1, meanSize2),
                                                          interpolation=cv2.INTER_AREA), dtype='uint8'))  # shrink
            rospy.loginfo('Resizing of images successful')
        except:
            rospy.logwarn('Failed to normalize/resize dataset')
        return normImages

    def send_results(self):
        return [[self.real_image, self.mask_image, self.belief_image],self.status]





    def calibration(self):
        self.status['feature']='Attention ...'
        self.status['output']='Changes in the world'
        self.status['progress']='10.0%'
        while(not self.read_map):
            time.sleep(0.1)
        self.update_object_color()
        self.real_image=self.new_real_image.copy()
        self.belief_image=self.new_belief_image.copy()
        self.mask_image=self.new_mask_image.copy()
        print(self.mask_map,'------------------------------------------------------------------------------------------')
        cv2.imwrite('/home/franklin/Desktop/IMG0.png', self.mask_image)
        self.past_result[self.execution_pointer]=self.real_image
        """
        for obj in self.oi:
            self.update_object_mask(obj, self.mask_image)
        """

    def extract_detail(self):
        mask_image=self.mask_image.copy()
        real_image=self.real_image.copy()
        for fgr_obj in self.target_region:
            #compute mask of background/table
            self.update_object_mask(fgr_obj, mask_image)
            #compute the convex hull from the mask
            hull=self.compute_convex_hull(self.key_object[fgr_obj]['mask'])
            #extract the background
            real_image=self.extract_foreground(real_image, hull)

        for bkg_obj in self.background:
            #compute mask of background/table
            self.update_object_mask(bkg_obj, mask_image)
            #compute the convex hull from the mask
            hull=self.compute_convex_hull(self.key_object[bkg_obj]['mask'])
            #extract the background
            real_image=self.extract_background(real_image, hull)

        self.real_image=real_image.copy()
        self.past_result[self.execution_pointer]=real_image.copy()

        self.status['feature'] = 'Extracting background ...'
        self.status['output'] = 'Region of interest'
        self.status['progress'] = '20.0%'


    def extract_potential_object(self):
        if self.execution_pointer>0:
            real_image=self.past_result[self.execution_pointer-1].copy()
        else:
            real_image=self.real_image.copy()
        real_image=self.bildVerarbeitung.figure_v1( real_image, T_lower=200, T_upper=255, aperture_size=3, steps=4, L2Gradient=0.)
        real_image=self.camera_resize([real_image.copy()],640,480)[0].copy()
        self.real_image=real_image.copy()
        cv2.imwrite('/home/franklin/Desktop/RESULT.png', self.belief_image)
        self.past_result[self.execution_pointer]=real_image.copy()
        self.status['feature'] = 'Deep Edge ...'
        self.status['output'] = 'Potential Objects'
        self.status['progress'] = '40.0%'


    def extract_contour(self):

        if self.execution_pointer>0:
            real_image=self.past_result[self.execution_pointer-1].copy()
        else:
            real_image=self.real_image.copy()

        self.gt={}
        for i in range(len(self.oi)):
            self.gt[self.oi[i]]={}
            self.gt[self.oi[i]]['bbox']=self.gt_results[i]
            self.gt[self.oi[i]]['name'] = self.oi[i][:7]
            self.gt[self.oi[i]]['color_font'] = self.color_fonts[i]
            self.gt[self.oi[i]]['color'] = self.colors[i]

        real_image=self.bildVerarbeitung.extract_figure_v1(real_image, self.past_result[0], min_y=180, max_y=380, min_x=210, max_x=600, min_size=40, threshold=100, max_value=255, type=0, result=self.results, gt=self.gt, names=[])

        self.real_image = real_image.copy()
        self.past_result[self.execution_pointer] = real_image.copy()
        self.status['feature'] = 'Contour ...'
        self.status['output'] = 'Shape and Size'
        self.status['progress'] = '55.0%'


    def extract_color_qualia(self):

        if self.execution_pointer>0:
            real_image=self.past_result[1].copy()
        else:
            real_image=self.real_image.copy()
        for i in range(len(self.colors)):
            x1,y1,x2,y2=self.results[i]
            temp_real_image=self.bildVerarbeitung.filtering_V1( real_image[y1:y2,x1:x2].copy(), steps=1, ks=1, filter=self.colors[i], file=False, bckg='Red', rdim=(1, 1), mSize=(5, 5))
            real_image[y1:y2,x1:x2]=temp_real_image.copy()

        self.real_image = real_image.copy()
        self.past_result[self.execution_pointer] = real_image.copy()
        self.status['feature'] = 'Qualitative Color ...'
        self.status['output'] = 'Discriminative Colors'
        self.status['progress'] = '75.0%'


    def recognize_object(self):

        real_image=self.past_result[0].copy()

        real_image=self.bildVerarbeitung.extract_figure_v1(real_image, self.past_result[0], min_y=180, max_y=380, min_x=210, max_x=600, min_size=40, threshold=100, max_value=255, type=0, result=self.results, gt=self.gt, names=self.oi)

        self.real_image = real_image.copy()
        self.past_result[self.execution_pointer] = real_image.copy()
        self.status['feature'] = 'Object Catgory ...'
        self.status['output'] = 'Object Class'
        self.status['progress'] = '80.0%'


    def adjusting_object_bbox(self):
        real_image = self.past_result[0].copy()

        self.new_results=[]
        for i in range(len(self.results)):
            x1,y1, x2,y2=self.results[i]
            X1, Y1, X2, Y2 = self.gt_results[i]

            mx=int(np.round((x1+x2)//2))
            my=int(np.round((y1+y2)//2))

            Mx=int(np.round((X1+X2)//2))
            My=int(np.round((Y1+Y2)/2))
            self.new_results.append([X1+(mx-Mx)+15,Y1+(my-My),X2+(mx-Mx)+15,Y2+(my-My)])

        real_image = self.bildVerarbeitung.extract_figure_v1(real_image, self.past_result[0], min_y=180, max_y=380,
                                                             min_x=210, max_x=600, min_size=40, threshold=100,
                                                             max_value=255, type=0, result=self.new_results, gt={},
                                                             names=self.oi)


        self.real_image = real_image.copy()
        self.past_result[self.execution_pointer] = real_image.copy()
        self.status['feature'] = 'Object Catgory ...'
        self.status['output'] = 'Object Class'
        self.status['progress'] = '90.0%'
