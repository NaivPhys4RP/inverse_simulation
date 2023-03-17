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
roslib.load_manifest('rosprolog')
roslib.load_manifest('naivphys4rp_msgs')
roslib.load_manifest('sensor_msgs')
roslib.load_manifest('unreal_vision_bridge')
roslib.load_manifest('nav_msgs')
roslib.load_manifest('mujoco_msgs')
import rospy
from naivphys4rp_msgs.msg import *
from mujoco_msgs.srv import *
from mujoco_msgs.msg import *
from sensor_msgs.msg import *
from unreal_vision_bridge.msg import *
from nav_msgs.msg import *
from pyquaternion import *
import cv_bridge
from cv_bridge import CvBridge

# A class that translate symbols into sensation
class Imagination():

    def __init__(self):
        #create all subscribers and publishers

        # set the robot view: Third or First
        self.robot_view=0

        print ("Creating controllers ...")

        self.robot_base_joint_name="base_footprint"
        #base motor
        self.robot_base_publisher=rospy.Publisher('/set_base_footprint', Odometry, queue_size=10)
        #self.robot_base_subscriber=rospy.Subscriber('/get_base_footprint', Odometry, self.callback)

        #joint motor
        self.robot_joint_publisher = rospy.Publisher('/set_joint_states', JointState, queue_size=10)
        #self.robot_joint_subscriber = rospy.Subscriber('/get_joint_states', JointState, self.callback)

        #sensor
        #self.robot_joint_subscriber = rospy.Subscriber('/unreal_vision/camera_info', CameraInfo, self.callback)
        #self.robot_joint_subscriber = rospy.Subscriber('/unreal_vision/image_color', Image, self.callback)
        #self.robot_joint_subscriber = rospy.Subscriber('/unreal_vision/image_depth', Image, self.callback)
        #self.robot_joint_subscriber = rospy.Subscriber('/unreal_vision/image_object', Image, self.callback)
        #self.robot_joint_subscriber = rospy.Subscriber('/unreal_vision/object_color_map', ObjectColorMap, self.callback)

        #object
        self.object_state_publisher = rospy.Publisher('/set_object_states', ObjectState, queue_size=10)
        #self.object_state_subscriber = rospy.Subscriber('/get_object_states', ObjectState, self.callback)

        print("Wating for controller servers ...")
        rospy.wait_for_service('/unreal/spawn_objects')
        self.object_assertion = rospy.ServiceProxy('/unreal/spawn_objects', SpawnObject)
        #cannot wait for this service (not yet implemented)
        #self.object_deletion = rospy.ServiceProxy('/unreal/destroy_objects', mujoco_msgs/DestroyObject)
        self.initialize_config_variables()
        self.initialize_robot_pose()
        self.initialize_robot_arms()
        self.initialize_robot_head()
        self.initialize_robot_grippers()
        self.initialize_robot_camera()

        print("Creation of controllers successfully terminated ...")

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

    def initialize_robot_camera(self):
        self.iheight = rospy.get_param('input_height', 480)
        self.iwidth = rospy.get_param('input_width', 640)
        self.height = rospy.get_param('output_height', 1000)
        self.width = rospy.get_param('output_width', 1000)
        self.color_hint = rospy.get_param('color_hint', "raw")

        # First View
        self.topic_color1 = rospy.get_param('topic_color_name1', "/unreal_vision1/image_color")
        self.topic_object1 = rospy.get_param('topic_object_name1', "/unreal_vision1/image_object")
        self.topic_depth1 = rospy.get_param('topic_depth_name1', "/unreal_vision1/image_depth")
        self.object_map1 = rospy.get_param('topic_object_map_name1', "/unreal_vision1/object_color_map")

        self.topic_color = rospy.get_param('topic_color_name1', "/unreal_vision/image_color")
        self.topic_object = rospy.get_param('topic_object_name1', "/unreal_vision/image_object")
        self.topic_depth = rospy.get_param('topic_depth_name1', "/unreal_vision/image_depth")
        self.object_map = rospy.get_param('topic_object_map_name1', "/unreal_vision/object_color_map")



        self.model = None
        self.color_hints = {"": Image, "Compressed": CompressedImage, "raw": Image, "Raw": Image,
                            "compressed": CompressedImage}
        if rospy.get_param('videomode', 'local') == 'local':
            self.cvMode = 'bgr8'
        else:
            self.cvMode = 'rgb8'
        self.bridge = CvBridge()
        self.color_image=np.zeros((self.iheight, self.iwidth,3), dtype="uint8")
        self.object_image = np.zeros((self.iheight, self.iwidth, 3), dtype="uint8")
        self.depth_image = np.zeros((self.iheight, self.iwidth, 3), dtype="uint8")

        self.color_image1 = np.zeros((self.iheight, self.iwidth, 3), dtype="uint8")
        self.object_image1 = np.zeros((self.iheight, self.iwidth, 3), dtype="uint8")
        self.depth_image1 = np.zeros((self.iheight, self.iwidth, 3), dtype="uint8")

        self.color_image_subscriber1 = rospy.Subscriber(self.topic_color1, self.color_hints[self.color_hint],
                                                       self.read_color_image1)
        self.object_image_subscriber1 = rospy.Subscriber(self.topic_object1, self.color_hints[self.color_hint],
                                                        self.read_object_image1)
        self.depth_image_subscriber1 = rospy.Subscriber(self.topic_depth1, self.color_hints[self.color_hint],
                                                       self.read_depth_image1)

        self.color_image_subscriber = rospy.Subscriber(self.topic_color, self.color_hints[self.color_hint],
                                                        self.read_color_image)
        self.object_image_subscriber = rospy.Subscriber(self.topic_object, self.color_hints[self.color_hint],
                                                         self.read_object_image)
        self.depth_image_subscriber = rospy.Subscriber(self.topic_depth, self.color_hints[self.color_hint],
                                                        self.read_depth_image)

    def read_color_image(self,image):
        image = self.bridge.imgmsg_to_cv2(image, self.cvMode)
        self.color_image = self.camera_resize([image], self.iwidth, self.iheight)[0]

    def read_object_image(self,image):
        image = self.bridge.imgmsg_to_cv2(image, self.cvMode)
        self.object_image = self.camera_resize([image], self.iwidth, self.iheight)[0]

    def read_depth_image(self,image):
        image = self.bridge.imgmsg_to_cv2(image, '32FC1')
        image=np.ceil((np.array(image)/np.max(image))*255)
        image=np.uint8(image)
        image=cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        print(image.shape)
        self.depth_image = self.camera_resize([image], self.iwidth, self.iheight)[0]
        """
        self.mutex = mutex.mutex()
        self.mutex.testandset()
        self.mutex.unlock()
        """

    def read_color_image1(self, image):
        image = self.bridge.imgmsg_to_cv2(image, self.cvMode)
        self.color_image1 = self.camera_resize([image], self.iwidth, self.iheight)[0]

    def read_object_image1(self, image):
        image = self.bridge.imgmsg_to_cv2(image, self.cvMode)
        self.object_image1 = self.camera_resize([image], self.iwidth, self.iheight)[0]

    def read_depth_image1(self, image):
        image = self.bridge.imgmsg_to_cv2(image, '32FC1')
        image = np.ceil((np.array(image) / np.max(image)) * 255)
        image = np.uint8(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        print(image.shape)
        self.depth_image1 = self.camera_resize([image], self.iwidth, self.iheight)[0]
        """
        self.mutex = mutex.mutex()
        self.mutex.testandset()
        self.mutex.unlock()
        """

    def initialize_config_variables(self):
        self.step=0.01
        self.steps=0.3
        self.sleep_time=1/25.

    def callback(self, msgs):
        pass

    def initialize_robot_pose(self):
        self.robot_pose = Odometry()
        # self.robot_pose.header.seq=0
        # self.robot_pose.header.stamp=0
        # self.robot_pose.header.frame_id=self.robot_base_joint_name
        # self.robot_pose.child_frame_id=self.robot_base_joint_name
        self.robot_pose.pose.pose.position.x = 4.66
        self.robot_pose.pose.pose.position.y = 2.61
        self.robot_pose.pose.pose.position.z = 0.0

        self.robot_pose.pose.pose.orientation.x=0
        self.robot_pose.pose.pose.orientation.y = 0
        self.robot_pose.pose.pose.orientation.z = 0
        self.robot_pose.pose.pose.orientation.w = 1
        # self.robot_pose.pose.covariance=np.identity(36, dtype='float64')
        # self.robot_pose.twist.twist.angular=0
        # self.robot_pose.twist.twist.linear = 0
        # self.robot_pose.twist.covariance=np.identity(36, dtype='float64')
        self.robot_base_upper_bounds = [17.87, 4.44,0.0,0.0,0.0,1.0,+np.Inf]
        self.robot_base_lower_bounds = [0.8,0.5,0.0,0.0,0.0,1.0,-np.Inf]

    def initialize_robot_head(self):
        self.head_tilt = JointState()
        self.head_tilt.name = ['head_tilt_joint']
        self.head_tilt.position = [0]
        self.head_pan=JointState()
        self.head_pan.name=['head_pan_joint']
        self.head_pan.position = [0]
        self.head_upper_bounds = [np.pi / 2, np.pi / 2]
        self.head_lower_bounds = [-np.pi / 2, -np.pi / 2]

    def initialize_robot_grippers(self):

        self.grippers = JointState()
        self.grippers.name = ['l_gripper_l_finger_joint','l_gripper_r_finger_joint','r_gripper_l_finger_joint','r_gripper_r_finger_joint']
        self.grippers.position = [0,0,0,0]
        self.grippers_upper_bounds=[np.pi/2,np.pi/2,np.pi/2,np.pi/2]
        self.grippers_lower_bounds = [0.,0.,0.,0.]

    def initialize_robot_arms(self):
        self.parkedArms=JointState()
        self.parkedArms.name=['r_shoulder_pan_joint','l_shoulder_pan_joint', 'r_elbow_flex_joint', 'l_elbow_flex_joint']
        self.arms_upper_bounds = [np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2]
        self.arms_lower_bounds = [-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2]

    def park_robot_arms(self, progressive=True):
        self.parkedArms.position = [-1.57,1.57,-1.57,-1.57]
        for i in range(len(self.parkedArms.position)):
            if self.parkedArms.position[i]> self.arms_upper_bounds[i]:
                self.parkedArms.position[i]=self.arms_upper_bounds[i]
                continue
            if self.parkedArms.position[i]< self.arms_lower_bounds[i]:
                self.parkedArms.position[i]=self.arms_lower_bounds[i]
                continue
        self.robot_joint_publisher.publish(self.parkedArms)

    def turn_robot_head_up(self, steps=0.3, step=0.01, progressive=True):
        if progressive:
            while (steps > 0):
                steps -= step
                if self.head_tilt.position[0]- step<self.head_lower_bounds[0]:
                    self.head_tilt.position[0]=self.head_lower_bounds[0]
                else:
                    if self.head_tilt.position[0] - step > self.head_upper_bounds[0]:
                        self.head_tilt.position[0] = self.head_upper_bounds[0]
                    else:
                        self.head_tilt.position[0]-= step
                self.robot_joint_publisher.publish(self.head_tilt)
                time.sleep(self.sleep_time)
        else:
            if self.head_tilt.position[0] - steps < self.head_lower_bounds[0]:
                self.head_tilt.position[0] = self.head_lower_bounds[0]
            else:
                if self.head_tilt.position[0] - steps > self.head_upper_bounds[0]:
                    self.head_tilt.position[0] = self.head_upper_bounds[0]
                else:
                    self.head_tilt.position[0] -= steps
            self.robot_joint_publisher.publish(self.head_tilt)

    def turn_robot_head_down(self, steps=0.3, step=0.01, progressive=True):
        if progressive:
            while (steps > 0):
                steps -= step
                if self.head_tilt.position[0] + step < self.head_lower_bounds[0]:
                    self.head_tilt.position[0] = self.head_lower_bounds[0]
                else:
                    if self.head_tilt.position[0] + step > self.head_upper_bounds[0]:
                        self.head_tilt.position[0] = self.head_upper_bounds[0]
                    else:
                        self.head_tilt.position[0] += step
                self.robot_joint_publisher.publish(self.head_tilt)
                time.sleep(self.sleep_time)
        else:
            if self.head_tilt.position[0] + steps < self.head_lower_bounds[0]:
                self.head_tilt.position[0] = self.head_lower_bounds[0]
            else:
                if self.head_tilt.position[0] + steps > self.head_upper_bounds[0]:
                    self.head_tilt.position[0] = self.head_upper_bounds[0]
                else:
                    self.head_tilt.position[0] += steps
            self.robot_joint_publisher.publish(self.head_tilt)

    def turn_robot_head_right(self, steps=0.3, step=0.01, progressive=True):
        if progressive:
            while (steps > 0):
                steps -= step
                if self.head_pan.position[0] - step < self.head_lower_bounds[1]:
                    self.head_pan.position[0] = self.head_lower_bounds[1]
                else:
                    if self.head_pan.position[0] - step > self.head_upper_bounds[1]:
                        self.head_pan.position[0] = self.head_upper_bounds[1]
                    else:
                        self.head_pan.position[0] -= step
                self.robot_joint_publisher.publish(self.head_pan)
                time.sleep(self.sleep_time)
        else:
            if self.head_pan.position[0] - steps < self.head_lower_bounds[1]:
                self.head_pan.position[0] = self.head_lower_bounds[1]
            else:
                if self.head_pan.position[0] - steps > self.head_upper_bounds[1]:
                    self.head_pan.position[0] = self.head_upper_bounds[1]
                else:
                    self.head_pan.position[0] -= steps
            self.robot_joint_publisher.publish(self.head_pan)

    def turn_robot_head_left(self, steps=0.3, step=0.01, progressive=True):
        if progressive:
            while (steps > 0):
                steps -= step
                if self.head_pan.position[0] + step < self.head_lower_bounds[1]:
                    self.head_pan.position[0] = self.head_lower_bounds[1]
                else:
                    if self.head_pan.position[0] + step > self.head_upper_bounds[1]:
                        self.head_pan.position[0] = self.head_upper_bounds[1]
                    else:
                        self.head_pan.position[0] += step
                self.robot_joint_publisher.publish(self.head_pan)
                time.sleep(self.sleep_time)
        else:
            if self.head_pan.position[0] + steps < self.head_lower_bounds[1]:
                self.head_pan.position[0] = self.head_lower_bounds[1]
            else:
                if self.head_pan.position[0] + steps > self.head_upper_bounds[1]:
                    self.head_pan.position[0] = self.head_upper_bounds[1]
                else:
                    self.head_pan.position[0] += steps
            self.robot_joint_publisher.publish(self.head_pan)

    def move_robot_forward(self, steps=0.3, step=0.01, progressive=True):
        # self.robot_pose.header.seq=0
        # self.robot_pose.header.stamp=0
        # self.robot_pose.header.frame_id=self.robot_base_joint_name
        # self.robot_pose.child_frame_id=self.robot_base_joint_name
        if progressive:
            while(steps>0):
                steps-=step
                if self.robot_pose.pose.pose.position.x+step>self.robot_base_upper_bounds[0]:
                    self.robot_pose.pose.pose.position.x=self.robot_base_upper_bounds[0]
                else:
                    if self.robot_pose.pose.pose.position.x + step < self.robot_base_lower_bounds[0]:
                        self.robot_pose.pose.pose.position.x = self.robot_base_lower_bounds[0]
                    else:
                        self.robot_pose.pose.pose.position.x +=step
                #self.robot_pose.pose.pose.position.y = -2.6
                #self.robot_pose.pose.pose.position.z = 0.1
                # self.robot_pose.pose.pose.orientation.x=0
                # self.robot_pose.pose.pose.orientation.y = 0
                # self.robot_pose.pose.pose.orientation.z = 0
                # self.robot_pose.pose.pose.orientation.w = 1
                # self.robot_pose.pose.covariance=np.identity(36, dtype='float64')
                # self.robot_pose.twist.twist.angular=0
                # self.robot_pose.twist.twist.linear = 0
                # self.robot_pose.twist.covariance=np.identity(36, dtype='float64')
                self.robot_base_publisher.publish(self.robot_pose)
                time.sleep(self.sleep_time)
        else:
            if self.robot_pose.pose.pose.position.x + steps > self.robot_base_upper_bounds[0]:
                self.robot_pose.pose.pose.position.x = self.robot_base_upper_bounds[0]
            else:
                if self.robot_pose.pose.pose.position.x + steps < self.robot_base_lower_bounds[0]:
                    self.robot_pose.pose.pose.position.x = self.robot_base_lower_bounds[0]
                else:
                    self.robot_pose.pose.pose.position.x += steps
            self.robot_base_publisher.publish(self.robot_pose)

    def move_robot_backward(self, steps=0.3, step=0.01, progressive=True):
        # self.robot_pose.header.seq=0
        # self.robot_pose.header.stamp=0
        # self.robot_pose.header.frame_id=self.robot_base_joint_name
        # self.robot_pose.child_frame_id=self.robot_base_joint_name
        if progressive:
            while (steps > 0):
                steps -= step
                if self.robot_pose.pose.pose.position.x - step > self.robot_base_upper_bounds[0]:
                    self.robot_pose.pose.pose.position.x = self.robot_base_upper_bounds[0]
                else:
                    if self.robot_pose.pose.pose.position.x - step < self.robot_base_lower_bounds[0]:
                        self.robot_pose.pose.pose.position.x = self.robot_base_lower_bounds[0]
                    else:
                        self.robot_pose.pose.pose.position.x -= step
                # self.robot_pose.pose.pose.position.y = -2.6
                # self.robot_pose.pose.pose.position.z = 0.1
                # self.robot_pose.pose.pose.orientation.x=0
                # self.robot_pose.pose.pose.orientation.y = 0
                # self.robot_pose.pose.pose.orientation.z = 0
                # self.robot_pose.pose.pose.orientation.w = 1
                # self.robot_pose.pose.covariance=np.identity(36, dtype='float64')
                # self.robot_pose.twist.twist.angular=0
                # self.robot_pose.twist.twist.linear = 0
                # self.robot_pose.twist.covariance=np.identity(36, dtype='float64')
                self.robot_base_publisher.publish(self.robot_pose)
                time.sleep(self.sleep_time)
        else:
            if self.robot_pose.pose.pose.position.x - steps > self.robot_base_upper_bounds[0]:
                self.robot_pose.pose.pose.position.x = self.robot_base_upper_bounds[0]
            else:
                if self.robot_pose.pose.pose.position.x - steps < self.robot_base_lower_bounds[0]:
                    self.robot_pose.pose.pose.position.x = self.robot_base_lower_bounds[0]
                else:
                    self.robot_pose.pose.pose.position.x -= steps
            self.robot_base_publisher.publish(self.robot_pose)

    def turn_robot_right(self, steps=0.3, step=0.01, progressive=True):
        # self.robot_pose.header.seq=0
        # self.robot_pose.header.stamp=0
        # self.robot_pose.header.frame_id=self.robot_base_joint_name
        # self.robot_pose.child_frame_id=self.robot_base_joint_name
        q=Quaternion(self.robot_pose.pose.pose.orientation.w,self.robot_pose.pose.pose.orientation.x, self.robot_pose.pose.pose.orientation.y,self.robot_pose.pose.pose.orientation.z)
        angle=q.angle
        axis=q.axis
        axis=np.array([0,0,1])
        if progressive:
            while (steps > 0):
                if angle-step>self.robot_base_upper_bounds[6]:
                    angle=self.robot_base_upper_bounds[6]
                else:
                    if angle - step < self.robot_base_lower_bounds[6]:
                        angle = self.robot_base_lower_bounds[6]
                    else:
                        angle-=step
                steps-=step
                p=Quaternion(axis=axis, angle=angle)
                print (axis, angle)
                # self.robot_pose.pose.pose.position.y = -2.6
                # self.robot_pose.pose.pose.position.z = 0.1
                self.robot_pose.pose.pose.orientation.x = p.x
                self.robot_pose.pose.pose.orientation.y = p.y
                self.robot_pose.pose.pose.orientation.z = p.z
                self.robot_pose.pose.pose.orientation.w = p.w
                # self.robot_pose.pose.covariance=np.identity(36, dtype='float64')
                # self.robot_pose.twist.twist.angular=0
                # self.robot_pose.twist.twist.linear = 0
                # self.robot_pose.twist.covariance=np.identity(36, dtype='float64')
                self.robot_base_publisher.publish(self.robot_pose)
                time.sleep(self.sleep_time)
        else:
            if angle - steps > self.robot_base_upper_bounds[6]:
                angle = self.robot_base_upper_bounds[6]
            else:
                if angle - steps < self.robot_base_lower_bounds[6]:
                    angle = self.robot_base_lower_bounds[6]
                else:
                    angle -= steps
            p = Quaternion(axis=axis, angle=angle)
            self.robot_pose.pose.pose.orientation.x = p.x
            self.robot_pose.pose.pose.orientation.y = p.y
            self.robot_pose.pose.pose.orientation.z = p.z
            self.robot_pose.pose.pose.orientation.w = p.w
            self.robot_base_publisher.publish(self.robot_pose)

    def turn_robot_left(self, steps=0.3, step=0.01, progressive=True):
        # self.robot_pose.header.seq=0
        # self.robot_pose.header.stamp=0
        # self.robot_pose.header.frame_id=self.robot_base_joint_name
        # self.robot_pose.child_frame_id=self.robot_base_joint_name
        q = Quaternion(self.robot_pose.pose.pose.orientation.w, self.robot_pose.pose.pose.orientation.x,
                       self.robot_pose.pose.pose.orientation.y, self.robot_pose.pose.pose.orientation.z)
        angle = q.angle
        axis = q.axis
        axis = np.array([0, 0, 1])
        if progressive:
            while (steps > 0):
                if angle + step > self.robot_base_upper_bounds[6]:
                    angle = self.robot_base_upper_bounds[6]
                else:
                    if angle + step < self.robot_base_lower_bounds[6]:
                        angle = self.robot_base_lower_bounds[6]
                    else:
                        angle += step
                steps -= step
                p = Quaternion(axis=axis, angle=angle)
                print(axis, angle)
                # self.robot_pose.pose.pose.position.y = -2.6
                # self.robot_pose.pose.pose.position.z = 0.1
                self.robot_pose.pose.pose.orientation.x = p.x
                self.robot_pose.pose.pose.orientation.y = p.y
                self.robot_pose.pose.pose.orientation.z = p.z
                self.robot_pose.pose.pose.orientation.w = p.w
                # self.robot_pose.pose.covariance=np.identity(36, dtype='float64')
                # self.robot_pose.twist.twist.angular=0
                # self.robot_pose.twist.twist.linear = 0
                # self.robot_pose.twist.covariance=np.identity(36, dtype='float64')
                self.robot_base_publisher.publish(self.robot_pose)
                time.sleep(self.sleep_time)
        else:
            if angle + steps > self.robot_base_upper_bounds[6]:
                angle = self.robot_base_upper_bounds[6]
            else:
                if angle + steps < self.robot_base_lower_bounds[6]:
                    angle = self.robot_base_lower_bounds[6]
                else:
                    angle += steps
            p = Quaternion(axis=axis, angle=angle)
            self.robot_pose.pose.pose.orientation.x = p.x
            self.robot_pose.pose.pose.orientation.y = p.y
            self.robot_pose.pose.pose.orientation.z = p.z
            self.robot_pose.pose.pose.orientation.w = p.w
            self.robot_base_publisher.publish(self.robot_pose)

    def move_robot_left(self, steps=0.3, step=0.01, progressive=True):
        # self.robot_pose.header.seq=0
        # self.robot_pose.header.stamp=0
        # self.robot_pose.header.frame_id=self.robot_base_joint_name
        # self.robot_pose.child_frame_id=self.robot_base_joint_name
        if progressive:
            while (steps > 0):
                steps -= step
                if self.robot_pose.pose.pose.position.y + step > self.robot_base_upper_bounds[1]:
                    self.robot_pose.pose.pose.position.y = self.robot_base_upper_bounds[1]
                else:
                    if self.robot_pose.pose.pose.position.y + step < self.robot_base_lower_bounds[1]:
                        self.robot_pose.pose.pose.position.y = self.robot_base_lower_bounds[1]
                    else:
                        self.robot_pose.pose.pose.position.y += step
                # self.robot_pose.pose.pose.position.y = -2.6
                # self.robot_pose.pose.pose.position.z = 0.1
                # self.robot_pose.pose.pose.orientation.x=0
                # self.robot_pose.pose.pose.orientation.y = 0
                # self.robot_pose.pose.pose.orientation.z = 0
                # self.robot_pose.pose.pose.orientation.w = 1
                # self.robot_pose.pose.covariance=np.identity(36, dtype='float64')
                # self.robot_pose.twist.twist.angular=0
                # self.robot_pose.twist.twist.linear = 0
                # self.robot_pose.twist.covariance=np.identity(36, dtype='float64')
                self.robot_base_publisher.publish(self.robot_pose)
                time.sleep(self.sleep_time)
        else:
            if self.robot_pose.pose.pose.position.y + steps > self.robot_base_upper_bounds[1]:
                self.robot_pose.pose.pose.position.y = self.robot_base_upper_bounds[1]
            else:
                if self.robot_pose.pose.pose.position.y + steps < self.robot_base_lower_bounds[1]:
                    self.robot_pose.pose.pose.position.y = self.robot_base_lower_bounds[1]
                else:
                    self.robot_pose.pose.pose.position.y += steps
            self.robot_base_publisher.publish(self.robot_pose)

    def move_robot_right(self, steps=0.3, step=0.01, progressive=True):
        # self.robot_pose.header.seq=0
        # self.robot_pose.header.stamp=0
        # self.robot_pose.header.frame_id=self.robot_base_joint_name
        # self.robot_pose.child_frame_id=self.robot_base_joint_name
        if progressive:
            while (steps > 0):
                steps -= step
                if self.robot_pose.pose.pose.position.y - step > self.robot_base_upper_bounds[1]:
                    self.robot_pose.pose.pose.position.y = self.robot_base_upper_bounds[1]
                else:
                    if self.robot_pose.pose.pose.position.y - step < self.robot_base_lower_bounds[1]:
                        self.robot_pose.pose.pose.position.y = self.robot_base_lower_bounds[1]
                    else:
                        self.robot_pose.pose.pose.position.y -= step
                # self.robot_pose.pose.pose.position.y = -2.6
                # self.robot_pose.pose.pose.position.z = 0.1
                # self.robot_pose.pose.pose.orientation.x=0
                # self.robot_pose.pose.pose.orientation.y = 0
                # self.robot_pose.pose.pose.orientation.z = 0
                # self.robot_pose.pose.pose.orientation.w = 1
                # self.robot_pose.pose.covariance=np.identity(36, dtype='float64')
                # self.robot_pose.twist.twist.angular=0
                # self.robot_pose.twist.twist.linear = 0
                # self.robot_pose.twist.covariance=np.identity(36, dtype='float64')
                self.robot_base_publisher.publish(self.robot_pose)
                time.sleep(self.sleep_time)
        else:
            if self.robot_pose.pose.pose.position.y - steps > self.robot_base_upper_bounds[1]:
                self.robot_pose.pose.pose.position.y = self.robot_base_upper_bounds[1]
            else:
                if self.robot_pose.pose.pose.position.y - steps < self.robot_base_lower_bounds[1]:
                    self.robot_pose.pose.pose.position.y = self.robot_base_lower_bounds[1]
                else:
                    self.robot_pose.pose.pose.position.y -= steps
            self.robot_base_publisher.publish(self.robot_pose)

    def close_robot_right_gripper(self, steps=0.3, step=0.01, progressive=True):
        index=2
        if progressive:
            while (steps > 0):
                steps -= step
                if self.grippers.position[index]-step<self.grippers_lower_bounds[index]:
                    self.grippers.position[index] = self.grippers_lower_bounds[index]
                    self.grippers.position[index + 1] = self.grippers_lower_bounds[index]
                else:
                    if self.grippers.position[index] - step > self.grippers_upper_bounds[index]:
                        self.grippers.position[index] = self.grippers_upper_bounds[index]
                        self.grippers.position[index + 1] = self.grippers_upper_bounds[index]
                    else:
                        self.grippers.position[index] -= step
                        self.grippers.position[index+1] -= step
                self.robot_joint_publisher.publish(self.grippers)
                time.sleep(self.sleep_time)
        else:
            if self.grippers.position[index] - steps < self.grippers_lower_bounds[index]:
                self.grippers.position[index] = self.grippers_lower_bounds[index]
                self.grippers.position[index + 1] = self.grippers_lower_bounds[index]
            else:
                if self.grippers.position[index] - steps > self.grippers_upper_bounds[index]:
                    self.grippers.position[index] = self.grippers_upper_bounds[index]
                    self.grippers.position[index + 1] = self.grippers_upper_bounds[index]
                else:
                    self.grippers.position[index] -= steps
                    self.grippers.position[index + 1] -= steps
            self.robot_joint_publisher.publish(self.grippers)

    def close_robot_left_gripper(self, steps=0.3, step=0.01, progressive=True):
        index = 0
        if progressive:
            while (steps > 0):
                steps -= step
                if self.grippers.position[index] - step < self.grippers_lower_bounds[index]:
                    self.grippers.position[index] = self.grippers_lower_bounds[index]
                    self.grippers.position[index + 1] = self.grippers_lower_bounds[index]
                else:
                    if self.grippers.position[index] - step > self.grippers_upper_bounds[index]:
                        self.grippers.position[index] = self.grippers_upper_bounds[index]
                        self.grippers.position[index + 1] = self.grippers_upper_bounds[index]
                    else:
                        self.grippers.position[index] -= step
                        self.grippers.position[index + 1] -= step
                self.robot_joint_publisher.publish(self.grippers)
                time.sleep(self.sleep_time)
        else:
            if self.grippers.position[index] - steps < self.grippers_lower_bounds[index]:
                self.grippers.position[index] = self.grippers_lower_bounds[index]
                self.grippers.position[index + 1] = self.grippers_lower_bounds[index]
            else:
                if self.grippers.position[index] - steps > self.grippers_upper_bounds[index]:
                    self.grippers.position[index] = self.grippers_upper_bounds[index]
                    self.grippers.position[index + 1] = self.grippers_upper_bounds[index]
                else:
                    self.grippers.position[index] -= steps
                    self.grippers.position[index + 1] -= steps
            self.robot_joint_publisher.publish(self.grippers)

    def open_robot_left_gripper(self, steps=0.3, step=0.01, progressive=True):
        index = 0
        if progressive:
            while (steps > 0):
                steps -= step
                if self.grippers.position[index] + step < self.grippers_lower_bounds[index]:
                    self.grippers.position[index] = self.grippers_lower_bounds[index]
                    self.grippers.position[index + 1] = self.grippers_lower_bounds[index]
                else:
                    if self.grippers.position[index] + step > self.grippers_upper_bounds[index]:
                        self.grippers.position[index] = self.grippers_upper_bounds[index]
                        self.grippers.position[index + 1] = self.grippers_upper_bounds[index]
                    else:
                        self.grippers.position[index] += step
                        self.grippers.position[index + 1] += step
                self.robot_joint_publisher.publish(self.grippers)
                time.sleep(self.sleep_time)
        else:
            if self.grippers.position[index] + steps < self.grippers_lower_bounds[index]:
                self.grippers.position[index] = self.grippers_lower_bounds[index]
                self.grippers.position[index + 1] = self.grippers_lower_bounds[index]
            else:
                if self.grippers.position[index] + steps > self.grippers_upper_bounds[index]:
                    self.grippers.position[index] = self.grippers_upper_bounds[index]
                    self.grippers.position[index + 1] = self.grippers_upper_bounds[index]
                else:
                    self.grippers.position[index] += steps
                    self.grippers.position[index + 1] += steps
            self.robot_joint_publisher.publish(self.grippers)

    def open_robot_right_gripper(self, steps=0.3, step=0.01, progressive=True):
        index = 2
        if progressive:
            while (steps > 0):
                steps -= step
                if self.grippers.position[index] + step < self.grippers_lower_bounds[index]:
                    self.grippers.position[index] = self.grippers_lower_bounds[index]
                    self.grippers.position[index + 1] = self.grippers_lower_bounds[index]
                else:
                    if self.grippers.position[index] + step > self.grippers_upper_bounds[index]:
                        self.grippers.position[index] = self.grippers_upper_bounds[index]
                        self.grippers.position[index + 1] = self.grippers_upper_bounds[index]
                    else:
                        self.grippers.position[index] += step
                        self.grippers.position[index + 1] += step
                self.robot_joint_publisher.publish(self.grippers)
                time.sleep(self.sleep_time)
        else:
            if self.grippers.position[index] + steps < self.grippers_lower_bounds[index]:
                self.grippers.position[index] = self.grippers_lower_bounds[index]
                self.grippers.position[index + 1] = self.grippers_lower_bounds[index]
            else:
                if self.grippers.position[index] + steps > self.grippers_upper_bounds[index]:
                    self.grippers.position[index] = self.grippers_upper_bounds[index]
                    self.grippers.position[index + 1] = self.grippers_upper_bounds[index]
                else:
                    self.grippers.position[index] += steps
                    self.grippers.position[index + 1] += steps
            self.robot_joint_publisher.publish(self.grippers)

    def observe(self):
        if self.robot_view==1:
            return [self.color_image1, self.object_image1, self.depth_image1]
        else:
            return [self.color_image, self.object_image, self.depth_image]