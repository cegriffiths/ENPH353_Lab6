
import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Lab06_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/enph353_lab6/launch/lab06_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected

        self.lower_blue = np.array([97,  0,   0])
        self.upper_blue = np.array([150, 255, 255])

    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # cv2.imshow("raw", cv_image)

        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        done = False
        gaussianKernel = (5, 5)
        threshold = 160

        # TODO: Analyze the cv_image and compute the state array and
        # episode termination condition.
        #
        # The state array is a list of 10 elements indicating where in the
        # image the line is:
        # i.e.
        #    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] indicates line is on the left
        #    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] indicates line is in the center
        #
        # The episode termination condition should be triggered when the line
        # is not detected for more than 30 frames. In this case set the done
        # variable to True.
        #
        # You can use the self.timeout variable to keep track of which frames
        # have no line detected.
        
        #Slice the image to only take the bottom into consideration
        img_slice = cv_image[200:240, :]
        # Convert the frame to a different grayscale
        img_gray = cv2.cvtColor(img_slice, cv2.COLOR_RGB2GRAY)
        # Blur image to reduce noise
        img_blur = cv2.GaussianBlur(img_gray, gaussianKernel, 0)
        # Binarize the image
        _, img_bin = cv2.threshold(img_blur, threshold, 255, cv2.THRESH_BINARY_INV)
        # Find the contours of the image
        contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Overlay the contours onto the original image
        # img_cont = cv2.drawContours(cv_image, contours, -1, (0, 0, 0), 1)

        # Check that contours contains at least one contour then on the largest contour, find the centre of its centre
        # Print a circle at the centre of mass on the original image
        if len(contours) > 0:
            maxCont = max(contours, key=cv2.contourArea)
            contMoments = cv2.moments(maxCont)

            # Check if the contour contains any pixels
            if contMoments['m00'] > 0:
                centre = (int(contMoments['m10']/contMoments['m00']), int(contMoments['m01']/contMoments['m00']))
                img_circle = cv2.circle(cv_image, (centre[0], centre[1] + 200), 5, (0, 0, 255), -1)

                # Find if the index of which state the robot is in
                state_index = (10 * centre[0]) // cv_image.shape[1]
                state[state_index] = 1

                img_state = cv2.putText(img_circle, str(state), (0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                cv2.imshow("Contour Image",img_circle)
                cv2.waitKey(3)

                self.timeout = 0
            else:
                self.timeout += 1

                if self.timeout > 3:
                    done = True
        
        # Run this if no contours are found, this means we do not see the line, so imcrement timeout
        # if timeout is above 30 reset the simulation
        else:
            self.timeout += 1

            if self.timeout > 3:
                done = True

        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            # vel_cmd.angular.z = 0.5
            vel_cmd.angular.z = 0.8
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            # vel_cmd.angular.z = -0.5
            vel_cmd.angular.z = -0.8

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        # Set the rewards for your action
        if not done:
            if action == 0:  # FORWARD
                reward = 4
            elif action == 1:  # LEFT
                # reward = 2
                reward = 0
            else:
                # reward = 2  # RIGHT
                reward = 0
            
            if np.argmax(state) == 4 or np.argmax(state) == 5:
                reward = 2
            elif np.argmax(state) == 0 or np.argmax(state) == 9:
                reward = -4

        else:
            reward = -200

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
