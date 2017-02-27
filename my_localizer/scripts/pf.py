#!/usr/bin/env python

"""Danny and Arpan's Robot Localization Project"""

from __future__ import division

import rospy

from std_msgs.msg import Header, String
from sensor_msgs.msg import LaserScan, PointCloud
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PoseArray, Pose, Point, Quaternion
from nav_msgs.srv import GetMap
from copy import deepcopy

import tf
from tf import TransformListener
from tf import TransformBroadcaster
from tf.transformations import euler_from_quaternion, rotation_matrix, quaternion_from_matrix
from random import gauss

import math
import time

import numpy as np
from numpy.random import random_sample
from sklearn.neighbors import NearestNeighbors
from occupancy_field import OccupancyField
from scipy import stats

from helper_functions import (convert_pose_inverse_transform,
                              convert_translation_rotation_to_pose,
                              convert_pose_to_xy_and_theta,
                              angle_diff, angle_normalize)


class ParticleCloud(object):

    """ Provides a view into the numpy particles matrix as an iterable object
        A wrapper for the array.
    """

    def __init__(self, np_particles):
        super(ParticleCloud, self).__init__()
        self.np_particles = np_particles

    def __len__(self):
        return self.np_particles.shape[0]

    def __getitem__(self, key):
        return Particle(self.np_particles[key, :])


class Particle(object):

    """ Represents a hypothesis (particle) of the robot's pose consisting of
        x,y and theta (yaw)
        Attributes:
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle
               weights are normalized
    """

    # def __init__(self,x=0.0,y=0.0,theta=0.0,w=1.0):
    #     """ Construct a new Particle
    #         x: the x-coordinate of the hypothesis relative to the map frame
    #         y: the y-coordinate of the hypothesis relative ot the map frame
    #         theta: the yaw of the hypothesis relative to the map frame
    #         w: the particle weight (the class does not ensure that particle
    #            weights are normalized
    #     """
    #     self.w = w
    #     self.theta = theta
    #     self.x = x
    #     self.y = y
    name_mapping = dict(w=0, x=1, y=2, theta=3)

    def __init__(self, np_particle):
        self.view = np_particle

    def __getattr__(self, name):
        if name in Particle.name_mapping:
            index = Particle.name_mapping[name]
            return self.view[index]
        else:
            raise AttributeError(
                "%r object has no attribute %r" % (self.__class__, attr)
            )

    def __setattr__(self, name, value):
        if name in Particle.name_mapping:
            index = Particle.name_mapping[name]
            self.view[index] = value
        else:
            object.__setattr__(self, name, value)

    def transform_scan(self, scan_points):
        """ Transforms a set of laser scan points from the base_link frame into
            the local to this particle frame.

            scan_points: numpy matrix of n rows, 2 columns (x, y)

            returns an n row, 2 column matrix of transformed points
        """
        # translate scan_points
        scan_points = scan_points + [self.x, self.y]

        # generate rotation matrix
        c, s = np.cos(self.theta), np.sin(self.theta)
        rotation_matrix = np.matrix([[c, -s], [s, c]])

        # rotate points
        return scan_points * rotation_matrix

    def as_pose(self):
        """ A helper function to convert a particle to a geometry_msgs/Pose
            message
        """
        orientation_tuple = tf.transformations.quaternion_from_euler(
            0,
            0,
            self.theta
        )

        return Pose(
            position=Point(x=self.x, y=self.y, z=0),
            orientation=Quaternion(
                x=orientation_tuple[0],
                y=orientation_tuple[1],
                z=orientation_tuple[2],
                w=orientation_tuple[3]
            )
        )

    # TODO: define additional helper functions if needed


class ParticleFilter:

    """ The class that represents a Particle Filter ROS Node
        Attributes list:
            initialized: a Boolean flag to communicate to other class methods
                    that initializaiton is complete
            base_frame: the name of the robot base coordinate frame
                    (should be "base_link" for most robots)
            map_frame: the name of the map coordinate frame
                    (should be "map" in most cases)
            odom_frame: the name of the odometry coordinate frame
                    (should be "odom" in most cases)
            scan_topic: the name of the scan topic to listen to
                    (should be "scan" in most cases)
            n_particles: the number of particles in the filter
            d_thresh: the amount of linear movement before triggering a filter
                    update
            a_thresh: the amount of angular movement before triggering a filter
                    update
            laser_max_distance: the maximum distance to an obstacle we should
                    use in a likelihood calculation
            pose_listener: a subscriber that listens for new approximate pose
                    estimates (i.e. generated through the rviz GUI)
            particle_pub: a publisher for the particle cloud
            laser_subscriber: listens for new scan data on topic
                    self.scan_topic
            tf_listener: listener for coordinate transforms
            tf_broadcaster: broadcaster for coordinate transforms
            particle_cloud: a list of particles representing a probability
                    distribution over robot poses
            current_odom_xy_theta: the pose of the robot in the odometry frame
                    when the last filter update was performed.
                    The pose is expressed as a list [x,y,theta]
                    (where theta is the yaw)
            map: the map we will be localizing ourselves in.
                 The map should be of type nav_msgs/OccupancyGrid
    """

    def __init__(self):
        # make sure we don't perform updates before everything is setup
        self.initialized = False
        # tell roscore that we are creating a new node named "pf"
        rospy.init_node('pf')

        # the frame of the robot base
        self.base_frame = "base_link"
        # the name of the map coordinate frame
        self.map_frame = "map"
        # the name of the odometry coordinate frame
        self.odom_frame = "odom"
        # the topic where we will get laser scans from
        self.scan_topic = "scan"

        # the number of particles to use
        self.n_particles = 300

        # the amount of linear movement before performing an update
        self.d_thresh = 0.2

        # the amount of angular movement before performing an update
        self.a_thresh = math.pi / 6

        # maximum penalty to assess in the likelihood field model
        self.laser_max_distance = 2.0

        # spreads for init
        self.xy_spread = 0.25
        self.theta_spread = .02 * math.pi
        # odom update error
        self.xy_odom_spread = .005
        self.theta_odom_spread = .005 * math.pi
        # resampling induced error
        self.resample_x_scale = .25
        self.resample_y_scale = .25
        self.resample_theta_scale = .05 * math.pi

        # Setup pubs and subs

        # pose_listener responds to selection of a new approximate robot
        # location (for instance using rviz)
        rospy.Subscriber(
            "initialpose",
            PoseWithCovarianceStamped,
            self.update_initial_pose
        )

        # publish the current particle cloud.
        # This enables viewing particles in rviz.
        self.particle_pub = rospy.Publisher(
            "particlecloud",
            PoseArray,
            queue_size=10
        )

        # laser_subscriber listens for data from the lidar
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_received)

        # enable listening for and broadcasting coordinate transforms
        self.tf_listener = TransformListener()
        self.tf_broadcaster = TransformBroadcaster()

        self.particle_cloud = None

        # change use_projected_stable_scan to True to use point clouds instead
        # of laser scans
        self.use_projected_stable_scan = False
        self.last_projected_stable_scan = None
        if self.use_projected_stable_scan:
            # subscriber to the odom point cloud
            rospy.Subscriber(
                "projected_stable_scan",
                PointCloud,
                self.projected_scan_received
            )

        self.current_odom_xy_theta = None

        # request the map from the map server, the map should be of type
        # nav_msgs/OccupancyGrid
        # TODO: fill in the appropriate service call here.  The resultant map
        # should be assigned be passed
        #       into the init method for OccupancyField
        rospy.wait_for_service('static_map')
        try:
            get_map = rospy.ServiceProxy('static_map', GetMap)
            self.occupancy_field = OccupancyField(get_map().map)
        except rospy.ServiceException, e:
            print("Service call to get map failed: {}".format(e))

        # for now we have commented out the occupancy field initialization
        # until you can successfully fetch the map
        # self.occupancy_field = OccupancyField(map)
        self.initialized = True

    def update_robot_pose(self):
        """ Update the estimate of the robot's pose given the updated particles.
            There are two logical methods for this:
                (1): compute the mean pose
                (2): compute the most likely pose
                     (i.e. the mode of the distribution)
        """
        # make sure that the particle weights are normalized
        self.normalize_particles()
        # find the particle with the max weight and publish its pose
        max_weight_index = np.argmax(self.particle_cloud[:, 0])

        print("update robot pose index {}".format(max_weight_index))
        if max_weight_index != -1:
            particle = ParticleCloud(self.particle_cloud)[max_weight_index]
            self.robot_pose = particle.as_pose()
        else:
            self.robot_pose = Pose()

    def projected_scan_received(self, msg):
        self.last_projected_stable_scan = msg

    def update_particles_with_odom(self, msg):
        """ Update the particles using the newly given odometry pose.
            The function computes the value delta which is a numpy array (x,y,theta,0 (for weight))
            that indicates the change in position and angle between the
            odometry when the particles were last updated and the current
            odometry. It then adds that delta to every element in the particle cloud and normalizes the angles.

            msg: this is not really needed to implement this, but is here just
            in case.
        """
        new_odom_xy_theta = convert_pose_to_xy_and_theta(self.odom_pose.pose)
        # compute the change in x,y,theta since our last update
        if self.current_odom_xy_theta:
            old_odom_xy_theta = self.current_odom_xy_theta
            delta = np.array([
                self.current_odom_xy_theta[0] - new_odom_xy_theta[0],
                self.current_odom_xy_theta[1] - new_odom_xy_theta[1],
                angle_diff(new_odom_xy_theta[2], self.current_odom_xy_theta[2])
            ])

            self.current_odom_xy_theta = new_odom_xy_theta
        else:
            self.current_odom_xy_theta = new_odom_xy_theta
            return

        # random_delta_error based on odomspread constants and delta
        # simulates encoder error

        #make sure thetascale is not 0 even if theta has not changed
        if delta[2] == 0:
            delta[2] = .00005

        scale = np.array([
            abs(delta[0] * self.xy_odom_spread),
            abs(delta[1] * self.xy_odom_spread),
            abs(delta[2] * self.theta_odom_spread)
        ]) 

        
        random_delta_error = np.random.normal(
            loc=[0, 0, 0],
            scale=scale,
            size=(self.n_particles, 3)
        )

        # add zero to the weights
        # generate column of zeros
        zero_weights = np.zeros((self.n_particles, 1))

        # add delta to every element of random deltaerror
        random_delta = np.add(random_delta_error, delta)

        # concat weights with generated points
        random_delta = np.concatenate(
            (zero_weights, random_delta),
            axis=1
        )

        # add randomized delta to particle cloud
        self.particle_cloud = np.add(random_delta, self.particle_cloud)

        # normalize angles
        self.particle_cloud[:, 3] = angle_normalize(self.particle_cloud[:, 3])

    def map_calc_range(self, x, y, theta):
        """ Difficulty Level 3: implement a ray tracing likelihood model...
            Let me know if you are interested
        """
        # TODO: nothing unless you want to try this alternate likelihood model
        pass

    def resample_particles(self):
        """ Resample the particles according to the new particle weights.
            The weights stored with each particle should define the probability
            that a particular particle is selected in the resampling step.
            You may want to make use of the given helper
            function draw_random_sample.
        """
        # make sure the distribution is normalized
        self.normalize_particles()

        # make cloud of probable particles
        probable_particles = ParticleFilter.draw_random_sample(
            self.particle_cloud,  # choices
            self.particle_cloud[:, 0],  # probability weights
            self.n_particles  # number of points
        )

        probable_particles = np.array(probable_particles)

        # introduce random error to each particle

        x_error = np.random.sample(self.n_particles) * self.resample_x_scale
        y_error = np.random.sample(self.n_particles) * self.resample_y_scale
        theta_error = np.random.sample(self.n_particles) * self.resample_theta_scale

        probable_particles[:, 1] = probable_particles[:, 1] + x_error

        probable_particles[:, 2] = probable_particles[:, 2] + y_error

        probable_particles[:, 3] = probable_particles[:, 3] + theta_error

        # resample cloud with equal weights
        # generate column of ones
        initial_weights = np.ones(self.n_particles)
        probable_particles[:, 0] = initial_weights

        self.particle_cloud = probable_particles
        

    def get_likelihood(self, p_frame_points):
        # generate array of distances
        distances = []
        for point in p_frame_points[::10, :]:
            distances.append(
                self.occupancy_field.get_closest_obstacle_distance(
                    point[0, 0], point[0, 1]
                )
            )

        distances = np.array(distances)

        # if any of the points are off the map
        #if np.isnan(distances).any():
            #return 0.0  # eliminate this

        sigma = 0.1  # how noisy the measurements are
        likes = np.exp(-distances * distances / (2.0 * sigma * sigma))
        return np.mean(likes)

    def update_particles_with_laser(self, msg):
        """ Updates the particle weights in response to the scan contained in
            the msg
        """
        # create or reuse big rotation matrix to transform laser scan into
        # points around (0, 0)
        thetas = np.deg2rad(
            np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        )

        points = np.array([
            msg.ranges * np.cos(thetas),
            msg.ranges * np.sin(thetas)
        ]).T

        start_time = time.time()
        for p in ParticleCloud(self.particle_cloud):
            p_frame_points = p.transform_scan(points)

            p.w = self.get_likelihood(p_frame_points)

        print time.time() - start_time

        self.normalize_particles()

    @staticmethod
    def draw_random_sample(choices, probabilities, n):
        """ Return a random sample of n elements from the set choices with the
            specified probabilities
            choices: the values to sample from represented as a list
            probabilities: the probability of selecting each element in choices
            represented as a list
            n: the number of samples
        """
        values = np.array(range(len(choices)))
        probs = np.array(probabilities)
        bins = np.add.accumulate(probabilities)
        inds = values[np.digitize(random_sample(n), bins)]
        samples = []
        for i in inds:
            samples.append(deepcopy(choices[int(i)]))
        return samples

    def update_initial_pose(self, msg):
        """ Callback function to handle re-initializing the particle filter
            based on a pose estimate. These pose estimates could be generated
            by another ROS Node or could come from the rviz GUI
        """
        xy_theta = convert_pose_to_xy_and_theta(msg.pose.pose)
        self.initialize_particle_cloud(xy_theta)
        self.fix_map_to_odom_transform(msg)

    def initialize_particle_cloud(self, xy_theta=None):
        """ Initialize the particle cloud.
            Arguments
            xy_theta: a triple consisting of the mean x, y, and theta (yaw) to
                initialize the particle cloud around. If this input is
                ommitted, the odometry will be used
        """
        if xy_theta is None:
            xy_theta = convert_pose_to_xy_and_theta(self.odom_pose.pose)
        self.particle_cloud = None

        # generate random particles about XY_Theta
        random_points = np.random.normal(
            loc=xy_theta,
            scale=[self.xy_spread, self.xy_spread, self.theta_spread],
            size=(self.n_particles, 3)
        )

        # initialize cloud with equal weights
        # generate column of ones
        initial_weights = np.ones((self.n_particles, 1))

        # concat weights with generated points
        self.particle_cloud = np.concatenate(
            (initial_weights, random_points),
            axis=1
        )

        self.normalize_particles()
        self.update_robot_pose()

    def normalize_particles(self):
        """ Make sure the particle weights define a valid distribution
            (i.e. sum to 1.0)
        """
        weight_sum = np.sum(self.particle_cloud[:, 0])  # grab the weights

        normalized_weight_column = self.particle_cloud[:, 0] / weight_sum #normalize

        # reshape it so it's actually a column
        normalized_weight_column = normalized_weight_column.reshape(
            (normalized_weight_column.shape[0], 1)
        )

        # concatenate normalized weights with the rest of the array
        self.particle_cloud = np.concatenate(
            (normalized_weight_column, self.particle_cloud[:, 1:]),
            axis=1
        )

    def publish_particles(self, msg):
        particles_conv = []
        #translate between ParticleCloud object and our numpy array
        for p in ParticleCloud(self.particle_cloud):
            particles_conv.append(p.as_pose())
        # actually send the message so that we can view it in rviz
        self.particle_pub.publish(
            PoseArray(
                header=Header(
                    stamp=rospy.Time.now(),
                    frame_id=self.map_frame
                ),
                poses=particles_conv
            )
        )

    def scan_received(self, msg):
        """ This is the default logic for what to do when processing scan data.
            Feel free to modify this, however, I hope it will provide a good
            guide.  The input msg is an object of type sensor_msgs/LaserScan
        """
        if not self.initialized:
            # wait for initialization to complete
            return

        can_transform_laser_to_base = self.tf_listener.canTransform(
            self.base_frame,
            msg.header.frame_id,
            msg.header.stamp
        )
        if not can_transform_laser_to_base:
            # need to know how to transform the laser to the base frame
            # this will be given by either Gazebo or neato_node
            return

        can_transform_base_do_odom = self.tf_listener.canTransform(
            self.base_frame,
            self.odom_frame,
            msg.header.stamp
        )
        if not can_transform_base_do_odom:
            # need to know how to transform between base and odometric frames
            # this will eventually be published by either Gazebo or neato_node
            return

        # calculate pose of laser relative ot the robot base
        p = PoseStamped(
            header=Header(
                stamp=rospy.Time(0),
                frame_id=msg.header.frame_id
            )
        )
        self.laser_pose = self.tf_listener.transformPose(self.base_frame, p)

        # find out where the robot thinks it is based on its odometry
        p = PoseStamped(
            header=Header(
                stamp=msg.header.stamp,
                frame_id=self.base_frame
            ),
            pose=Pose()
        )
        self.odom_pose = self.tf_listener.transformPose(self.odom_frame, p)
        # store the the odometry pose in a more convenient format (x,y,theta)
        new_odom_xy_theta = convert_pose_to_xy_and_theta(self.odom_pose.pose)

        if self.particle_cloud is None:
            # now that we have all of the necessary transforms we can update
            # the particle cloud
            self.initialize_particle_cloud()
            # cache the last odometric pose so we can only update our particle
            # filter if we move more than self.d_thresh or self.a_thresh
            self.current_odom_xy_theta = new_odom_xy_theta
            # update our map to odom transform now that the particles are
            # initialized
            self.fix_map_to_odom_transform(msg)
        elif (math.fabs(new_odom_xy_theta[0] - self.current_odom_xy_theta[0]) > self.d_thresh or
              math.fabs(new_odom_xy_theta[1] - self.current_odom_xy_theta[1]) > self.d_thresh or
              math.fabs(new_odom_xy_theta[2] - self.current_odom_xy_theta[2]) > self.a_thresh):
            # we have moved far enough to do an update!
            self.update_particles_with_odom(msg)    # update based on odometry
            if self.last_projected_stable_scan:
                last_projected_scan_timeshift = deepcopy(
                    self.last_projected_stable_scan
                )

                last_projected_scan_timeshift.header.stamp = msg.header.stamp

                self.scan_in_base_link = self.tf_listener.transformPointCloud(
                    "base_link",
                    last_projected_scan_timeshift
                )

            # update based on laser scan
            self.update_particles_with_laser(msg)

            # update robot's pose
            self.update_robot_pose()

            # resample particles to focus on areas of high density
            self.resample_particles()

            # update map to odom transform now that we have new particles
            self.fix_map_to_odom_transform(msg)

        # publish particles (so things like rviz can see them)
        self.publish_particles(msg)

    def fix_map_to_odom_transform(self, msg):
        """ This method constantly updates the offset of the map and odometry
            coordinate systems based on the latest results from the localizer
        """
        translation, rotation = convert_pose_inverse_transform(self.robot_pose)
        p = PoseStamped(
            pose=convert_translation_rotation_to_pose(translation, rotation),
            header=Header(stamp=msg.header.stamp, frame_id=self.base_frame)
        )
        self.tf_listener.waitForTransform(
            self.base_frame,
            self.odom_frame,
            msg.header.stamp,
            rospy.Duration(1.0)
        )
        self.odom_to_map = self.tf_listener.transformPose(self.odom_frame, p)
        self.translation, self.rotation = convert_pose_inverse_transform(
            self.odom_to_map.pose
        )

    def broadcast_last_transform(self):
        """ Make sure that we are always broadcasting the last map
            to odom transformation.  This is necessary so things like
            move_base can work properly. """
        if not (hasattr(self, 'translation') and hasattr(self, 'rotation')):
            return

        self.tf_broadcaster.sendTransform(
            self.translation,
            self.rotation,
            rospy.get_rostime(),
            self.odom_frame,
            self.map_frame
        )

if __name__ == '__main__':
    n = ParticleFilter()
    r = rospy.Rate(5)

    while not(rospy.is_shutdown()):
        # in the main loop all we do is continuously broadcast the latest map
        # to odom transform
        n.broadcast_last_transform()
        r.sleep()
