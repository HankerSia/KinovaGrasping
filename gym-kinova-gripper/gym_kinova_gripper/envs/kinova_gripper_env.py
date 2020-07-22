#!/usr/bin/env python3

###############
# Author: Yi Herng Ong
# Purpose: Kinova 3-fingered gripper in mujoco environment
# Summer 2019

###############


import gym
from gym import utils, spaces
from gym.utils import seeding
# from gym.envs.mujoco import mujoco_env
import numpy as np
from mujoco_py import MjViewer, load_model_from_path, MjSim
# from PID_Kinova_MJ import *
import math
import matplotlib.pyplot as plt
import time
import os, sys
from scipy.spatial.transform import Rotation as R
import random
import pickle
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import xml.etree.ElementTree as ET
import copy
from classifier_network import LinearNetwork
import csv
import pandas as pd
# resolve cv2 issue
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# frame skip = 20
# action update time = 0.002 * 20 = 0.04
# total run time = 40 (n_steps) * 0.04 (action update time) = 1.6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
anj = False
#obj_freq_dict = {}

class KinovaGripper_Env(gym.Env):
	metadata = {'render.modes': ['human']}
	def __init__(self, arm_or_end_effector="hand", frame_skip=4):
		self.file_dir = os.path.dirname(os.path.realpath(__file__))
		if arm_or_end_effector == "arm":
			self._model = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300.xml")
			full_path = self.file_dir + "/kinova_description/j2s7s300.xml"
		elif arm_or_end_effector == "hand":
			pass
			self._model = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector.xml")
		else:
			print("CHOOSE EITHER HAND OR ARM")
			raise ValueError

		self._sim = MjSim(self._model)

		self._viewer = None
		# self.viewer = None

		### STEPH REMOVE added for histogram data
		self.obj_freqe = np.zeros(6)
		self.obj_freqn = np.zeros(6)
		self.obj_freq = np.zeros(6)
		self.obj_freq_dict = {}
		#self.reset_obj_freq_dict()
		self.obj_coords = [0,0,0]
		self.objects = {}
		self.obj_keys = list()

		# STEPH CHANGE TO BE ARGUMENT
		self.exp_num = 11
		self.stage_num = 1

		##### Indicate object size (Nigel, data collection only) ######
		self.obj_size = "b"
		self.Grasp_Reward=False
		self._timestep = self._sim.model.opt.timestep
		self._torque = [0,0,0,0]
		self._velocity = [0,0,0,0]

		self._jointAngle = [0,0,0,0]
		self._positions = [] # ??
		self._numSteps = 0
		self._simulator = "Mujoco"
		self.action_scale = 0.0333
		self.max_episode_steps = 50
		# Parameters for cost function
		self.state_des = 0.20
		self.initial_state = np.array([0.0, 0.0, 0.0, 0.0])
		self.frame_skip = frame_skip
		self.all_states = None
		self.action_space = spaces.Box(low=np.array([-0.8, -0.8, -0.8, -0.8]), high=np.array([0.8, 0.8, 0.8, 0.8]), dtype=np.float32) # Velocity action space
		# self.action_space = spaces.Box(low=np.array([-0.3, -0.3, -0.3, -0.3]), high=np.array([0.3, 0.3, 0.3, 0.3]), dtype=np.float32) # Velocity action space
		# self.action_space = spaces.Box(low=np.array([-1.5, -1.5, -1.5, -1.5]), high=np.array([1.5, 1.5, 1.5, 1.5]), dtype=np.float32) # Position action space
		self.state_rep = "global" # change accordingly
		# self.action_space = spaces.Box(low=np.array([-0.2]), high=np.array([0.2]), dtype=np.float32)
		# self.action_space = spaces.Box(low=np.array([-0.8, -0.8, -0.8]), high=np.array([0.8, 0.8, 0.8]), dtype=np.float32)

		min_hand_xyz = [-0.1, -0.1, 0.0, -0.1, -0.1, 0.0, -0.1, -0.1, 0.0,-0.1, -0.1, 0.0, -0.1, -0.1, 0.0,-0.1, -0.1, 0.0, -0.1, -0.1, 0.0]
		min_obj_xyz = [-0.1, -0.01, 0.0]
		min_joint_states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		min_obj_size = [0.0, 0.0, 0.0]
		min_finger_obj_dist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		min_obj_dot_prod = [0.0]
		min_f_dot_prod = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

		max_hand_xyz = [0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1, 0.5,0.1, 0.1, 0.5, 0.1, 0.1, 0.5,0.1, 0.1, 0.5, 0.1, 0.1, 0.5]
		max_obj_xyz = [0.1, 0.7, 0.5]
		max_joint_states = [0.2, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
		max_obj_size = [0.5, 0.5, 0.5]
		max_finger_obj_dist = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
		max_obj_dot_prod = [1.0]
		max_f_dot_prod = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

		#obj_freq_dict = {}

		# print()
		if self.state_rep == "global" or self.state_rep == "local":

			obs_min = min_hand_xyz + min_obj_xyz + min_joint_states + min_obj_size + min_finger_obj_dist + min_obj_dot_prod #+ min_f_dot_prod
			obs_min = np.array(obs_min)
			# print(len(obs_min))

			obs_max = max_hand_xyz + max_obj_xyz + max_joint_states + max_obj_size + max_finger_obj_dist + max_obj_dot_prod #+ max_f_dot_prod
			obs_max = np.array(obs_max)
			# print(len(obs_max))

			self.observation_space = spaces.Box(low=obs_min , high=obs_max, dtype=np.float32)
		elif self.state_rep == "metric":
			obs_min = list(np.zeros(17)) + [-0.1, -0.1, 0.0] + min_obj_xyz + min_joint_states + min_obj_size + min_finger_obj_dist + min_dot_prod
			obs_max = list(np.full(17, np.inf)) + [0.1, 0.1, 0.5] + max_obj_xyz + max_joint_states + max_obj_size + max_finger_obj_dist + max_dot_prod
			self.observation_space = spaces.Box(low=np.array(obs_min) , high=np.array(obs_max), dtype=np.float32)

		elif self.state_rep == "joint_states":
			obs_min = min_joint_states + min_obj_xyz + min_obj_size + min_dot_prod
			obs_max = max_joint_states + max_obj_xyz + max_obj_size + max_dot_prod
			self.observation_space = spaces.Box(low=np.array(obs_min) , high=np.array(obs_max), dtype=np.float32)

		self.Grasp_net = LinearNetwork().to(device)
		#trained_model = "/home/graspinglab/NCS_data/trained_model_01_23_20_0111.pt"
		trained_model = "/scratch/hugheste/NCSGen_7_7_2020/gym-kinova-gripper/trained_model_01_23_20_0111.pt"
		#trained_model = "/Users/vanil/5_3_20_Heatmap/NCSGen_7_7_2020_Cone1S/gym-kinova-gripper/trained_model_01_23_20_0111.pt"
		#trained_model = "/home/graspinglab/NCS_data/trained_model_01_23_20_2052local.pt"
		#trained_model = "/home/graspinglab/NCS_data/data_cube_9_grasp_classifier_10_17_19_1734.pt"
		# self.Grasp_net = GraspValid_net(54).to(device)
		# trained_model = "/home/graspinglab/NCS_data/ExpertTrainedNet_01_04_20_0250.pt"
		model = torch.load(trained_model)
		self.Grasp_net.load_state_dict(model)
		self.Grasp_net.eval()


	# get 3D transformation matrix of each joint
	def _get_trans_mat(self, joint_geom_name):
		finger_joints = joint_geom_name
		finger_pose = []
		empty = np.array([0,0,0,1])
		for each_joint in finger_joints:
			arr = []
			for axis in range(3):
				temp = np.append(self._sim.data.get_geom_xmat(each_joint)[axis], self._sim.data.get_geom_xpos(each_joint)[axis])
				arr.append(temp)
			arr.append(empty)
			arr = np.array(arr)
			finger_pose.append(arr)
		return np.array(finger_pose)


	def _get_local_pose(self, mat):
		rot_mat = []
		trans = []
		# print(mat)
		for i in range(3):
			orient_temp = []

			for j in range(4):
				if j != 3:
					orient_temp.append(mat[i][j])
				elif j == 3:
					trans.append(mat[i][j])
			rot_mat.append(orient_temp)
		pose = list(trans)
		# pdb.set_trace()
		return pose

	def _get_joint_states(self):
		arr = []
		for i in range(7):
			arr.append(self._sim.data.sensordata[i])

		return arr # it is a list

	# return global or local transformation matrix
	def _get_obs(self, state_rep=None):
		if state_rep == None:
			state_rep = self.state_rep

		range_data = self._get_rangefinder_data()
		# states rep
		obj_pose = self._get_obj_pose()
		obj_dot_prod = self._get_dot_product(obj_pose)
		wrist_pose  = self._sim.data.get_geom_xpos("palm")
		joint_states = self._get_joint_states()
		obj_size = self._sim.model.geom_size[-1]
		finger_obj_dist = self._get_finger_obj_dist()

		palm = self._get_trans_mat(["palm"])[0]
		# for global
		finger_joints = ["f1_prox", "f2_prox", "f3_prox", "f1_dist", "f2_dist", "f3_dist"]
		# for inverse
		finger_joints_transmat = self._get_trans_mat(["f1_prox", "f2_prox", "f3_prox", "f1_dist", "f2_dist", "f3_dist"])
		fingers_6D_pose = []
		if state_rep == "global":
			for joint in finger_joints:
				trans = self._sim.data.get_geom_xpos(joint)
				trans = list(trans)
				for i in range(3):
					fingers_6D_pose.append(trans[i])
			fingers_dot_prod = self._get_fingers_dot_product(fingers_6D_pose)
			fingers_6D_pose = fingers_6D_pose + list(wrist_pose) + list(obj_pose) + joint_states + [obj_size[0], obj_size[1], obj_size[2]*2] + finger_obj_dist + [obj_dot_prod] + fingers_dot_prod + range_data

		elif state_rep == "local":
			finger_joints_local = []
			palm_inverse = np.linalg.inv(palm)
			for joint in range(len(finger_joints_transmat)):
				joint_in_local_frame = np.matmul(finger_joints_transmat[joint], palm_inverse)
				pose = self._get_local_pose(joint_in_local_frame)
				for i in range(3):
					fingers_6D_pose.append(pose[i])
			fingers_dot_prod = self._get_fingers_dot_product(fingers_6D_pose)
			fingers_6D_pose = fingers_6D_pose + list(wrist_pose) + list(obj_pose) + joint_states + [obj_size[0], obj_size[1], obj_size[2]*2] + finger_obj_dist + [obj_dot_prod] + fingers_dot_prod + range_data

		elif state_rep == "metric":
			fingers_6D_pose = self._get_rangefinder_data()
			fingers_6D_pose = fingers_6D_pose + list(wrist_pose) + list(obj_pose) + joint_states + [obj_size[0], obj_size[1], obj_size[2]*2] + finger_obj_dist + [obj_dot_prod] #+ fingers_dot_prod

		elif state_rep == "joint_states":
			fingers_6D_pose = joint_states + list(obj_pose) + [obj_size[0], obj_size[1], obj_size[2]*2] + [obj_dot_prod] #+ fingers_dot_prod

		# print(joint_states[0:4])
		return fingers_6D_pose

	def _get_finger_obj_dist(self):
		# finger_joints = ["palm", "f1_prox", "f2_prox", "f3_prox", "f1_dist", "f2_dist", "f3_dist"]
		finger_joints = ["palm_1", "f1_prox","f1_prox_1", "f2_prox", "f2_prox_1", "f3_prox", "f3_prox_1", "f1_dist", "f1_dist_1", "f2_dist", "f2_dist_1", "f3_dist", "f3_dist_1"]

		obj = self._get_obj_pose()
		dists = []
		for i in finger_joints:
			pos = self._sim.data.get_site_xpos(i)
			dist = np.absolute(pos[0:2] - obj[0:2])
			dist[0] -= 0.0175
			temp = np.linalg.norm(dist)
			dists.append(temp)
			# pdb.set_trace()
		return dists

	# get range data from 1 step of time
	# Uncertainty: rangefinder could only detect distance to the nearest geom, therefore it could detect geom that is not object
	def _get_rangefinder_data(self):
		range_data = []
		for i in range(17):
			range_data.append(self._sim.data.sensordata[i+7])

		return range_data

	def _get_obj_pose(self):
		arr = self._sim.data.get_geom_xpos("object")
		return arr

	def _get_fingers_dot_product(self, fingers_6D_pose):
		fingers_dot_product = []
		for i in range(6):
			fingers_dot_product.append(self._get_dot_product(fingers_6D_pose[3*i:3*i+3]))
		return fingers_dot_product

	# Function to return dot product based on object location
	def _get_dot_product(self, obj_state):
		# obj_state = self._get_obj_pose()
		hand_pose = self._sim.data.get_body_xpos("j2s7s300_link_7")
		obj_state_x = abs(obj_state[0] - hand_pose[0])
		obj_state_y = abs(obj_state[1] - hand_pose[1])
		obj_vec = np.array([obj_state_x, obj_state_y])
		obj_vec_norm = np.linalg.norm(obj_vec)
		obj_unit_vec = obj_vec / obj_vec_norm

		center_x = abs(0.0 - hand_pose[0])
		center_y = abs(0.0 - hand_pose[1])
		center_vec = np.array([center_x, center_y])
		center_vec_norm = np.linalg.norm(center_vec)
		center_unit_vec = center_vec / center_vec_norm
		dot_prod = np.dot(obj_unit_vec, center_unit_vec)
		return dot_prod**20 # cuspy to get distinct reward

	def _get_reward_DataCollection(self):
		obj_target = 0.2
		obs = self._get_obs(state_rep="global")
		if abs(obs[23] - obj_target) < 0.005 or (obs[23] >= obj_target):
			lift_reward = 1
			done = True
		else:
			lift_reward = 0
			done = False
		return lift_reward, {}, done

	'''
	Reward function (Actual)
	'''
	def _get_reward(self):

		# object height target
		obj_target = 0.2

		#Grasp reward
		grasp_reward = 0.0
		obs = self._get_obs(state_rep="global")
		network_inputs=obs[0:5]
		network_inputs=np.append(network_inputs,obs[6:23])
		network_inputs=np.append(network_inputs,obs[24:])
		inputs = torch.FloatTensor(np.array(network_inputs)).to(device)
		#if np.max(np.array(obs[41:47])) < 0.035 or np.max(np.array(obs[35:41])) < 0.015:
		#	outputs = self.Grasp_net(inputs).cpu().data.numpy().flatten()
		#	if (outputs >= 0.3) & (not self.Grasp_Reward):
		#		grasp_reward = 5.0
		#		self.Grasp_Reward=True
		#	else:
		#		grasp_reward = 0.0
		#grasp_reward = outputs

		if abs(obs[23] - obj_target) < 0.005 or (obs[23] >= obj_target):
			lift_reward = 50.0
			done = True
		else:
			lift_reward = 0.0
			done = False

		finger_reward = -np.sum((np.array(obs[41:47])) + (np.array(obs[35:41])))

		reward = 0.2*finger_reward + lift_reward #+ grasp_reward

		info = {"lift_reward":lift_reward}

		return reward, info, done

	# only set proximal joints, cuz this is an underactuated hand
	def _set_state(self, states):
		self._sim.data.qpos[0] = states[0]
		self._sim.data.qpos[1] = states[1]
		self._sim.data.qpos[3] = states[2]
		self._sim.data.qpos[5] = states[3]
		self._sim.data.set_joint_qpos("object", [states[4], states[5], states[6], 1.0, 0.0, 0.0, 0.0])
		self._sim.forward()

	def _get_obj_size(self):
		return self._sim.model.geom_size[-1]

	# STEPH - Yi's old code
	def set_obj_size(self, default = False):
		hand_param = {}
		hand_param["span"] = 0.15
		hand_param["depth"] = 0.08
		hand_param["height"] = 0.15 # including distance between table and hand

		geom_types = ["box", "cylinder"]#, "sphere"]
		geom_sizes = ["s", "m", "b"]

		geom_type = random.choice(geom_types)
		geom_size = random.choice(geom_sizes)

		# Cube w: 0.1, 0.2, 0.3
		# Cylinder w: 0.1, 0.2, 0.3
		# Sphere w: 0.1, 0.2, 0.3

		# Cube & Cylinder
		width_max = hand_param["span"] * 0.3333 # 5 cm
		width_mid = hand_param["span"] * 0.2833 # 4.25 cm
		width_min = hand_param["span"] * 0.2333 # 3.5 cm
		width_choice = np.array([width_min, width_mid, width_max])

		height_max = hand_param["height"] * 0.80 # 0.12
		height_mid = hand_param["height"] * 0.73333 # 0.11
		height_min = hand_param["height"] * 0.66667 # 0.10
		height_choice = np.array([height_min, height_mid, height_max])

		# Sphere
		# radius_max = hand_param["span"] * 0.
		# radius_mid = hand_param["span"] * 0.2833
		# radius_min = hand_param["span"] * 0.2333
		# radius_choice = np.array([radius_min, radius_mid, radius_max])

		if default:
			# print("here")
			return "box", np.array([width_choice[1]/2.0, width_choice[1]/2.0, height_choice[1]/2.0])
		else:

			if geom_type == "box": #or geom_type == "cylinder":
				if geom_size == "s":
					geom_dim = np.array([width_choice[0] / 2.0, width_choice[0] / 2.0, height_choice[0] / 2.0])
				if geom_size == "m":
					geom_dim = np.array([width_choice[1] / 2.0, width_choice[1] / 2.0, height_choice[1] / 2.0])
				if geom_size == "b":
					geom_dim = np.array([width_choice[2] / 2.0, width_choice[2] / 2.0, height_choice[2] / 2.0])
			if geom_type == "cylinder":
				if geom_size == "s":
					geom_dim = np.array([width_choice[0] / 2.0, height_choice[0] / 2.0])
				if geom_size == "m":
					geom_dim = np.array([width_choice[1] / 2.0, height_choice[1] / 2.0])
				if geom_size == "b":
					geom_dim = np.array([width_choice[2] / 2.0, height_choice[2] / 2.0])

			return geom_type, geom_dim, geom_size

	# STEPH - potentially remove, Yi's code
	def gen_new_obj(self, default = False):
		file_dir = "./gym_kinova_gripper/envs/kinova_description"
		filename = "/objects.xml"
		tree = ET.parse(file_dir + filename)
		root = tree.getroot()
		d = default
		next_root = root.find("body")
		# print(next_root)
		# pick a shape and size
		geom_type, geom_dim, geom_size = self.set_obj_size(default = d)
		# if geom_type == "sphere":
		# 	next_root.find("geom").attrib["size"] = "{}".format(geom_dim[0])
		if geom_type == "box":
			next_root.find("geom").attrib["size"] = "{} {} {}".format(geom_dim[0], geom_dim[1], geom_dim[2])
		if geom_type == "cylinder":
			next_root.find("geom").attrib["size"] = "{} {}".format(geom_dim[0], geom_dim[1])

		next_root.find("geom").attrib["type"] = geom_type
		tree.write(file_dir + "/objects.xml")

		return geom_type, geom_dim, geom_size

	def render_img(self, w, h, cam_name, mode='human'):
		#if self._viewer is None:
		#	self._viewer = MjViewer(self._sim)
		#self._viewer.render()
		#return MjSim(self._model).render(width=w, height=h, camera_name=cam_name, depth=True)
		self._sim = MjSim(self._model)
		return self._sim.render(width=800, height=800,camera_name="camera")

	def str_mj_arr(arr):
		return ' '.join(['%0.3f' % arr[i] for i in range(arr._length_)])

	# STEPH - REMOVE
	def check_range(self,x,y,xmin,xmax,margin):
	  # Point to be checked
	  point = [x,y]

	  # Get hand position info
	  obs =  [0.05190709651841579, 0.05922107376362523, 0.07247797768513309, -0.047800791389975575, 0.05905594293043447, 0.09147749870602867, -0.047842290951250165, 0.05909756861129018, 0.039205400315928014, 0.07998453867584633, 0.03696301651652619, 0.07845583096202666, -0.07601887636517492, 0.036798442851176685, 0.09675404785043466, -0.07606888985873166, 0.03684846417454405, 0.033938991084013805, -0.004143432463175748, 0.1190549719678052, 0.06538203183550335, 0.02244741979521349, 0.012131665631547115, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0175, 0.0175, 0.1, 0.05806224895818543, 0.03800543634144927, 0.041675726324202376, 0.06554336298012287, 0.06282502277326353, 0.06560188337419393, 0.06288158586217023, 0.05187632119490675, 0.046141940679937744, 0.09233360299381589, 0.08553589277629997, 0.09239262808506454, 0.08559575724063456, 0.8375858053261143]

	  # State 0 - 2 : finger 1 proximal joint XYZ pose
	  prox_1 = [obs[0],obs[1]]

	  # State 3 - 5 : finger 2 Proximal joint XYZ pose
	  prox_2 = [obs[3],obs[4]]

	  #State 6 - 8 : finger 3 proximal joint XYZ pose
	  prox_3 = [obs[6],obs[7]]

	  #State 9 - 11 : finger 1 distal joint XYZ pose
	  distal_1 = [obs[9],obs[10]]

	  #State 12 - 14 : finger 2 distal joint XYZ pose
	  distal_2 = [obs[12],obs[13]]

	  #State 15 - 17 : finger 3 distal joint XYZ pose
	  distal_3 = [obs[15],obs[16]]

	  #State 18 - 20 : wrist joint XYZ pose
	  #wrist_joint = [obs[18],obs[19]]
	  palm_center = [0,0.065]

	  #State 21 - 23 : object pose XYZ
	  object_pose = [obs[21],obs[22]]

	  # Determine 'ranges' by tracing for hand span lines
	  slope1 = (prox_1[1]-distal_1[1])/(prox_1[0]-distal_1[0])
	  b1 = -(slope1*prox_1[0] - prox_1[1]) - margin
	  range1 = slope1*point[0]+b1

	  slope2 = (prox_1[1]-palm_center[1])/(prox_1[0]-palm_center[0])
	  b2 = -(slope2*prox_1[0] - prox_1[1]) - margin
	  range2 = slope2*point[0]+b2

	  slope3 = (palm_center[1]-prox_2[1])/(palm_center[0]-prox_2[0])
	  b3 = -(slope3*prox_2[0] - prox_2[1]) - margin
	  range3 = slope3*point[0]+b3

	  slope4 = (prox_2[1]-distal_2[1])/(prox_2[0]-distal_2[0])
	  b4 = -(slope4*prox_2[0] - prox_2[1]) - margin
	  range4 = slope4*point[0]+b4

	  within_hand = False
	  if point[0] >= xmin and point[0] <= xmax:
		  if point[0] >= prox_1[0] and point[1] < range1:
			  #print("range1: point is within range")
			  within_hand = True
		  elif point[0] >= palm_center[0] and point[0] < prox_1[0] and point[1] < range2:
			  #print("range2: point is within range")
			  within_hand = True
		  elif point[0] < palm_center[0] and point[0] >= prox_2[0] and point[1] < range3:
			  #print("range3: point is within range")
			  within_hand = True
		  elif point[0] < prox_2[0] and point[1] < range4:
			  #print("range4: point is within range")
			  within_hand = True
		  else:
			  #print("range check: point is NOT within range")
			  within_hand = False
	  else:
		  #print("point is NOT within x range")
		  within_hand = False

	  #z = self._get_obj_size()[-1]
	  #print("z: ",z)
	  contact_check = self.object_contact(point[0],point[1])
	  if within_hand == True and contact_check == False:
		  return True
	  elif within_hand == True and contact_check == True:
		  return False
	  else:
		  return False

	# STEPH ReMOVE
	def object_contact(self,x,y):
		#"f1_prox", "f2_prox", "f3_prox", "f1_dist", "f2_dist", "f3_dist"
		#print("Prev state render: ")
		#self.render()

		#self.all_states = np.array([0.0, 0.0, 0.0, 0.0, x, y, z])
		#self._set_state(self.all_states)

		#self._sim = MjSim(self._model)

		#print("New state render: ")
		#self.render()


		contacts = tuple()
		for coni in range(self._sim.data.ncon):
			con = self._sim.data.contact[coni]
			contacts += ((con.geom1, con.geom2), )

		print("You're in object contact")
		d = self._sim.data
		for coni in range(d.ncon):
			print('  Contact %d:' % (coni,))
			con = d.obj.contact[coni]
			print('    dist     = %0.3f' % (con.dist,))
			print('    pos      = %s' % (str_mj_arr(con.pos),))
			print('    frame    = %s' % (str_mj_arr(con.frame),))
			print('    friction = %s' % (str_mj_arr(con.friction),))
			print('    dim      = %d' % (con.dim,))
			print('    geom1    = %d' % (con.geom1,))
			print('    geom2    = %d' % (con.geom2,))

		print()
		print("just printed stuff")

		for i in range(len(contacts)):
			print("These are touching: ")
			print("self._sim.model.geom_id2name(",contacts[i][0],"): ",self._sim.model.geom_id2name(contacts[i][0]))
			print("self._sim.model.geom_id2name(",contacts[i][1],"): ",self._sim.model.geom_id2name(contacts[i][1]))
			print()

		if len(contacts) > 0:
			print("***This would be rejected")
			self.render()
			return False
		else:
			print("This is accpeted")
			return False

	# Get the initial object position
	def sample_initial_valid_object_pos(self,shapeName):
		filename = "gym_kinova_gripper/envs/kinova_description/shape_coords/" + shapeName + ".txt"
		with open(filename) as csvfile:
			data = [(float(x), float(y), float(z)) for x, y, z in csv.reader(csvfile, delimiter= ' ')]

		rand_coord = random.choice(data)
		x = rand_coord[0]
		y = rand_coord[1]
		z = rand_coord[2]

		# self.set_obj_freq(size,shape,'normal',full_shape)

		return x, y, z

	# STEPH - REMOVE
	def randomize_initial_pose(self, collect_data, size):
		# geom_type, geom_dim, geom_size = self.gen_new_obj()
		# geom_size = "s"
		geom_size = size

		# self._model = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1.xml")
		# self._sim = MjSim(self._model)

		if geom_size == "s":
			if not collect_data:
				x = [0.05, 0.04, 0.03, 0.02, -0.05, -0.04, -0.03, -0.02]
				y = [0.0, 0.02, 0.03, 0.04]
				rand_x = random.choice(x)
				rand_y = 0.0
				if rand_x == 0.05 or rand_x == -0.05:
					rand_y = 0.0
				elif rand_x == 0.04 or rand_x == -0.04:
					rand_y = random.uniform(0.0, 0.02)
				elif rand_x == 0.03 or rand_x == -0.03:
					rand_y = random.uniform(0.0, 0.03)
				elif rand_x == 0.02 or rand_x == -0.02:
					rand_y = random.uniform(0.0, 0.04)
			else:
				x = [0.04, 0.03, 0.02, -0.04, -0.03, -0.02]
				y = [0.0, 0.02, 0.03, 0.04]
				rand_x = random.choice(x)
				rand_y = 0.0
				if rand_x == 0.04 or rand_x == -0.04:
					rand_y = random.uniform(0.0, 0.02)
				elif rand_x == 0.03 or rand_x == -0.03:
					rand_y = random.uniform(0.0, 0.03)
				elif rand_x == 0.02 or rand_x == -0.02:
					rand_y = random.uniform(0.0, 0.04)
		if geom_size == "m":
			x = [0.04, 0.03, 0.02, -0.04, -0.03, -0.02]
			y = [0.0, 0.02, 0.03]
			rand_x = random.choice(x)
			rand_y = 0.0
			if rand_x == 0.04 or rand_x == -0.04:
				rand_y = 0.0
			elif rand_x == 0.03 or rand_x == -0.03:
				rand_y = random.uniform(0.0, 0.02)
			elif rand_x == 0.02 or rand_x == -0.02:
				rand_y = random.uniform(0.0, 0.03)
		if geom_size == "b":
			x = [0.03, 0.02, -0.03, -0.02]
			y = [0.0, 0.02]
			rand_x = random.choice(x)
			rand_y = 0.0
			if rand_x == 0.03 or rand_x == -0.03:
				rand_y = 0.0
			elif rand_x == 0.02 or rand_x == -0.02:
				rand_y = random.uniform(0.0, 0.02)
		# return rand_x, rand_y, geom_dim[-1]
		# print(rand_x, rand_y)
		return rand_x, rand_y

		# medium x = [0.04, 0.03, 0.02]
		# med y = [0.0, 0.02, 0.03]
		# large x = [0.03, 0.02]
		# large y = [0.0, 0.02]

	# UPDATE AS A GROUP
	def experiment(self, exp_num, stage_num, test=False):
		objects = {}

		if not test:

			# ------ Experiment 1 ------- #
			if exp_num == 1:

				# Exp 1 Stage 1: Change size --->
				if stage_num == 1:
					objects["sbox"] = "/kinova_description/j2s7s300_end_effector.xml"
					objects["bbox"] = "/kinova_description/j2s7s300_end_effector_v1_bbox.xml"

					# Testing Exp 1 Stage 1

				# Exp 1 Stage 2: Change shape
				if stage_num == 2:
					objects["sbox"] = "/kinova_description/j2s7s300_end_effector.xml"
					objects["bbox"] = "/kinova_description/j2s7s300_end_effector_v1_bbox.xml"
					objects["scyl"] = "/kinova_description/j2s7s300_end_effector_v1_scyl.xml"
					objects["bcyl"] = "/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"

			# ------ Experiment 2 ------- #
			elif exp_num == 2:

				# Exp 2 Stage 1: Change shape
				if stage_num == 1:
					objects["sbox"] = "/kinova_description/j2s7s300_end_effector.xml"
					objects["scyl"] = "/kinova_description/j2s7s300_end_effector_v1_scyl.xml"

				# Exp 2 Stage 2: Change size
				if stage_num == 2:
					objects["sbox"] = "/kinova_description/j2s7s300_end_effector.xml"
					objects["scyl"] = "/kinova_description/j2s7s300_end_effector_v1_scyl.xml"
					objects["bbox"] = "/kinova_description/j2s7s300_end_effector_v1_bbox.xml"
					objects["bcyl"] = "/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"

				# Testing Exp 2
				# objects["mbox"] = "/kinova_description/j2s7s300_end_effector_v1_mbox.xml"
				# objects["mcyl"] = "/kinova_description/j2s7s300_end_effector_v1_mcyl.xml"
				# objects["bbox"] = "/kinova_description/j2s7s300_end_effector_v1_bbox.xml"
				# objects["bcyl"] = "/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"
				# objects["sbox"] = "/kinova_description/j2s7s300_end_effector.xml"
				# objects["scyl"] = "/kinova_description/j2s7s300_end_effector_v1_scyl.xml"
			# ------ Experiment 3 ------ #
			elif exp_num == 3:
				# Mix all
				objects["sbox"] = "/kinova_description/j2s7s300_end_effector.xml"
				objects["bbox"] = "/kinova_description/j2s7s300_end_effector_v1_bbox.xml"
				objects["scyl"] = "/kinova_description/j2s7s300_end_effector_v1_scyl.xml"
				objects["bcyl"] = "/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"

			elif exp_num == 4:
				objects["mcon20"] = "/kinova_description/j2s7s300_end_effector_mcone20.xml"
				objects["mcon30"] = "/kinova_description/j2s7s300_end_effector_mcone30.xml"

			elif exp_num == 5:
				objects["svase"] = "/kinova_description/j2s7s300_end_effector_v1_svase.xml"
				objects["mvase"] = "/kinova_description/j2s7s300_end_effector_v1_mvase.xml"
				objects["bvase"] = "/kinova_description/j2s7s300_end_effector_v1_bvase.xml"

			elif exp_num == 6:
				objects["sbox"] = "/kinova_description/j2s7s300_end_effector.xml"
				objects["scyl"] = "/kinova_description/j2s7s300_end_effector_v1_scyl.xml"
				objects["mbox_r45"] = "/kinova_description/j2s7s300_end_effector_mvase9_10.xml"
				objects["mcyl_side"] = "/kinova_description/j2s7s300_end_effector_mcyl_side.xml"

			elif exp_num == 7:
				objects["mvase"] = "/kinova_description/j2s7s300_end_effector_v1_mhg.xml"

			elif exp_num == 8:
				objects["sbox"] = "/kinova_description/j2s7s300_end_effector.xml"
				objects["bbox"] = "/kinova_description/j2s7s300_end_effector_v1_bbox.xml"

			elif exp_num == 9:
				objects["scyl"] = "/kinova_description/j2s7s300_end_effector_v1_scyl.xml"
				objects["bcyl"] = "/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"

			elif exp_num == 10:
				objects["svase"] = "/kinova_description/j2s7s300_end_effector_v1_svase.xml"
				objects["bvase"] = "/kinova_description/j2s7s300_end_effector_v1_bvase.xml"

			elif exp_num == 11:
				objects["CubeS"] = "/kinova_description/j2s7s300_end_effector_v1_CubeS.xml"

			elif exp_num == 12:
				objects["mbox"] = "/kinova_description/j2s7s300_end_effector_v1_mbox.xml"

			elif exp_num == 13:
				objects["bbox"] = "/kinova_description/j2s7s300_end_effector_v1_bbox.xml"

			elif exp_num == 14:
				objects["CylinderS"] = "/kinova_description/j2s7s300_end_effector_v1_CylinderS.xml"

			elif exp_num == 15:
				objects["mcyl"] = "/kinova_description/j2s7s300_end_effector_v1_mcyl.xml"

			elif exp_num == 16:
				objects["bcyl"] = "/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"

			elif exp_num == 17:
				objects["Cube45S"] = "/kinova_description/j2s7s300_end_effector_v1_bvase110.xml"

			elif exp_num == 18:
				objects["Vase1S"] = "/kinova_description/j2s7s300_end_effector_v1_Vase1S.xml"

			elif exp_num == 19:
				objects["Vase2S"] = "/kinova_description/j2s7s300_end_effector_v1_Vase2S.xml"

			elif exp_num == 20:
				objects["Cone1S"] = "/kinova_description/j2s7s300_end_effector_v1_Cone1S.xml"

			elif exp_num == 21:
				objects["Cone2S"] = "/kinova_description/j2s7s300_end_effector_v1_Cone2S.xml"

			else:
				print("Enter Valid Experiment Number")
				raise ValueError
		else:
			# Test objects:
			objects["mcyl"] = "/kinova_description/j2s7s300_end_effector_v1_mcyl.xml"
			objects["mbox"] = "/kinova_description/j2s7s300_end_effector_v1_mbox.xml"

		return objects

	# STEPH LOOK INTO
	def set_obj_coords(self,x,y,z):
		self.obj_coords[0] = x
		self.obj_coords[1] = y
		self.obj_coords[2] = z

	def get_obj_coords(self):
		return self.obj_coords

	def get_obj_freqe(self):
		return self.obj_freqe

	def get_obj_freqn(self):
		return self.obj_freqn

	def get_obj_freq(self):
		return self.obj_freq

	def get_obj_freq_dict(self):
		return self.obj_freq_dict

	def set_obj_freq_dict(self,key,value):
		obj_freq_dict[key] = value

	#def reset_obj_freq_dict(self):
	#	#o_freq = self.get_obj_freq()
	#	obj_freq_dict["sbox"] = 0
	#	obj_freq_dict["scyl"] = 0
	#	obj_freq_dict["mbox"] = 0
	#	obj_freq_dict["mcyl"] = 0
	#	obj_freq_dict["bbox"] = 0
	#	obj_freq_dict["bcyl"] = 0

	## STEPH UPDATE this ###
	def set_obj_freq(self, size, shape, edge_or_normal,full_shape):
		if edge_or_normal == "normal":
			if size == 's':
				if shape == "box":
					self.obj_freqn[0] += 1
				elif shape == "cyl":
					self.obj_freqn[1] += 1
			elif size == 'm':
				if shape == "box":
					self.obj_freqn[2] += 1
				elif shape == "cyl":
					self.obj_freqn[3] += 1
			elif size == 'b':
				if shape == "box":
					self.obj_freqn[4] += 1
				elif shape == "cyl":
					self.obj_freqn[5] += 1
			else:
				print("size and shape are incorrect")
				raise ValueError
		elif edge_or_normal == "edge":
			if size == 's':
				if shape == "box":
					self.obj_freqe[0] += 1
				elif shape == "cyl":
					self.obj_freqe[1] += 1
			elif size == 'm':
				if shape == "box":
					self.obj_freqe[2] += 1
				elif shape == "cyl":
					self.obj_freqe[3] += 1
			elif size == 'b':
				if shape == "box":
					self.obj_freqe[4] += 1
				elif shape == "cyl":
					self.obj_freqe[5] += 1
			else:
				print("size and shape are incorrect")
				raise ValueError
		else:
			print("Edge or Normal case is incorrect")
			raise ValueError
		self.obj_freq = np.add(self.obj_freqn,self.obj_freqe)

		full_shape = edge_or_normal[:4] + full_shape
		if full_shape in self.obj_freq_dict:
			self.obj_freq_dict[full_shape] += 1
		else:
			self.obj_freq_dict[full_shape] = 1

	def check_obj_file_empty(self,filename):
		if os.path.exists(filename) == False:
			return False
		with open(filename, 'r') as read_obj:
			# read first character
			one_char = read_obj.read(1)
			# if not fetched then file is empty
			if not one_char:
			   return True
			return False

	def Generate_Latin_Square(self,max_elements,filename):

		print("GENERATE LATIN SQUARE")

		### Choose an experiment ###
		self.objects = self.experiment(self.exp_num, self.stage_num, test=False)
		#print("*self.exp_num: ",self.exp_num)

		# n is the number of object types (sbox, bbox, bcyl, etc.)
		num_elements = 0
		elem_gen_done = 0
		printed_row = 0

		while num_elements < max_elements:
			n = len(self.objects.keys())-1
			#print("This is n: ",n)
			k = n
			# Loop to prrows
			for i in range(0, n+1, 1):
				# This loops runs only after first iteration of outer loop
				# Prints nummbers from n to k
				keys = list(self.objects.keys())
				temp = k

				while (temp <= n) :
					if printed_row <= n: # Just used to print out one row instead of all of them
						printed_row += 1

					key_name = str(keys[temp])
					self.obj_keys.append(key_name)
					temp += 1
					num_elements +=1
					if num_elements == max_elements:
						elem_gen_done = 1
						break
				if elem_gen_done:
					break

				# This loop prints numbers from 1 to k-1.
				for j in range(0, k):
					key_name = str(keys[j])
					self.obj_keys.append(key_name)
					num_elements +=1
					if num_elements == max_elements:
						elem_gen_done = 1
						break
				if elem_gen_done:
					break
				k -= 1

			w = csv.writer(open(filename, "w"))
			for key in self.obj_keys:
				w.writerow(key)

	def objects_file_to_list(self,filename, num_objects):
		df = pd.read_csv(filename)
		if (df.empty):
			"Object file is empty!"
			self.Generate_Latin_Square(num_objects,filename)
		with open(filename, newline='') as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				row = ''.join(row)
				self.obj_keys.append(row)

	def get_obj_keys(self):
		return self.obj_keys

	def get_object(self,filename):
		# Get random shape
		random_shape = self.obj_keys.pop()

		# remove current object file contents
		f = open(filename, "w")
		f.truncate()
		f.close()

		# write new object keys to file so new env will have updated list
		w = csv.writer(open(filename, "w"))
		for key in self.obj_keys:
			w.writerow(key)

		# Load model
		self._model = load_model_from_path(self.file_dir + self.objects[random_shape])
		self._sim = MjSim(self._model)

		#print("random_shape[1:4]: ",random_shape[1:4])

		print("random_shape: ",random_shape)
		x, y, z = self.sample_initial_valid_object_pos(random_shape)
		self.set_obj_coords(x,y,z)

		return x, y, z

	def reset(self,env_name):
		filename = ""
		num_objects = 200
		if env_name == "env":
			filename = "objects.csv"
			num_objects = 20000
		else:
			filename = "eval_objects.csv"
			num_objects = 200

		if len(self.objects) == 0:
			self.objects = self.experiment(self.exp_num, self.stage_num, test=False)
			#self.Generate_Latin_Square(20000,filename)
		if len(self.obj_keys) == 0:
			self.objects_file_to_list(filename,num_objects)
		#print("This is obj_keys: ",self.obj_keys)
		x, y, z = self.get_object(filename)

		self.all_states = np.array([0.0, 0.0, 0.0, 0.0, x, y, z])
		self._set_state(self.all_states)
		states = self._get_obs()
		self.t_vel = 0
		self.prev_obs = []
		self.Grasp_Reward=False
		return states

	def render(self, mode='human'):
		if self._viewer is None:
			self._viewer = MjViewer(self._sim)
		self._viewer.render()

		#print("self.sim.data.ncon: ",self._sim.data.ncon)
		#if self._viewer is None:
		#	self._viewer = MjViewer(self._sim)
		#	self._viewer._paused = True
		#	self._viewer.render()
		#else:

		# For paused version
		#self._viewer = MjViewer(self._sim)
		#self._viewer._paused = True
		#self._viewer.render()

	def close(self):
		if self._viewer is not None:
			self._viewer = None

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	###################################################
	##### ---- Action space : Joint Velocity ---- #####
	###################################################
	def step(self, action):
		total_reward = 0
		for _ in range(self.frame_skip):
			if action[0] < 0.0:
				self._sim.data.ctrl[0] = 0.0
			else:
				self._sim.data.ctrl[0] = (action[0] / 0.8) * 0.2
				# self._sim.data.ctrl[0] = action[0]

			for i in range(3):
				# vel = action[i]
				if action[i+1] < 0.0:
					self._sim.data.ctrl[i+1] = 0.0
				else:
					self._sim.data.ctrl[i+1] = action[i+1]
			self._sim.step()

		obs = self._get_obs()

		### Get this reward for RL training ###
		total_reward, info, done = self._get_reward()
		### Get this reward for data collection ###
		# total_reward, info, done = self._get_reward_DataCollection()

		# print(obs[15:18], self._get_dot_product(obs[15:18]))

		# print(self._get_dot_product)
		return obs, total_reward, done, info
	#####################################################

	###################################################
	##### ---- Action space : Joint Angle ---- ########
	###################################################
	# def step(self, action):
	# 	total_reward = 0
	# 	for _ in range(self.frame_skip):
	# 		self.pos_control(action)
	# 		self._sim.step()

	# 	obs = self._get_obs()
	# 	total_reward, info, done = self._get_reward()
	# 	self.t_vel += 1
	# 	self.prev_obs.append(obs)
	# 	# print(self._sim.data.qpos[0], self._sim.data.qpos[1], self._sim.data.qpos[3], self._sim.data.qpos[5])
	# 	return obs, total_reward, done, info

	# def pos_control(self, action):
	# 	# position
	# 	# print(action)

	# 	self._sim.data.ctrl[0] = (action[0] / 1.5) * 0.2
	# 	self._sim.data.ctrl[1] = action[1]
	# 	self._sim.data.ctrl[2] = action[2]
	# 	self._sim.data.ctrl[3] = action[3]
	# 	# velocity
	# 	if abs(action[0] - 0.0) < 0.0001:
	# 		self._sim.data.ctrl[4] = 0.0
	# 	else:
	# 		self._sim.data.ctrl[4] = 0.1
	# 		# self._sim.data.ctrl[4] = (action[0] - self.prev_action[0] / 25)

	# 	if abs(action[1] - 0.0) < 0.001:
	# 		self._sim.data.ctrl[5] = 0.0
	# 	else:
	# 		self._sim.data.ctrl[5] = 0.01069
	# 		# self._sim.data.ctrl[5] = (action[1] - self.prev_action[1] / 25)

	# 	if abs(action[2] - 0.0) < 0.001:
	# 		self._sim.data.ctrl[6] = 0.0
	# 	else:
	# 		self._sim.data.ctrl[6] = 0.01069
	# 		# self._sim.data.ctrl[6] = (action[2] - self.prev_action[2] / 25)

	# 	if abs(action[3] - 0.0) < 0.001:
	# 		self._sim.data.ctrl[7] = 0.0
	# 	else:
	# 		self._sim.data.ctrl[7] = 0.01069
	# 		# self._sim.data.ctrl[7] = (action[3] - self.prev_action[3] / 25)

		# self.prev_action = np.array([self._sim.data.qpos[0], self._sim.data.qpos[1], self._sim.data.qpos[3], self._sim.data.qpos[5]])
		# self.prev_action = np.array([self._sim.data.qpos[0], self._sim.data.qpos[1], self._sim.data.qpos[3], self._sim.data.qpos[5]])

	#####################################################


class GraspValid_net(nn.Module):
	def __init__(self, state_dim):
		super(GraspValid_net, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

	def forward(self, state):
		# pdb.set_trace()

		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a =	torch.sigmoid(self.l3(a))
		return a
