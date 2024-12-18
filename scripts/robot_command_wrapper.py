import logging
import math
import numpy as np
import trimesh
import sys
import pickle
import os

import pybullet as pb
import itertools

import zmq
import pickle
import zlib

import random
import cv2
import torch
from concepts.math.rotationlib_xyzw import mat2pos_quat
from concepts.simulator.pybullet.client import BulletClient
from tracikpy import TracIKSolver, MultiTracIKSolver
from local_config import UPC_IP, UPC_PORT

URDF_PATH =  os.path.join(os.path.dirname(__file__), '../models/rby1a/urdf/model_tracikpy.urdf')

class IKSolver_Wrapper:
    def __init__(self, ik_solver):
        self.ik_solver = ik_solver
        self.joint_names = self.ik_solver.joint_names

    def solve(self, tool_pose_mat4, seed_conf=None, pos_tolerance=1e-4, ori_tolerance=math.radians(5e-2)):
        # will use random seed_conf if seed_conf is None
        bx, by, bz = pos_tolerance * np.ones(3)
        brx, bry, brz = ori_tolerance * np.ones(3)
        conf = self.ik_solver.ik(tool_pose_mat4, qinit=seed_conf, bx=bx, by=by, bz=bz, brx=brx, bry=bry, brz=brz)
        jointname_to_conf = {name:j for name, j in zip(self.joint_names, conf)} 
        return jointname_to_conf

    def solve_multi(self, tool_pose_mat4, seed_conf=None, pos_tolerance=1e-4, ori_tolerance=math.radians(5e-2), n_result=20):
        # will use random seed_conf if seed_conf is None
        bx, by, bz = pos_tolerance * np.ones(3)
        brx, bry, brz = ori_tolerance * np.ones(3)
        is_valid_conf, confs = self.ik_solver.iks(
            np.repeat(tool_pose_mat4[np.newaxis, ...], n_result, axis=0),
            qinits=np.repeat(seed_conf[np.newaxis, ...], n_result, axis=0) if seed_conf is not None else None,
            bx=bx, by=by, bz=bz, brx=brx, bry=bry, brz=brz
        )
        all_jointname_to_conf = [{name:j for name, j in zip(self.joint_names, conf)} for (is_valid, conf) in zip(is_valid_conf, confs) if is_valid]
        return all_jointname_to_conf

    def solve_closest(self, tool_pose, seed_conf, **kwargs):
        return self.solve(tool_pose, seed_conf=seed_conf, **kwargs)

class RobotController:
    def __init__(self, urdf_path, base_link_name: str, tool_link_names: list):
        self.urdf_path = urdf_path
        self.base_link_name = base_link_name
        self.ik_solvers = {}
        for tool_link_name in tool_link_names:
            self.ik_solvers[tool_link_name] = self.create_ik_solver(tool_link_name)

    def create_ik_solver(self, tool_link_name):
        tracik_solver = self.create_tracik_solver(tool_link_name)
        ik_solver = IKSolver_Wrapper(tracik_solver)
        return ik_solver

    def create_tracik_solver(self, tool_link_name, max_time=0.025, error=1e-3):
        tracik_solver = MultiTracIKSolver(
                urdf_file=self.urdf_path, 
                base_link=self.base_link_name,
                tip_link=tool_link_name,
                timeout=max_time, 
                epsilon=error,
                solve_type='Distance'  # Speed | Distance | Manipulation1 | Manipulation2
        )  
        assert tracik_solver.joint_names
        tracik_solver.urdf_file = self.urdf_path
        return tracik_solver

    def capture_image(self, *args, **kwargs):
        raise NotImplementedError


class RubyPybulletController(RobotController):
    def __init__(self, urdf_path, base_link_name, tool_link_names, vis=True):
        super().__init__(urdf_path, base_link_name, tool_link_names)
        self.base_link_name = base_link_name
        self.bullet_client = BulletClient(is_gui=vis)
        robot_in_pb = self.bullet_client.load_urdf(urdf_path)

    def set_joint_confs_by_name(self, joint_name_to_config: dict):
        for joint_name, joint_pos_val in joint_name_to_config.items():
            self.bullet_client.world.set_qpos(joint_name, joint_pos_val)
        self.bullet_client.update_viewer_twice()

    def execute_joint_traj(self, joint_traj: list[dict]):
        for joint_name_to_config in joint_traj:
            self.set_joint_confs_by_name(joint_name_to_config)

    def get_current_joint_confs_by_name(self, joint_names):
        return self.bullet_client.world.get_batched_qpos(joint_names)

    def get_ik_solution_for_link(self, link_name, link_target_pose, seed_conf=None, return_all=False, n_proposals=20):
        if link_name not in self.ik_solvers:
            self.ik_solvers[link_name] = self.create_ik_solver(link_name)
        if seed_conf is not None:
            # TODO joint name mapping
            raise NotImplementedError
        ik_solutions = self.ik_solvers[link_name].solve_multi(link_target_pose, n_result=n_proposals)
        valid_solutions = [ik_solution for ik_solution in ik_solutions if not self.is_self_collision(ik_solution)]
        if len(valid_solutions) == 0:
            return None

        # sort according to torso changes
        valid_solutions_sorted = sorted(valid_solutions, key=lambda x: np.abs(np.array([qval for (joint_name, qval) in x.items() if 'torso' in joint_name])).sum())
        if return_all:
            return valid_solutions_sorted
        return valid_solutions_sorted[0]

    def is_self_collision(self, jointname_to_conf):
        # TODO
        return False

class RubyRealworldController(RobotController):
    def __init__(self, urdf_path, base_link_name, tool_link_names):
        super().__init__(urdf_path, base_link_name, tool_link_names)
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://{UPC_IP}:{UPC_PORT}")
        self.socket = socket
        self.camera_intrinsics = None
        self.image_dim = None
        self.capture_rs = None

    def record_body_readings_for_Nsec(self, timeout=10):
        self.socket.send(zlib.compress(pickle.dumps({"message_name": "record_body_readings_for_Nsec",
                                                     'timeout': timeout
                                                     })))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message['recorded_state']


    def replay_traj(self, recorded_traj):
        self.socket.send(zlib.compress(pickle.dumps({"message_name": "replay_traj",
                                                     'recorded_state': recorded_traj
                                                })))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

def initialize_robot_controller(robot_urdf_path, base_link_name, tool_link_names, debug=False):
    robot_pybullet_controller = RubyPybulletController(
            urdf_path=robot_urdf_path,
            base_link_name=base_link_name,
            tool_link_names=tool_link_names,
            # vis=False
    )
    if debug:
        robot_realworld_controller = None
    else:
        robot_realworld_controller = RubyRealworldController(
            urdf_path=robot_urdf_path,
            base_link_name=base_link_name,
            tool_link_names=tool_link_names
    )
    return robot_pybullet_controller, robot_realworld_controller
    #########################

def create_pybullet_box(pb_client:BulletClient, size:list, pose_mat4:np.ndarray, color:list):
    width, length, height = size
    pos, quat = mat2pos_quat(pose_mat4)
    box_visual_info = {
        'shapeType': pb.GEOM_BOX,
        'halfExtents': [width / 2.0, length / 2.0, height / 2.0],
        'rgbaColor': color if len(color) == 4 else color + [1],
        'visualFramePosition': pos,
        'visualFrameOrientation': quat,
        'physicsClientId': pb_client.client_id
    }
    visualShapeId = pb.createVisualShape(**box_visual_info)
    box_collision_info = {
        'shapeType': pb.GEOM_BOX,
        'halfExtents': [width / 2.0, length / 2.0, height / 2.0],
        'collisionFramePosition': pos,
        'collisionFrameOrientation': quat,
        'physicsClientId': pb_client.client_id
    }
    collisionShapeId = pb.createCollisionShape(**box_collision_info)
    box_in_pb = pb.createMultiBody(baseCollisionShapeIndex=collisionShapeId, baseVisualShapeIndex=visualShapeId, physicsClientId=pb_client.client_id)
    return box_in_pb

def grasping_test():
    tool_link_names = ['ee_finger_r1', 'ee_finger_l1']
    robot_pybullet_controller, _ = initialize_robot_controller(URDF_PATH, base_link_name='base', tool_link_names=tool_link_names, debug=True)

    box_pose_mat = np.eye(4)
    box_pose_mat[:3, 3] = [.6, 0, 1.1]

    grasp_pose = np.eye(4)
    grasp_pose[:3, 3] = [0, 0, -0.1]
    tgt_pose = box_pose_mat.dot(np.linalg.inv(grasp_pose))
    tgt_point = create_pybullet_box(robot_pybullet_controller.bullet_client, size=[.02, .02, .02], pose_mat4=box_pose_mat, color=[0, 0, 1])

    for tool_link_name in tool_link_names:
        print(f'>>> ik for {tool_link_name}')
        ik_solutions = robot_pybullet_controller.get_ik_solution_for_link(tool_link_name, tgt_pose, return_all=True, n_proposals=20)
        for ik_solution in ik_solutions:
            print(ik_solution)
            robot_pybullet_controller.set_joint_confs_by_name(ik_solution)
            robot_pybullet_controller.bullet_client.wait_for_user()


if __name__=='__main__':
    grasping_test()
    # robot_policy = initialize_robot_controller(URDF_PATH, base_link_name='base', tool_link_names=['ee_right', 'ee_left'], debug=True)
    # # loaded_traj = np.load('./traj_recorded_debug.npz',allow_pickle=True)['data'][:,2:-2]
    # # assert loaded_traj.shape[1]==20
    # # # send commands via ipdb for debugging
    # import ipdb
    # ipdb.set_trace()
    # print('exit')
