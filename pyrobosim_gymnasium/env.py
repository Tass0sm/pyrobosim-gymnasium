import numpy as np
import gymnasium as gym

from unified_planning.shortcuts import UserType, BoolType, Not, Equals
from unified_planning.model import Problem, Fluent, InstantaneousAction, Object

from planning_with_constraints import (
    ConstraintEnabledInstantaneousAction,
    LambdaConstraintGenerator
)

from pyrobosim.core.robot import Robot
from pyrobosim.core.world import World
from pyrobosim.manipulation import GraspGenerator, ParallelGraspProperties
from pyrobosim.navigation.occupancy_grid import OccupancyGrid
from pyrobosim.navigation.execution import ConstantVelocityExecutor
from pyrobosim.navigation.prm import PRMPlanner
from pyrobosim.navigation.rrt import RRTPlanner
from pyrobosim.sensors.lidar import Lidar2D
from pyrobosim.utils.general import get_data_folder
from pyrobosim.utils.pose import Pose

from importlib.resources import files
import pyrobosim_gymnasium


class PyRoboGym(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
            self,
            use_lidar: bool = False,
            partial_obs_objects: bool = False,
            cmd_mode: str = "velocity",
            render_mode: str | None = None
    ):
        super().__init__()
        self._cmd_mode = cmd_mode
        self.render_mode = render_mode
        self._app = None

        self.world = World()
        data_folder = get_data_folder()

        # Set the location and object metadata
        self.world.add_metadata(
            locations=[
                data_folder / "example_location_data_furniture.yaml",
                data_folder / "example_location_data_accessories.yaml",
            ],
            objects=[
                data_folder / "example_object_data_food.yaml",
                data_folder / "example_object_data_drink.yaml",
            ],
        )

        # Add ONE large room (5x5 meter square)
        room_coords = [(-2.5, -2.5), (2.5, -2.5), (2.5, 2.5), (-2.5, 2.5)]
        self.world.add_room(
            name="main_room",
            pose=Pose(x=0.0, y=0.0, z=0.0, yaw=0.0),
            footprint=room_coords,
            color="lightgray",
        )

        # Add Locations (Tables/Desks) within the same room
        table = self.world.add_location(
            category="table",
            parent="main_room",
            pose=Pose(x=-1.5, y=-1.5, z=0.0, yaw=0.0),
        )

        desk = self.world.add_location(
            category="desk",
            parent="main_room",
            pose=Pose(x=1.5, y=1.5, z=0.0, yaw=0.0)
        )

        # Add Objects to the locations for planning tasks
        self.world.add_object(category="banana", parent=table)
        self.world.add_object(category="apple", parent=table)
        self.world.add_object(category="water", parent=desk)
        self.world.add_object(category="apple", parent=desk)

        # Add robots
        grasp_props = ParallelGraspProperties(
            max_width=0.175,
            depth=0.1,
            height=0.04,
            width_clearance=0.01,
            depth_clearance=0.01,
        )
        lidar = Lidar2D(
            update_rate_s=0.1,
            angle_units="degrees",
            min_angle=-120.0,
            max_angle=120.0,
            angular_resolution=5.0,
            max_range_m=2.0,
        )

        self.robot = Robot(
            name="robot0",
            radius=0.1,
            path_executor=ConstantVelocityExecutor(
                linear_velocity=1.0,
                dt=0.1,
                max_angular_velocity=4.0,
                validate_during_execution=True,
            ),
            sensors={"lidar": lidar} if use_lidar else None,
            grasp_generator=GraspGenerator(grasp_props),
            partial_obs_objects=partial_obs_objects,
            color="#CC00CC",
        )
        self.world.add_robot(self.robot, pose=Pose())
        planner_config_rrt = {
            "bidirectional": True,
            "rrt_connect": False,
            "rrt_star": True,
            "collision_check_step_dist": 0.025,
            "max_connection_dist": 0.5,
            "rewire_radius": 1.5,
            "compress_path": False,
        }
        rrt_planner = RRTPlanner(**planner_config_rrt)
        self.robot.set_path_planner(rrt_planner)

        self._t = 0.0
        self._dt = 0.1

    def reset(self):
        self.world.reset()
        self.robot = self.world.robots[0]
        self._t = 0.0

        x, x_dot = self._get_obs()

        return (self._t, x, x_dot), {}

    def step(self, cmd):

        self._t += self._dt

        if self._cmd_mode == "position":
            new_pose = self.robot.dynamics.pose
            new_pose.x = cmd[0]
            new_pose.y = cmd[1]
            new_pose.set_euler_angles(yaw = cmd[2])

            if self.robot.is_in_collision(pose=new_pose):
                self.robot.dynamics.velocity = np.array([0.0, 0.0, 0.0])
            else:
                self.robot.set_pose(new_pose)
        elif self._cmd_mode == "velocity":
            new_pose = self.robot.dynamics.step(cmd_vel, self._dt)

            if self.robot.is_in_collision(pose=new_pose):
                self.robot.dynamics.velocity = np.array([0.0, 0.0, 0.0])
            else:
                self.robot.set_pose(new_pose)
        elif self._cmd_mode == "acceleration":
            avg_velocity = self.robot.dynamics.velocity + (0.5 * cmd * self._dt)
            final_velocity = self.robot.dynamics.velocity + (cmd * self._dt)
            new_pose = self.robot.dynamics.step(avg_velocity, self._dt)

            if self.robot.is_in_collision(pose=new_pose):
                self.robot.dynamics.velocity = np.array([0.0, 0.0, 0.0])
            else:
                self.robot.dynamics.velocity = final_velocity
                self.robot.set_pose(new_pose)
        else:
            raise NotImplementedError(f"Command mode '{self._cmd_mode}' is not supported")



        x, x_dot = self._get_obs()

        return (self._t, x, x_dot), 0.0, False, False, {}

    def _get_obs(self):
        x = np.array([self.robot.dynamics.pose.x,
                      self.robot.dynamics.pose.y,
                      self.robot.dynamics.pose.get_yaw()])
        x_dot = self.robot.dynamics.velocity
        return x, x_dot
    
    def render(self):
        if self.render_mode != "human":
            return
        if self._app is None:
            import sys
            from pyrobosim.gui.main import PyRoboSimGUI
            self._app = PyRoboSimGUI(self.world, sys.argv)
        canvas = self._app.main_window.canvas
        canvas.update_robots_plot()
        canvas.queue_draw()
        canvas.draw_and_sleep()
        self._app.processEvents()

    # other ###################################################################

    def occupancy_grid(self, resolution=0.05, inflation_radius=0.05):
        return OccupancyGrid.from_world(
            self.world,
            resolution=resolution,
            inflation_radius=inflation_radius,
        )

    def make_planning_problem(self):

        Location = UserType('Location')
        Robot = UserType('Robot')

        # Define fluents
        holding = Fluent('holding', BoolType(), r=Robot)
        on = Fluent('on', BoolType(), b=Location, l=Location)  # block b is on location l
        clear = Fluent('clear', BoolType(), l=Location)        # location is clear

        # Create objects
        robot1 = Object('robot1', Robot)
        robot2 = Object('robot2', Robot)
        A = Object('A', Location)
        B = Object('B', Location)
        table = Object('table', Location)

        # Define actions
        pickup = ConstraintEnabledInstantaneousAction('PickUp', r=Robot, b=Location, l=Location)
        r, b, l = pickup.parameters
        pickup.add_precondition(on(b, l))
        pickup.add_precondition(clear(b))  # block b must be clear to be picked up
        pickup.add_precondition(clear(l))  # location must be clear (optional, or remove)
        pickup.add_precondition(Not(holding(r)))
        pickup.add_effect(on(b, l), False)
        pickup.add_effect(clear(l), True)
        pickup.add_effect(clear(b), False)
        pickup.add_effect(holding(r), True)
        pickup.add_constraint_generator(LambdaConstraintGenerator(
            lambda goc, node_id: goc.add_linear_eq(node_id, np.eye(3), np.array([1.0, 0.0, 0.0]))
        ))


        putdown = ConstraintEnabledInstantaneousAction('PutDown', r=Robot, b=Location, l=Location)
        r, b, l = putdown.parameters
        putdown.add_precondition(holding(r))
        putdown.add_precondition(clear(l))
        putdown.add_precondition(Not(Equals(b, l)))  # can't put a block down on itself
        putdown.add_effect(on(b, l), True)
        putdown.add_effect(clear(l), False, condition=Not(Equals(l, table)))
        putdown.add_effect(clear(b), True)
        putdown.add_effect(holding(r), False)
        putdown.add_constraint_generator(LambdaConstraintGenerator(
            lambda goc, node_id: goc.add_linear_eq(node_id, np.eye(3), np.array([1.0, 1.0, 0.0]))
        ))

        # Create the problem
        problem = Problem('BlockStackingRobots')
        problem.add_fluent(holding, default_initial_value=False)
        problem.add_fluent(on, default_initial_value=False)
        problem.add_fluent(clear, default_initial_value=False)

        problem.add_action(pickup)
        problem.add_action(putdown)

        problem.add_object(robot1)
        problem.add_object(robot2)
        problem.add_object(A)
        problem.add_object(B)
        problem.add_object(table)

        # Initial state
        problem.set_initial_value(on(A, table), True)
        problem.set_initial_value(on(B, table), True)
        problem.set_initial_value(clear(A), True)
        problem.set_initial_value(clear(B), True)
        problem.set_initial_value(clear(table), True)
        problem.set_initial_value(holding(robot1), False)
        problem.set_initial_value(holding(robot2), False)

        # Goal: B on A
        problem.add_goal(on(B, A))

        return problem
