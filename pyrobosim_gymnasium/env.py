import copy

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


# Default single-robot 5x5 m square room, preserved as the zero-arg default
# so existing single-robot callers (e.g. `po_goc_mpc.environments.registry`)
# see identical behavior to before `PyRoboGym` grew multi-robot/room support.
DEFAULT_ROOM_FOOTPRINT = [(-2.5, -2.5), (2.5, -2.5), (2.5, 2.5), (-2.5, 2.5)]
DEFAULT_ROBOT_SPECS = [("robot0", Pose())]


class PyRoboGym(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
            self,
            robot_specs: list[tuple[str, Pose]] | None = None,
            room_name: str = "main_room",
            room_footprint: list[tuple[float, float]] | None = None,
            add_furniture: bool = True,
            use_lidar: bool = False,
            partial_obs_objects: bool = False,
            cmd_mode: str = "velocity",
            obs_object_names: list[str] | None = None,
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

        self.world.add_room(
            name=room_name,
            pose=Pose(x=0.0, y=0.0, z=0.0, yaw=0.0),
            footprint=room_footprint or DEFAULT_ROOM_FOOTPRINT,
            color="lightgray",
        )

        # Add Objects to the locations for planning tasks
        self._objects = {}

        if add_furniture:
            # Add Locations (Tables/Desks) within the same room
            table = self.world.add_location(
                category="table",
                parent=room_name,
                pose=Pose(x=-1.5, y=-1.5, z=0.0, yaw=0.0),
            )

            desk = self.world.add_location(
                category="desk",
                parent=room_name,
                pose=Pose(x=1.5, y=1.5, z=0.0, yaw=0.0)
            )

            def add_obj(category, parent):
                obj = self.world.add_object(category=category, parent=parent)
                self._objects[obj.name] = obj

            add_obj("banana", table)
            add_obj("apple", table)
            add_obj("water", desk)
            add_obj("apple", desk)

        if obs_object_names is not None:
            missing = [name for name in obs_object_names if name not in self._objects]
            if missing:
                raise ValueError(
                    f"obs_object_names contains unknown object names: {missing}. "
                    f"Available objects: {list(self._objects)}"
                )
        self._obs_object_names = obs_object_names or []

        # Add robots. `grasp_props` is stateless config, shared safely across
        # every robot; `lidar`/`path_executor`/the RRT planner each bind to
        # the one robot they're attached to internally, so those are built
        # fresh per robot instead.
        grasp_props = ParallelGraspProperties(
            max_width=0.175,
            depth=0.1,
            height=0.04,
            width_clearance=0.01,
            depth_clearance=0.01,
        )
        planner_config_rrt = {
            "bidirectional": True,
            "rrt_connect": False,
            "rrt_star": True,
            "collision_check_step_dist": 0.025,
            "max_connection_dist": 0.5,
            "rewire_radius": 1.5,
            "compress_path": False,
        }

        self.robots = []
        for name, pose in (robot_specs or DEFAULT_ROBOT_SPECS):
            lidar = Lidar2D(
                update_rate_s=0.1,
                angle_units="degrees",
                min_angle=-120.0,
                max_angle=120.0,
                angular_resolution=5.0,
                max_range_m=2.0,
            ) if use_lidar else None

            robot = Robot(
                name=name,
                radius=0.1,
                path_executor=ConstantVelocityExecutor(
                    linear_velocity=1.0,
                    dt=0.1,
                    max_angular_velocity=4.0,
                    validate_during_execution=True,
                ),
                sensors={"lidar": lidar} if lidar else None,
                grasp_generator=GraspGenerator(grasp_props),
                partial_obs_objects=partial_obs_objects,
                color="#CC00CC",
            )
            self.world.add_robot(robot, pose=pose)
            robot.set_path_planner(RRTPlanner(**planner_config_rrt))
            self.robots.append(robot)

        # Kept for single-robot callers that reach into `env.robot` directly
        # (e.g. `object_grasp_experiment.py`'s `env.robot._attach_object`).
        self.robot = self.robots[0]

        self._t = 0.0
        self._dt = 0.1

    @property
    def cmd_mode(self) -> str:
        """The `step(cmd)` command shape this env was constructed with --
        one of "position", "velocity", "position_velocity", "acceleration".
        Lets external callers (e.g. an MPC drive loop) build the right `cmd`
        shape without reaching into the private `_cmd_mode` attribute.
        """
        return self._cmd_mode

    @property
    def dt(self) -> float:
        """The fixed simulation timestep `step()` advances `t` by."""
        return self._dt

    def reset(self):
        self.world.reset()
        self.robots = list(self.world.robots)
        self.robot = self.robots[0]
        # `World.reset()` reloads the whole world from its stored YAML,
        # recreating every entity -- robots AND objects -- as brand-new
        # instances. `self.robots`/`self.robot` above are refreshed from the
        # new `self.world.robots` for exactly this reason; `self._objects`
        # needs the same treatment, or every entry keeps pointing at an
        # orphaned pre-reset `Object` that the live `World`/GUI no longer
        # knows about (so nothing done to it -- pose changes, grasp/carry,
        # `_get_obs()` reads -- has any visible or physical effect).
        self._objects = {obj.name: obj for obj in self.world.objects}
        self._t = 0.0

        x, x_dot = self._get_obs()

        return (self._t, x, x_dot), {}

    def step(self, cmd):
        """Applies `cmd` to every robot, in the order `robot_specs` were
        given at construction time (see `self.robots`).

        `cmd`'s shape depends on `cmd_mode`: for "position"/"velocity"/
        "acceleration", a flat `(num_robots * 3,)`-shaped array/sequence
        (reshaped below to `(num_robots, 3)`, one (x, y, yaw)-sized row per
        robot); for "position_velocity", a `(pose_cmd, velocity_cmd)` tuple
        of two such arrays. A single robot's flat 3-vector (or pair of
        3-vectors) reshapes to `(1, 3)` the same way, so single- and
        multi-robot callers share this exact path.
        """
        self._t += self._dt
        n = len(self.robots)

        if self._cmd_mode == "position_velocity":
            pose_cmd, velocity_cmd = cmd
            pose_cmd = np.asarray(pose_cmd).reshape(n, 3)
            velocity_cmd = np.asarray(velocity_cmd).reshape(n, 3)
        else:
            cmd_arr = np.asarray(cmd).reshape(n, 3)

        for i, robot in enumerate(self.robots):
            if self._cmd_mode == "position":
                new_pose = copy.copy(robot.dynamics.pose)
                new_pose.x = cmd_arr[i, 0]
                new_pose.y = cmd_arr[i, 1]
                new_pose.set_euler_angles(yaw=cmd_arr[i, 2])

                if robot.is_in_collision(pose=new_pose):
                    robot.dynamics.velocity = np.array([0.0, 0.0, 0.0])
                else:
                    robot.set_pose(new_pose)
            elif self._cmd_mode == "velocity":
                new_pose = robot.dynamics.step(cmd_arr[i], self._dt)

                if robot.is_in_collision(pose=new_pose):
                    robot.dynamics.velocity = np.array([0.0, 0.0, 0.0])
                else:
                    robot.set_pose(new_pose)
            elif self._cmd_mode == "position_velocity":
                new_pose = copy.copy(robot.dynamics.pose)
                new_pose.x = pose_cmd[i, 0]
                new_pose.y = pose_cmd[i, 1]
                new_pose.set_euler_angles(yaw=pose_cmd[i, 2])

                if robot.is_in_collision(pose=new_pose):
                    robot.dynamics.velocity = np.array([0.0, 0.0, 0.0])
                else:
                    robot.dynamics.velocity = velocity_cmd[i]
                    robot.set_pose(new_pose)
            elif self._cmd_mode == "acceleration":
                avg_velocity = robot.dynamics.velocity + (0.5 * cmd_arr[i] * self._dt)
                final_velocity = robot.dynamics.velocity + (cmd_arr[i] * self._dt)
                new_pose = robot.dynamics.step(avg_velocity, self._dt)

                if robot.is_in_collision(pose=new_pose):
                    robot.dynamics.velocity = np.array([0.0, 0.0, 0.0])
                else:
                    robot.dynamics.velocity = final_velocity
                    robot.set_pose(new_pose)
            else:
                raise NotImplementedError(f"Command mode '{self._cmd_mode}' is not supported")

            # Mirror pyrobosim's own carry convention for a held object --
            # `ConstantVelocityExecutor.execute_trajectory`
            # (pyrobosim/navigation/execution.py) re-poses
            # `robot.manipulated_object` to the robot's current pose every
            # tick while `follow_path()` runs. This env moves robots
            # directly rather than through that executor, so nothing else
            # replays that per-tick carry; using `robot.get_pose()` (rather
            # than `new_pose`) means a held object stays put, glued to the
            # robot, even on a step where collision blocked the robot's move.
            #
            # Deliberately NOT calling `create_polygons()` here:
            # `Object.update_visualization_polygon()` builds a brand-new
            # `PathPatch` every time it runs, discarding whatever patch is
            # actually attached to the GUI's axes (`WorldCanvas.show_objects()`
            # only wires up `obj.viz_patch` once). Calling it every tick would
            # silently orphan the on-screen patch at its very first pose,
            # while `WorldCanvas.update_object_plot()` kept moving a patch
            # nobody ever draws. `update_object_plot()` doesn't need a fresh
            # polygon anyway -- it moves the existing patch via an affine
            # transform derived from `obj.centroid` (fixed at the last
            # `create_polygons()` call) and the object's current `pose`.
            if robot.manipulated_object is not None:
                robot.manipulated_object.set_pose(robot.get_pose())

        x, x_dot = self._get_obs()

        return (self._t, x, x_dot), 0.0, False, False, {}

    def _get_obs(self):
        x_parts, x_dot_parts = [], []
        for robot in self.robots:
            x_parts.extend([robot.dynamics.pose.x,
                             robot.dynamics.pose.y,
                             robot.dynamics.pose.get_yaw()])
            x_dot_parts.extend(list(robot.dynamics.velocity))

        for name in self._obs_object_names:
            obj = self._objects[name]
            x_parts.extend([obj.pose.x, obj.pose.y])
            x_dot_parts.extend([0.0, 0.0])

        return np.array(x_parts), np.array(x_dot_parts)
    
    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return
        if self._app is None:
            import sys
            from pyrobosim.gui.main import PyRoboSimGUI
            self._app = PyRoboSimGUI(self.world, sys.argv, show=(self.render_mode == "human"))
        canvas = self._app.main_window.canvas
        canvas.update_robots_plot()
        # `update_robots_plot()` alone doesn't move a held object's plot --
        # PyRoboGym has no free-standing per-object render refresh, so a
        # carried object (see `step()`) would otherwise render frozen at its
        # last-drawn position even though `obj.pose` is moving every tick.
        for robot in self.robots:
            if robot.manipulated_object is not None:
                canvas.update_object_plot(robot.manipulated_object)
        if self.render_mode == "human":
            canvas.queue_draw()
            canvas.draw_and_sleep()
            self._app.processEvents()
        else:
            canvas.fig.canvas.draw()
            buf = canvas.fig.canvas.buffer_rgba()
            w, h = canvas.fig.canvas.get_width_height()
            img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            return img[:, :, :3]

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
            lambda goc, node_id: goc.add_robot_to_point_displacement_constraint(node_id, 0, 0, np.array([0.1, 0.0]))
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
            lambda goc, node_id: goc.add_robot_linear_eq(node_id, 0, np.eye(3), np.array([-1.0, -1.0, 0.0]))
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
