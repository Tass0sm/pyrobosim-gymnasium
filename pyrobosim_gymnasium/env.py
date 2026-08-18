import copy

import numpy as np
import gymnasium as gym

from unified_planning.shortcuts import UserType, BoolType
from unified_planning.model import Problem, Fluent, InstantaneousAction, Object

from planning_with_constraints import (
    ConstraintEnabledInstantaneousAction,
    LambdaConstraintGenerator,
    LambdaEdgeConstraintGenerator,
    UngroundedVariable,
)

# `goc-mpc`'s symbolic constraint API (used below to build `Pick`/`Place`'s
# constraint generators) is built on pydrake `Formula`s. `pyrobosim_gymnasium`
# doesn't declare a direct dependency on `pydrake` itself -- it's pulled in
# transitively by whatever environment this package is installed into
# (e.g. `po-goc-mpc`'s `goc-mpc` dependency), the same way `po-goc-mpc`
# itself never declares `pydrake` directly either.
from pydrake.symbolic import logical_and

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

# `make_planning_problem`'s `Pick` constraint generator's fallback approach
# offset/yaw, only used if a surface has no `nav_poses` configured at all
# (see `_nearest_nav_pose`) -- same 0.15 m magnitude `po_goc_mpc`'s
# hand-built experiments use (see e.g. `object_grasp_experiment.py`'s
# `GRASP_OFFSET`).
PICK_GRASP_OFFSET = np.array([-0.15, 0.0])
PICK_GRASP_YAW = 0.0


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
            buf = canvas.fig.canvas.buffer_rgba()
            w, h = canvas.fig.canvas.get_width_height()
            img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            return img[:, :, :3]
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

    def nearest_nav_pose(self, surface_name, target_xy):
        """Where `Pick` pins the robot: the closest of `surface_name`'s
        precomputed, collision-free approach poses (`Location.nav_poses`,
        and its object-spawn children's -- e.g. a desk's "desktop" spawn
        region has its own `nav_poses` ringing the desk, distinct from the
        (here, empty) `Location.nav_poses`) to `target_xy`.

        A fixed offset from the item's position (e.g. always "0.15m to its
        west") isn't safe in general: whether that lands the robot clear of
        the surface's own footprint or inside it depends on which side of
        the surface the item happens to be on. pyrobosim already computes
        real approach points for exactly this reason; picking the nearest
        one is a general, collision-aware replacement for a per-scenario
        hand-tuned offset constant. Falls back to
        `PICK_GRASP_OFFSET`/`PICK_GRASP_YAW` if the surface has no nav_poses
        configured at all.

        A real method (not a `make_planning_problem`-local closure) so a
        caller with fresher knowledge of an item's true position -- e.g. a
        driving loop's `on_reset` hook, once it has actually pinned the
        item, well after `make_planning_problem`/`Pick`'s constraint
        generator ran against whatever position was live at PLAN-build time
        -- can recompute the SAME approach point Pick's generator used and
        correct it in place via `GraphOfConstraints.set_param`, without
        duplicating this geometry (see `_pick_add`'s use of
        `goc.add_param`/`goc.param` below)."""
        loc = next(l for l in self.world.locations if l.name == surface_name)
        poses = list(loc.nav_poses)
        for child in loc.children:
            poses.extend(child.nav_poses)
        if not poses:
            return target_xy + PICK_GRASP_OFFSET, PICK_GRASP_YAW
        best = min(poses, key=lambda p: (p.x - target_xy[0]) ** 2 + (p.y - target_xy[1]) ** 2)
        return np.array([best.x, best.y]), best.get_yaw()

    def make_planning_problem(self):
        """Builds a `unified_planning.model.Problem` (types, fluents,
        actions, objects, and initial values) reflecting this env's live
        `World` state: one `Robot` object per `self.robots`, one `Surface`
        object per `self.world.locations` (tables/desks), one `Item` object
        per `self._objects` (graspable objects).

        No goal is set here -- callers add the goal for their specific task
        (`problem.add_goal(...)`) and solve with PCOP
        (`po_goc_mpc.symbolic_planning`).

        `Pick`/`Place` are `ConstraintEnabledInstantaneousAction`s whose
        generators emit goc-mpc's symbolic-formula constraint API
        (`goc.add_constraint`/`goc.add_edge_constraint` with pydrake
        `Formula`s), the same style `object_grasp_experiment.py`/
        `object_handoff_experiment.py` hand-build, rather than the old typed
        convenience calls (`add_robot_to_point_displacement_constraint`,
        `add_robot_linear_eq`). `Move` is deliberately a plain
        `InstantaneousAction` (no constraint generator): it exists only so
        the planner can satisfy `Pick`'s `robot_at` precondition, and is
        skipped entirely by the plan-to-GoC translator
        (`po_goc_mpc.symbolic_planning.goc_builder`) -- an unconstrained
        node in the middle of the GoC isn't expected to resolve, so `Move`
        contributes no geometry, only task-planning-time causal structure.
        """
        Surface = UserType('Surface')
        RobotType = UserType('Robot')
        Item = UserType('Item')

        # Define fluents
        robot_at = Fluent('robot_at', BoolType(), r=RobotType, s=Surface)
        on = Fluent('on', BoolType(), i=Item, s=Surface)          # item i is on surface s
        holding = Fluent('holding', BoolType(), r=RobotType, i=Item)
        hand_empty = Fluent('hand_empty', BoolType(), r=RobotType)

        # Create objects from live World state
        robot_objs = {robot.name: Object(robot.name, RobotType) for robot in self.robots}
        surface_objs = {loc.name: Object(loc.name, Surface) for loc in self.world.locations}
        item_objs = {name: Object(name, Item) for name in self._objects}

        # Alias -- see PyRoboGym.nearest_nav_pose (a real method, not a
        # local closure, so an on_reset hook with fresher item-position
        # knowledge can call the exact same logic later via
        # env.nearest_nav_pose -- see _pick_add's use of goc.add_param below).
        _nearest_nav_pose = self.nearest_nav_pose

        def _place_region_bounds(surface_name, item_xy):
            """Continuous, collision-free placement region for `Place` to
            search over (see `_place_add`): a 1D band running along
            whichever side of `surface_name` is nearest `item_xy` (via
            `_nearest_nav_pose`), pinned on the approach axis to that
            nav_pose's own already-collision-free coordinate, and free to
            slide along the OTHER axis across the surface's own extent.

            A box over the surface's own footprint (what this used to be)
            is NOT safe: PyRoboGym's carry model glues a held item exactly
            to the robot's own pose (`step()`'s
            `robot.manipulated_object.set_pose(robot.get_pose())`, zero
            standoff), so the item's resting position has to be
            collision-free FOR THE ROBOT, not merely "on" the surface --
            and a surface's own footprint is normally inside the robot's
            collision zone (confirmed: `table0`'s tabletop spawn region
            (-1.85, -2.0, -1.15, -1.0) sits entirely inside its own
            physical footprint (-1.95, -2.1, -1.05, -0.9)). Pinning the
            band to the nearest nav_pose's coordinate on one axis keeps
            every point in the region strictly outside the surface's own
            polygon on that axis, exactly like a single pinned nav_pose
            would, while the other axis is genuinely free.

            Which axis is "the approach axis" is derived from which axis
            the nearest nav_pose is offset along relative to the surface's
            own bounding-box center, rather than hardcoded to x or y, so
            this generalizes past table0's left/right nav_poses to a
            surface approached from above/below instead.
            """
            loc = next(l for l in self.world.locations if l.name == surface_name)
            xmin, ymin, xmax, ymax = loc.polygon.bounds
            nav_xy, _ = _nearest_nav_pose(surface_name, item_xy)
            cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
            if abs(nav_xy[0] - cx) >= abs(nav_xy[1] - cy):
                return float(nav_xy[0]), ymin, float(nav_xy[0]), ymax
            return xmin, float(nav_xy[1]), xmax, float(nav_xy[1])

        def _nearest_surface_name(robot):
            """Which surface a robot's `robot_at` starts pinned to: this
            domain is purely qualitative (task-level "who's near what"),
            not a claim about exact geometry -- real navigation is resolved
            later by goc-mpc's waypoint/timing solvers, `Move` here is only
            a symbolic bookkeeping step -- so "nearest surface by Euclidean
            distance" is a reasonable default even when the robot's actual
            spawn pose is out in open floor, not literally at any surface.
            """
            if not self.world.locations:
                return None
            best = min(
                self.world.locations,
                key=lambda loc: (loc.pose.x - robot.dynamics.pose.x) ** 2
                                 + (loc.pose.y - robot.dynamics.pose.y) ** 2,
            )
            return best.name

        # Define actions

        move = InstantaneousAction('Move', r=RobotType, src=Surface, dst=Surface)
        r, src, dst = move.parameters
        move.add_precondition(robot_at(r, src))
        move.add_effect(robot_at(r, src), False)
        move.add_effect(robot_at(r, dst), True)

        pick = ConstraintEnabledInstantaneousAction('Pick', r=RobotType, i=Item, s=Surface)
        r, i, s = pick.parameters
        pick.add_precondition(robot_at(r, s))
        pick.add_precondition(on(i, s))
        pick.add_precondition(hand_empty(r))
        pick.add_effect(on(i, s), False)
        pick.add_effect(holding(r, i), True)
        pick.add_effect(hand_empty(r), False)

        def _pick_add(goc, node_id, params, robot_index, object_index):
            # `params` are grounded object-name strings, except a Robot
            # parameter PCOP left unresolved comes through as an
            # `UngroundedVariable` sentinel instead (see
            # `po_goc_mpc.symbolic_planning.grounding` -- deferred to
            # goc-mpc's own assignable-variable machinery rather than
            # resolved before calling generators).
            r_name, i_name, s_name = params
            item = self._objects[i_name]
            item_xy = np.array([item.pose.x, item.pose.y])
            approach_xy, approach_yaw = self.nearest_nav_pose(s_name, item_xy)
            # No object_q pin here: the object's position at THIS node is
            # already the live, real one via GraphOfConstraints' own
            # stationarity-to-x0 machinery (both waypoint solvers tie an
            # un-held object's object_q back to the actual runtime state --
            # see MILPWaypointMPC's depot exact-rigidity and
            # EvolutionaryWaypointSolver's _batch_depot_stationary_fn), not
            # something this generator needs to (re-)assert. A hard pin here
            # would instead FIGHT that machinery whenever `item_xy` (read at
            # graph-BUILD time) drifts from the object's true position by
            # solve time (e.g. it hasn't been placed at its real start pose
            # yet when make_graph() runs) -- exactly the bug this generator
            # used to have.
            #
            # `approach_xy`/`approach_yaw` (where Pick pins the ROBOT) are
            # a different story: they're not carried by any dynamics, so
            # they DO need pinning -- but as goc.add_param placeholders
            # (an editable runtime constant), not a value baked into the
            # Formula from this same possibly-stale `item_xy`. A caller with
            # fresher knowledge of the item's real position (e.g.
            # on_reset, once it has actually pinned the item -- see
            # pick_place_single_robot_experiment.py's make_hooks) can correct
            # these via goc.set_param without touching the constraint's
            # Formula or the graph/solver structure -- see PyRoboGym.
            # nearest_nav_pose's docstring.
            agent_q = (goc.var_agent_q(r_name.var_id) if isinstance(r_name, UngroundedVariable)
                       else goc.agent_q(robot_index[r_name]))
            px = goc.add_param(float(approach_xy[0]))
            py = goc.add_param(float(approach_xy[1]))
            pyaw = goc.add_param(float(approach_yaw))
            goc.add_constraint(node_id, agent_q[0] == goc.param(px))
            goc.add_constraint(node_id, agent_q[1] == goc.param(py))
            goc.add_constraint(node_id, agent_q[2] == goc.param(pyaw))
            # Stashed for a runtime on_reset hook to correct once the item's
            # real start pose is known -- see nearest_nav_pose's docstring.
            # Overwritten harmlessly if make_planning_problem ever runs more
            # than once (only the last Pick's params are recoverable this
            # way; fine for this single-item-single-pick generator).
            self._pick_runtime_params = {
                "surface": s_name, "agent_x": px, "agent_y": py, "agent_yaw": pyaw,
            }

        pick.add_constraint_generator(LambdaConstraintGenerator(_pick_add))

        place = ConstraintEnabledInstantaneousAction('Place', r=RobotType, i=Item, s=Surface)
        r, i, s = place.parameters
        place.add_precondition(holding(r, i))
        place.add_effect(holding(r, i), False)
        place.add_effect(hand_empty(r), True)
        place.add_effect(on(i, s), True)
        place.add_effect(robot_at(r, s), True)

        def _place_add(goc, node_id, params, robot_index, object_index):
            _r_name, i_name, s_name = params
            item = self._objects[i_name]
            item_xy = np.array([item.pose.x, item.pose.y])
            i_idx = object_index[i_name]
            # A single pinned target point (the old approach: nearest
            # collision-free nav_pose to the surface's resting position) is
            # over-constraining: it forces one specific spot regardless of
            # whether the robot can actually reach it without crossing an
            # obstacle. Instead, let the item be placed anywhere along a
            # collision-free band next to the surface (an inequality
            # region, not an equality pin -- see `_place_region_bounds` for
            # why it's a band next to the surface rather than a box over
            # it) -- the robot's own position at this node is still fully
            # determined by the transport edge back to Pick
            # (`_place_edge_add`, unchanged: same relative offset from the
            # object the robot grasped it at), so freeing the item's
            # position also frees the robot's, and the waypoint solver's
            # own edge-cost objective (see po_goc_mpc's `edge_cost_fn=`
            # wiring) is what picks a point in that region minimizing real
            # (obstacle-aware) travel cost -- rather than this generator
            # hand-picking a single point via a Euclidean nearest-neighbor
            # heuristic that has no idea whether the resulting robot pose is
            # reachable.
            xmin, ymin, xmax, ymax = _place_region_bounds(s_name, item_xy)
            obj_q = goc.object_q(i_idx)
            goc.add_constraint(node_id, logical_and(
                obj_q[0] >= xmin, obj_q[0] <= xmax,
                obj_q[1] >= ymin, obj_q[1] <= ymax,
            ))
            # Still deliberately NOT pinning the robot's own position here --
            # see object_grasp_experiment.py's build_controller docstring:
            # the robot's (x, y) at the place node is meant to be determined
            # *purely* by the transport edge back to the grasp node.

        place.add_constraint_generator(LambdaConstraintGenerator(_place_add))

        def _place_edge_add(goc, u_node_id, v_node_id, u_step, u_params, v_params,
                             robot_index, object_index):
            # Only add the transport/holding edge constraint if the
            # predecessor is the matching Pick (same robot AND item) --
            # domain-specific pairing logic that belongs here, in Place's
            # own generator, not in the generic plan-to-GoC translator.
            if u_step.action.name != 'Pick':
                return
            u_r, u_i, _u_s = u_params
            v_r, v_i, _v_s = v_params
            if u_r != v_r or u_i != v_i:
                return
            # `UngroundedVariable(var_id)` compares equal across both params
            # exactly when they're the same deferred PCOP variable (frozen
            # dataclass equality), so the check above already covers "same
            # deferred robot" the same way it covers "same resolved name" --
            # no special-casing needed there.
            i_idx = object_index[v_i]

            # Transport (rigid-carry) edge: while grasped, the object moves
            # rigidly with the robot from u_node_id to v_node_id. Uses the
            # canonical hold registry (`add_hold`/`add_assignable_hold`,
            # `GraphOfConstraints`) rather than hand-written
            # `add_edge_constraint` calls -- see `goc-mpc/examples/
            # test_hold_registry.py`'s docstring and `GraphOfConstraintsMPC`'s
            # own `hold_drift_tolerance` doc comment: this single call
            # replaces both the old rigid-carry edge constraint (`live=True`,
            # for the waypoint solve) and the separate runtime proximity/
            # drift check (formerly a second hand-written `HOLDING_MAX_DIST`
            # edge constraint) -- `GraphOfConstraintsMPC._backtrack` already
            # re-checks every registered hold's drift against the real state
            # each control cycle (`_hold_violated`) and reopens `u_node_id`
            # if it's exceeded, for both a statically-assigned hold and an
            # assignable one (`_hold_agent` resolves `hold.var_id` via the
            # solver's own last assignment). That drift check reads real
            # state through `graph.link_pose`/`graph.point_position`, which
            # require `GraphOfConstraints(..., workspace_dim=2)` for this
            # (planar) domain -- see `pick_place_task_experiment.py`'s own
            # comment on that constructor call, and goc-mpc's `PointPosFromRow`
            # (utils.hpp), fixed to stop reading past a 2-wide object's own
            # slice for exactly this case.
            if isinstance(v_r, UngroundedVariable):
                goc.add_assignable_hold(u_node_id, v_node_id, v_r.var_id, [i_idx])
            else:
                goc.add_hold(u_node_id, v_node_id, robot_index[v_r], [i_idx])

        place.add_edge_constraint_generator(LambdaEdgeConstraintGenerator(_place_edge_add))

        # Create the problem
        problem = Problem('PickPlaceTask')
        problem.add_fluent(robot_at, default_initial_value=False)
        problem.add_fluent(on, default_initial_value=False)
        problem.add_fluent(holding, default_initial_value=False)
        problem.add_fluent(hand_empty, default_initial_value=True)

        problem.add_action(move)
        problem.add_action(pick)
        problem.add_action(place)

        for obj in robot_objs.values():
            problem.add_object(obj)
        for obj in surface_objs.values():
            problem.add_object(obj)
        for obj in item_objs.values():
            problem.add_object(obj)

        # Initial state, read from the live World.
        for robot in self.robots:
            r_obj = robot_objs[robot.name]
            if robot.manipulated_object is not None:
                i_obj = item_objs[robot.manipulated_object.name]
                problem.set_initial_value(holding(r_obj, i_obj), True)
                problem.set_initial_value(hand_empty(r_obj), False)
            else:
                nearest = _nearest_surface_name(robot)
                if nearest is not None:
                    problem.set_initial_value(robot_at(r_obj, surface_objs[nearest]), True)

        held_item_names = {robot.manipulated_object.name for robot in self.robots
                            if robot.manipulated_object is not None}
        def _surface_name_of(entity):
            """`Object.parent` is typically an `ObjectSpawn`, not the
            `Location` itself (see `pyrobosim.core.objects.Object`'s
            docstring) -- walk up `.parent` links until hitting a name that
            matches a known `Surface`, or run out of ancestors."""
            node = entity
            while node is not None:
                if node.name in surface_objs:
                    return node.name
                node = getattr(node, 'parent', None)
            return None

        for name, item in self._objects.items():
            if name in held_item_names:
                continue
            surface_name = _surface_name_of(item.parent)
            if surface_name is not None:
                problem.set_initial_value(on(item_objs[name], surface_objs[surface_name]), True)

        return problem
