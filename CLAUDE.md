# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A [Gymnasium](https://gymnasium.farama.org/) environment wrapper around [pyrobosim](https://github.com/sea-bass/pyrobosim), a 2D robot simulation library. The package exposes a `PyRoboGym(gym.Env)` class for training RL agents in a pyrobosim world.

## Package Management

This project uses [Poetry](https://python-poetry.org/). Python 3.12 is required (pinned to `>=3.12,<3.13`).

```bash
poetry install          # install dependencies
poetry add <pkg>        # add a dependency
poetry run python ...   # run within the virtualenv
```

## Key Dependencies

- `pyrobosim ^4.3.4` — robot world, navigation (RRT/PRM planners), manipulation, sensors
- `gymnasium ^1.3.0` — standard RL environment interface
- `unified_planning` — PDDL-style task planning (used in `make_planning_problem`)
- `planning_with_constraints` — extends unified_planning with constraint-enabled actions (not in pyproject.toml; must be installed separately)

## Architecture

All environment logic lives in `pyrobosim_gymnasium/env.py`. The single class `PyRoboGym` does everything:

- **World setup** (`__init__`): constructs a fixed 5×5 m single-room world with a table, desk, and four food/drink objects. Adds one `Robot` with an RRT* path planner and optional Lidar2D sensor.
- **Gym interface**: `reset()` resets the pyrobosim world and returns `(t, x, x_dot), info`; `step(cmd_vel)` advances the dynamics by `dt=0.1 s`, handles collision, and returns the same tuple with a zero reward.
- **Observation**: `_get_obs()` returns `(x, x_dot)` where `x = [pos_x, pos_y, yaw]` and `x_dot` is the 3-DOF velocity from `robot.dynamics`.
- **Utilities**: `occupancy_grid(resolution, inflation_radius)` builds an `OccupancyGrid` from the world; `make_planning_problem()` constructs a block-stacking `unified_planning.Problem` with constraint-annotated `PickUp`/`PutDown` actions.

Constructor flags:
- `use_lidar: bool` — attaches a 2D lidar sensor to the robot
- `partial_obs_objects: bool` — enables partial object observability on the robot
