import math
from typing import List
import numpy as np
from rsoccer_gym.Entities import Robot
from utils.Point import Point
from utils.Geometry import Geometry


PROP_VELOCITY_MIN_FACTOR: float = 0.1
MAX_VELOCITY: float = 1.5
ANGLE_EPSILON: float = 0.1
ANGLE_KP: float = 5
MIN_DIST_TO_PROP_VELOCITY: float = 720

ADJUST_ANGLE_MIN_DIST: float = 50
M_TO_MM: float = 1000.0


class Navigation:

  @staticmethod
  def degrees_to_radians(degrees):
    return degrees * (math.pi / 180.0)
  
  @staticmethod
  def radians_to_degrees(radians):
    return radians * (180.0 / math.pi)
  
  @staticmethod
  def global_to_local_velocity(vx, vy, theta):
    vx_local = vx * math.cos(theta) + vy * math.sin(theta)
    vy_local = -vx * math.sin(theta) + vy * math.cos(theta)
    return Point(vx_local, vy_local)

  @staticmethod
  def map_value(value, lLower, lHigher, rLower, rHigher):
    if (lHigher - lLower) == 0:
      return
    
    return ((value - lLower) * (rHigher - rLower) / (lHigher - lLower) + rLower)

  @staticmethod
  def goToPoint(robot: Robot, target: Point):
    target = Point(target.x * M_TO_MM, target.y * M_TO_MM)
    robot_position = Point(robot.x * M_TO_MM, robot.y * M_TO_MM)
    robot_angle = Navigation.degrees_to_radians(Geometry.normalize_angle(robot.theta, 0, 180))

    max_velocity = MAX_VELOCITY
    distance_to_target = robot_position.dist_to(target)
    kp = ANGLE_KP

    # Use proportional speed to decelerate when getting close to desired target
    proportional_velocity_factor = PROP_VELOCITY_MIN_FACTOR
    min_proportional_distance = MIN_DIST_TO_PROP_VELOCITY

    if distance_to_target <= min_proportional_distance:
      max_velocity = max_velocity * Navigation.map_value(distance_to_target, 0.0, min_proportional_distance, proportional_velocity_factor, 1.0)

    target_angle = (target - robot_position).angle()
    d_theta = Geometry.smallest_angle_diff(target_angle, robot_angle)

    if distance_to_target > ADJUST_ANGLE_MIN_DIST:
      v_angle = Geometry.abs_smallest_angle_diff(math.pi - ANGLE_EPSILON, d_theta)

      v_proportional = v_angle * (max_velocity / (math.pi - ANGLE_EPSILON))
      global_final_velocity = Geometry.from_polar(v_proportional, target_angle)
      target_velocity = Navigation.global_to_local_velocity(global_final_velocity.x, global_final_velocity.y, robot_angle)

      return target_velocity, -kp * d_theta
    else:
      return Point(0.0, 0.0), -kp * d_theta

  @staticmethod
  def navigate_to_goal(robot: Robot, goal: Point, obstacles: List[Point], sensing_radius: float):
    # Constants
    SAFE_DISTANCE = 100  # Minimum safe distance from obstacles (mm)
    KP_ANGLE = 0.3       # Proportional constant for angular velocity
    KP_SPEED = 0.5       # Proportional constant for linear speed

    robot_pos = Point(robot.x * M_TO_MM, robot.y * M_TO_MM)
    goal_pos = Point(goal.x * M_TO_MM, goal.y * M_TO_MM)
    robot_angle = robot.theta  # Current angle of the robot in radians

    # Step 1: Detect nearest obstacles
    visible_obstacles = [
        obs for obs in obstacles
        if robot_pos.dist_to(obs) <= sensing_radius
    ]

    # Step 2: Calculate sub-goal
    if visible_obstacles:
        # Find the nearest obstacle
        nearest_obs = min(visible_obstacles, key=lambda obs: robot_pos.dist_to(obs))
        obs_angle = (nearest_obs - robot_pos).angle()

        # Tangential escape point (to the left or right of the obstacle)
        escape_angle = obs_angle + math.pi / 2  # Adjust sign for left/right escape
        sub_goal = Point(
            nearest_obs.x + SAFE_DISTANCE * math.cos(escape_angle),
            nearest_obs.y + SAFE_DISTANCE * math.sin(escape_angle)
        )
    else:
        # No obstacles in range; head directly to the goal
        sub_goal = goal_pos

    # Step 3: Compute velocity commands
    distance_to_sub_goal = robot_pos.dist_to(sub_goal)
    target_angle = (sub_goal - robot_pos).angle()
    angle_error = Geometry.smallest_angle_diff(target_angle, robot_angle)

    # Proportional control for smooth movement
    linear_velocity = min(MAX_VELOCITY, KP_SPEED * distance_to_sub_goal)
    angular_velocity = KP_ANGLE * angle_error

    # Encapsulate the linear velocity to point towards the sub-goal
    linear_velocity_point = Point(
        linear_velocity * math.cos(target_angle),
        linear_velocity * math.sin(target_angle)
    )

    return linear_velocity_point, angular_velocity

