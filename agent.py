from utils.ssl.Navigation import Navigation
from utils.ssl.base_agent import BaseAgent

class ExampleAgent(BaseAgent):
    def __init__(self, id=0, yellow=False):
        super().__init__(id, yellow)

    def decision(self):
        if len(self.targets) == 0:
            return

        obstacles = list(self.opponents.values()) + list(self.teammates.values())

        #target_velocity, target_angle_velocity = Navigation.goToPoint(self.robot, self.targets[0])
        target_velocity, target_angle_velocity = Navigation.navigate_to_goal(robot=self.robot, goal=self.targets[0], obstacles=obstacles, sensing_radius=100)
        self.set_vel(target_velocity)
        self.set_angle_vel(target_angle_velocity)

        return

    def post_decision(self):
        pass
