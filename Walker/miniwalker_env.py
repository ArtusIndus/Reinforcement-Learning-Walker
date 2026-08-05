import gymnasium as gym
from gymnasium import spaces
import numpy as np
import Box2D
from Box2D import b2RevoluteJointDef
import pygame


class MiniWalkerEnv(gym.Env):

    def __init__(self):
        super().__init__()

        pygame.init()

        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("MiniWalker")

        self.clock = pygame.time.Clock()

        self.world = Box2D.b2World(gravity=(0, -9.81))

        self.body = None

        self.legs = []

        self.joints = []

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(4,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )



    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.world = Box2D.b2World(gravity=(0, -9.81))

        self.legs = []
        self.joints = []

        self.ground = self.world.CreateStaticBody(
            position=(0, 0)
        )

        self.ground.CreatePolygonFixture(
            box=(20, 1),
            friction=0.8
        )

        self.body = self.world.CreateDynamicBody(
            position=(0, 3)
        )

        self.body.CreatePolygonFixture(
            box=(0.4, 0.2),
            density=1.0,
            friction=0.5
        )

        self.left_upper = self.world.CreateDynamicBody(
            position=(-0.2, 2.5),
            angle=-0.15
        )

        self.left_upper.CreatePolygonFixture(
            box=(0.08, 0.35),
            density=1.0
        )

        self.left_lower = self.world.CreateDynamicBody(
            position=(-0.2, 1.8),
            angle=0.15
        )

        self.left_lower.CreatePolygonFixture(
            box=(0.07, 0.35),
            density=1.0
        )

        self.left_foot = self.world.CreateDynamicBody(
            position=(-0.08, 1.35)
        )

        self.left_foot.CreatePolygonFixture(
            box=(0.15, 0.04),
            density=0.5,
            friction=2.0
        )

        self.right_upper = self.world.CreateDynamicBody(
            position=(0.2, 2.5),
            angle=0.2
        )

        self.right_upper.CreatePolygonFixture(
            box=(0.08, 0.35),
            density=1.0
        )

        self.right_lower = self.world.CreateDynamicBody(
            position=(0.2, 1.8),
            angle=0.4
        )

        self.right_lower.CreatePolygonFixture(
            box=(0.07, 0.35),
            density=1.0
        )

        self.right_foot = self.world.CreateDynamicBody(
            position=(0.08, 1.35)
        )

        self.right_foot.CreatePolygonFixture(
            box=(0.15, 0.04),
            density=0.5,
            friction=2.0
        )

        hip = b2RevoluteJointDef(
            bodyA=self.body,
            bodyB=self.left_upper,
            localAnchorA=(-0.2, -0.2),
            localAnchorB=(0, 0.35),
            enableMotor=True,
            enableLimit=True,
            lowerAngle=-0.8,
            upperAngle=1.1,
            maxMotorTorque=20,
            motorSpeed=0
        )

        knee = b2RevoluteJointDef(
            bodyA=self.left_upper,
            bodyB=self.left_lower,
            localAnchorA=(0, -0.35),
            localAnchorB=(0, 0.35),

            enableMotor=True,
            enableLimit=True,

            lowerAngle=-1.6,
            upperAngle=1,

            maxMotorTorque=20,
            motorSpeed=0
        )

        right_hip = b2RevoluteJointDef(
            bodyA=self.body,
            bodyB=self.right_upper,

            localAnchorA=(0.2, -0.2),
            localAnchorB=(0, 0.35),

            enableMotor=True,
            enableLimit=True,

            lowerAngle=-1.1,
            upperAngle=0.8,

            maxMotorTorque=20,
            motorSpeed=0
        )

        right_knee = b2RevoluteJointDef(
            bodyA=self.right_upper,
            bodyB=self.right_lower,

            localAnchorA=(0, -0.35),
            localAnchorB=(0, 0.35),

            enableMotor=True,
            enableLimit=True,

            lowerAngle=-1.6,
            upperAngle=1,

            maxMotorTorque=20,
            motorSpeed=0
        )

        left_ankle = b2RevoluteJointDef(
            bodyA=self.left_lower,
            bodyB=self.left_foot,

            localAnchorA=(0, -0.35),
            localAnchorB=(-0.10, 0),

            enableMotor=False,

            enableLimit=True,
            lowerAngle=0,
            upperAngle=0
        )

        right_ankle = b2RevoluteJointDef(
            bodyA=self.right_lower,
            bodyB=self.right_foot,

            localAnchorA=(0, -0.35),
            localAnchorB=(-0.10, 0),

            enableMotor=False,

            enableLimit=True,
            lowerAngle=0,
            upperAngle=0
        )

        self.right_ankle = self.world.CreateJoint(right_ankle)

        self.left_ankle = self.world.CreateJoint(left_ankle)

        self.hip = self.world.CreateJoint(hip)

        self.knee = self.world.CreateJoint(knee)

        self.right_hip = self.world.CreateJoint(right_hip)

        self.right_knee = self.world.CreateJoint(right_knee)

        self.hip_target = self.hip.angle
        self.knee_target = self.knee.angle

        self.right_hip_target = self.right_hip.angle
        self.right_knee_target = self.right_knee.angle

        obs = np.array([
            self.body.position.x,
            self.body.linearVelocity.x,
            self.body.angle,
            self.body.angularVelocity,
            self.hip.angle,
            self.hip.speed,
            self.knee.angle,
            self.knee.speed,
            self.right_hip.angle,
            self.right_hip.speed,
            self.right_knee.angle,
            self.right_knee.speed,
        ], dtype=np.float32)

        return obs, {}



    def render(self):

        self.screen.fill((150, 150, 230))

        self.draw_body(self.ground, (80, 80, 80))

        self.draw_body(self.body, (0,0,255))

        self.draw_body(self.left_upper, (255,0,0))

        self.draw_body(self.left_lower, (0,180,0))

        self.draw_body(self.right_upper, (255, 120, 0))

        self.draw_body(self.right_lower, (0, 255, 120))

        self.draw_body(self.left_foot, (40,40,40))

        self.draw_body(self.right_foot, (40,40,40))

        pygame.display.flip()

        pygame.event.pump()
        self.clock.tick(60)



    def draw_body(self, body, color=(0, 0, 255)):

        for fixture in body.fixtures:

            vertices = []

            for v in fixture.shape.vertices:

                world_v = body.transform * v

                screen_x = 400 + world_v[0] * 50
                screen_y = 550 - world_v[1] * 50

                vertices.append((screen_x, screen_y))

            pygame.draw.polygon(
                self.screen,
                color,
                vertices
            )



    def step(self, action):

        action = np.asarray(action, dtype=np.float32)


        self.hip_target  += action[0] * 0.4
        self.knee_target += action[1] * 0.4

        self.hip_target = np.clip(self.hip_target, -0.8, 1.1)
        self.knee_target = np.clip(self.knee_target, -1.6, -0.1)

        hip_error = self.hip_target - self.hip.angle
        knee_error = self.knee_target - self.knee.angle

        self.hip.motorSpeed = float(hip_error) * 7.0
        self.knee.motorSpeed = float(knee_error) * 7.0

#######################################################################

        self.right_hip_target  += action[2] * 0.4
        self.right_knee_target += action[3] * 0.4

        self.right_hip_target = np.clip(self.right_hip_target, -0.8, 1.1)
        self.right_knee_target = np.clip(self.right_knee_target, -1.6, 0.1)

        right_hip_error = self.right_hip_target - self.right_hip.angle
        right_knee_error = self.right_knee_target - self.right_knee.angle

        self.right_hip.motorSpeed = float(right_hip_error) * 7.0
        self.right_knee.motorSpeed = float(right_knee_error) * 7.0



        self.world.Step(1/60, 6, 2)

        obs = np.array([
            self.body.position.x,
            self.body.linearVelocity.x,
            self.body.angle,
            self.body.angularVelocity,
            self.hip.angle,
            self.hip.speed,
            self.knee.angle,
            self.knee.speed,
            self.right_hip.angle,
            self.right_hip.speed,
            self.right_knee.angle,
            self.right_knee.speed,
        ], dtype=np.float32)

        x_vel = float(self.body.linearVelocity.x)
        angle = float(self.body.angle)

        hip_err = float(self.hip_target - self.hip.angle)
        knee_err = float(self.knee_target - self.knee.angle)

        right_hip_err = float(self.right_hip_target - self.right_hip.angle)
        right_knee_err = float(self.right_knee_target - self.right_knee.angle)

        control_cost = 0.05 * (
            hip_err**2 +
            knee_err**2 +
            right_hip_err**2 +
            right_knee_err**2
        )

        balance_cost = 2.0 * (angle**2)

        action_cost = 0.005 * np.sum(action**2)

        alive_bonus = 0.2

        reward = (
            2.0 * x_vel
            + alive_bonus
            - control_cost
            - balance_cost 
            - action_cost
        )

        terminated = False
        if abs(self.body.angle) > 1:
            print("RESET!!!")
            terminated = True
            reward -= 100

        return obs, reward, terminated, False, {}
