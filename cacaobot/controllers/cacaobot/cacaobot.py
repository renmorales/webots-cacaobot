from controller import Supervisor
from math import isinf
import os

try:
    import gym
    import numpy as np
    from numpy import inf
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install numpy gym stable_baselines3"')


class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=1000):
        super().__init__()

        # maximum observed objects
        self.__maxObjects = 7

        # 5 actions
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low='-1', high='1', shape=(self.__maxObjects * 12 + 58,), dtype=np.float32)
        self.state = None
        self.spec = gym.envs.registration.EnvSpec(
            id='WebotsEnv-v0', max_episode_steps=max_episode_steps)

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())

        # Robot
        self.__wheels = []
        self.__cacaobot = self.getFromDef("cacaobot")
        self.__cacaobot_transfield = self.__cacaobot.getField('translation')

        # Camera
        self.__camera = self.getDevice('CAM')
        self.__camera.enable(self.__timestep)

        # Camera Recognition
        self.__camera.recognitionEnable(self.__timestep)

        # init touch sensors
        self.__body_bumper = self.getDevice('body_bumper_sensor')
        self.__body_bumper.enable(self.__timestep)
        self.__container_bottom = self.getDevice('container_bottom_sensor')
        self.__container_bottom.enable(self.__timestep)

        # init robot parts
        #self.__disk = self.getDevice('disks')
        self.__containerSensor = self.getDevice('containerSensor')
        self.__container = self.getDevice('container')
        self.__containerSensor = self.getDevice('containerSensor')
        self.__containerSensor.enable(self.__timestep)

        # init GPS
        self.__gps = self.getDevice('gps')
        self.__gps.enable(self.__timestep)
        self.__gps2 = self.getDevice('gps2')
        self.__gps2.enable(self.__timestep)

        # int Lidar
        self.__lidar = self.getDevice('lidar')
        self.__lidar.enable(self.__timestep)
        self.__lidar.enablePointCloud()
        self.__lidar2 = self.getDevice('lidar2')
        self.__lidar2.enable(self.__timestep)
        self.__lidar2.enablePointCloud()

        self.total_steps = 0
        # get collection_area __cacao_transfield
        self.__collection_area = self.getFromDef('collection_area')
        self.__collection_area_Transfield = self.__collection_area.getField(
            'translation')

        # Init cacaos
        # for some reason "cacao1" does not work
        self.__cacaoNames = ["cacao0", "cacao5", "cacao2", "cacao3", "cacao4"]
        self.__cacao = []
        self.__cacao_transfield = []
        self.__cacao_rotationfield = []

        # for i in range(5):
        for i in range(5):
            self.__cacao.append(self.getFromDef(self.__cacaoNames[i]))
            self.__cacao_transfield.append(
                self.__cacao[i].getField("translation"))
            self.__cacao_rotationfield.append(
                self.__cacao[i].getField("rotation"))

        # container translation
        self.__container_box = self.getFromDef("container")
        self.__container_translation = self.__container_box.getField("translation")
        # Tools
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.__timestep)

        print("initialized Robot and Environment")

        self.step_count = 0

    def wait_keyboard(self):
        while self.keyboard.getKey() != ord('Y'):
            super().step(self.__timestep)

    def wait_keyboard(self):
        while self.keyboard.getKey() != ord('Y'):
            super().step(self.__timestep)

    def reset(self):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        # Motors
        self.__wheels = []

        # Speed Init
        self.maxVelocity = 6
        self.leftSpeed = 0
        self.rightSpeed = 0

        self.__camera.disable()

        for name in ['wheel1', 'wheel2', 'wheel3', 'wheel4']:
            wheel = self.getDevice(name)
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0)
            self.__wheels.append(wheel)

        self.__container.setVelocity(1)
        self.__container.setPosition(0)

        self.leftSpeed = 0
        self.rightSpeed = 0
        self.in_collision = 0

        self.reward = 0
        self.reward_multi = 1

        cacaobotX_position = np.random.uniform(low=-5, high=5)
        cacaobotY_position = -3.70

        cacaobotZ_position = -0.019
        self.__cacaobot_transfield.setSFVec3f(
            [cacaobotX_position, cacaobotY_position, cacaobotZ_position])

        cacaoZ_position = 0.02

        for translation in self.__cacao_transfield:

            x = np.random.uniform(low=-3, high=3)
            y = np.random.uniform(low=-3, high=2)

            if y > -1.3 and y < -0.95:
                if self.total_steps % 2:
                    y += 0.8
                else:
                    y -= 0.8

            elif y > 1.15 and y < 1.5:
                if self.total_steps % 2:
                    y += 0.8
                else:
                    y -= 0.8

            cacaoX_position = x
            cacaoY_position = y
            translation.setSFVec3f(
                [cacaoX_position, cacaoY_position, cacaoZ_position])

        # for observations

        self.__total_reward = 0
        self.__observed_objects = {}
        self.__cacao_status = {}

        # randominze collection_area_position
        collection_areaY_position = 4
        collection_areaX_position = np.random.uniform(low=-3, high=3)
        collection_areaZ_position = -0.04
        self.__collection_area_Transfield.setSFVec3f(
            [collection_areaX_position, collection_areaY_position, collection_areaZ_position])

        self.__camera.enable(self.__timestep)
        # Internals
        super().step(self.__timestep)

        # Open AI Gym generic
        return np.array([-1 for _ in range(self.observation_space.shape[0])]).astype(np.float32)

    def setLeftSpeed(self):
        self.__wheels[0].setVelocity(self.leftSpeed)
        self.__wheels[2].setVelocity(self.leftSpeed)

    def setRightSpeed(self):
        self.__wheels[1].setVelocity(self.rightSpeed)
        self.__wheels[3].setVelocity(self.rightSpeed)

    # scale anything from -1 to 1
    def scaler(self, value, min_value, max_value):
        return ((value - min_value)/((max_value - min_value) * 0.5) - 1)

    def cacao_in_container(self, cacao_position):
        cacaobot_position = self.__cacaobot_transfield.getSFVec3f()

        if (cacao_position[0] > cacaobot_position[0] - 0.4 and
                cacao_position[0] < cacaobot_position[0] + 0.4 and
                cacao_position[1] > cacaobot_position[1] - 0.4 and
                cacao_position[1] < cacaobot_position[1] + 0.4 and
                cacao_position[2] < 1):
            return True
        else:
            return False

    def cacaobot_in_collection_area(self, cacaobot_position):

        area = self.__collection_area_Transfield.getSFVec3f()

        if (cacaobot_position[0] > area[0] - 1 and
                cacaobot_position[0] < area[0] + 1 and
                cacaobot_position[1] > area[1] - 1 and
                cacaobot_position[1] < area[1] + 1):
            return True
        else:
            return False

    # function to determine if cacao is inside collection area
    def cacao_in_collection_area(self, cacao_position):

        area = self.__collection_area_Transfield.getSFVec3f()

        if (cacao_position[0] > area[0] - 1 and
                cacao_position[0] < area[0] + 1 and
                cacao_position[1] > area[1] - 1 and
                cacao_position[1] < area[1] + 1 and
                cacao_position[2] < 0.06):
            return True
        else:
            return False

    def step(self, action):

        self.step_count += 1
        self.total_steps += 1
        self.reward = 0
        done = False

        # for no disk only
        self.__container.setPosition(0)

        if action == 0:
            # go forward/increase speed
            if self.leftSpeed < self.rightSpeed:
                self.leftSpeed = self.rightSpeed
            else:
                self.rightSpeed = self.leftSpeed

            # if max speed do nothing
            if self.leftSpeed < self.maxVelocity:
                self.rightSpeed += 2
                self.leftSpeed += 2

        elif action == 1:
            # go backwards/decrease speed
            if self.leftSpeed > self.rightSpeed:
                self.leftSpeed = self.rightSpeed
            else:
                self.rightSpeed = self.leftSpeed

            if self.leftSpeed > (-1 * self.maxVelocity):
                self.rightSpeed -= 2
                self.leftSpeed -= 2

        elif action == 2:

            self.leftSpeed = 0.5 * self.maxVelocity
            self.rightSpeed = -0.5 * self.maxVelocity

        elif action == 3:

            self.leftSpeed = -0.5 * self.maxVelocity
            self.rightSpeed = 0.5 * self.maxVelocity

        elif action == 4:
            pass
            # For with disk only
            # Only works if in front of collection area
            # Flip Container
            # if self.__containerSensor.getValue() < 0.1 and self.__container_bottom.getValue() == 1:
            #    self.__container.setPosition(pi)
            # elif self.__containerSensor.getValue() > 3.14:
            #    self.__container.setPosition(0)

        if self.rightSpeed > self.maxVelocity:
            self.rightSpeed = self.maxVelocity

        if self.leftSpeed > self.maxVelocity:
            self.leftSpeed = self.maxVelocity

        if self.rightSpeed < (-1 * self.maxVelocity):
            self.rightSpeed = (-1 * self.maxVelocity)

        if self.leftSpeed < (-1 * self.maxVelocity):
            self.leftSpeed = (-1 * self.maxVelocity)

        # Sets the speed
        self.setLeftSpeed()
        self.setRightSpeed()

        super().step(self.__timestep)

        current_observed_objects = {}

        # loop all observed objects, didto mag compute
        collection_area_position_on_cam = [0, 0, 0]
        collection_area_position_on_image = [0, 0]
        collection_area_size_on_image = [0, 0]
        objectDetected = self.__camera.getRecognitionObjects()

        for obj in objectDetected:

            id = obj.get_id()

            if obj.model == 'collection_area':
                collection_area_position_on_cam = obj.get_position()
                collection_area_position_on_image = obj.get_position_on_image()
                collection_area_size_on_image = obj.get_size_on_image()

            elif obj.model == 'cacao':

                if id not in self.__observed_objects:
                    self.__observed_objects[id] = obj

                if id not in current_observed_objects:
                    current_observed_objects[id] = obj

        cacaobot_position = self.__cacaobot_transfield.getSFVec3f()

        closest_cacao_distance = 999
        closest_cacao_X = 0
        closest_cacao_id = None

        for obj in self.__observed_objects.values():

            id = obj.get_id()

            cacao_node = self.getFromId(id)
            cacao_transfield = cacao_node.getField('translation')
            cacao_position = cacao_transfield.getSFVec3f()
            distance = np.sqrt(
                (cacaobot_position[0]-cacao_position[0])**2 + (cacaobot_position[1]-cacao_position[1])**2)

            # get distance

            try:
                if obj.status == 0 or obj.status:
                    pass
            except:
                if self.cacao_in_collection_area(cacao_position):
                    obj.status = 2
                    self.__cacao_status[id] = 2
                else:
                    obj.status = 0
                    self.__cacao_status[id] = 0

            finally:
                if obj.status == 0:  # cacao on the ground
                    if distance < closest_cacao_distance:
                        closest_cacao_distance = distance
                        closest_cacao_id = id
                        closest_cacao_X = obj.get_position_on_image()[0]

                    if self.cacao_in_collection_area(cacao_position):
                        self.reward += 500
                        obj.status = 2
                        self.__cacao_status[id] = 2

                    elif self.cacao_in_container(cacao_position):
                        self.reward += 500
                        obj.status = 1
                        self.__cacao_status[id] = 1

                elif obj.status == 1:     # picked up by cacaobot

                    cacao_transfield.setSFVec3f(
                        [cacaobot_position[0], cacaobot_position[1], cacaobot_position[2]+1.5])
                    area = self.__collection_area_Transfield.getSFVec3f()

                    if action == 4:
                        if self.cacaobot_in_collection_area(cacaobot_position):
                            self.reward += 1000
                            obj.status = 2
                            self.__cacao_status[id] = 2
                            cacao_transfield.setSFVec3f(
                                [area[0], area[1], area[2]+2])

                elif obj.status == 2:  # cacao is inside the collecttion area
                    if not self.cacao_in_collection_area(cacao_position) and not self.cacao_in_container(cacao_position) and cacao_position[2] < 0.06:
                        # if cacao is on the ground
                        self.reward -= 700
                        self.__observed_objects[id].status = 0
                        self.__cacao_status[id] = 0
                    elif self.cacao_in_container(cacao_position):
                        self.reward -= 550
                        self.__observed_objects[id].status = 1
                        self.__cacao_status[id] = 1

        range_image = self.__lidar.getRangeImage()
        range_image2 = self.__lidar2.getRangeImage()

        # check collision
        if self.in_collision == 0:
            if self.__body_bumper.getValue() == 1.0:
                self.in_collision = 1
            else:
                #collision in front
                for val in range_image:
                    if val < 1.25:
                        self.in_collision = 1
                        break
        else:
            #self.reward -= 1
            if self.__body_bumper.getValue() == 0.0:
                self.in_collision = 0
                for val in range_image:
                    if val < 1.25:
                        self.in_collision = 1
                        break

        # compute updated distance
        closest_visible_cacao_distance = 999
        closest_visible_cacao_x = 0

        for obj in current_observed_objects.values():
            id = obj.get_id()

            cacao_node = self.getFromId(id)
            cacao_transfield = cacao_node.getField('translation')
            cacao_position = cacao_transfield.getSFVec3f()
            distance = np.sqrt(
                (cacaobot_position[0]-cacao_position[0])**2 + (cacaobot_position[1]-cacao_position[1])**2)

            if closest_visible_cacao_distance > distance:
                closest_visible_cacao_distance = distance
                closest_visible_cacao_x = obj.get_position_on_image()[0]
            else:
                closest_visible_cacao_x = 0

        container_has_cacao = 0
        for value in self.__cacao_status.values():
            if value == 1:
                container_has_cacao = 1
                break

        if self.__containerSensor.getValue() < 0.1:

            if self.in_collision == 0:

                if closest_cacao_id in current_observed_objects:
                    if ((action == 0 and closest_cacao_X > 180 and closest_cacao_X < 460) or
                        (action == 2 and closest_cacao_X > 460) or
                            (action == 3 and closest_cacao_X < 180)):
                        self.reward += 1/(closest_cacao_distance**2 + 0.5)
                    elif action == 1:
                        self.reward -= 1/(closest_cacao_distance**2 + 0.5)

                else:
                    if ((action == 0 and closest_visible_cacao_x > 180 and closest_visible_cacao_x < 460) or
                        (action == 2 and closest_visible_cacao_x > 460) or
                            (action == 3 and closest_visible_cacao_x < 180)):
                        self.reward += 0.75 / \
                            (closest_visible_cacao_distance**2 + 0.5)
                    elif action == 1:
                        self.reward -= 0.75 / \
                            (closest_visible_cacao_distance**2 + 0.5)

                if container_has_cacao:
                    if collection_area_position_on_image[1] > 170 and collection_area_position_on_image[1] and action == 0:
                        self.reward += (1/(((cacaobot_position[0]-self.__collection_area_Transfield.getSFVec3f()[
                                        0])**2 + (cacaobot_position[1]-self.__collection_area_Transfield.getSFVec3f()[1])**2) + 1))

        lidar_observations = range_image + range_image2

        for i in range(len(lidar_observations)):
            if isinf(lidar_observations[i]):
                lidar_observations[i] = 3
            lidar_observations[i] = self.scaler(lidar_observations[i], 0, 3)

        robot_observation = lidar_observations

        robot_observation.extend([self.scaler(collection_area_position_on_cam[0], 0, 10), self.scaler(collection_area_position_on_cam[1], -3, 3),
                                  self.scaler(collection_area_position_on_image[0], 0, 640), self.scaler(
                                      collection_area_position_on_image[1], 0, 640),
                                  self.scaler(collection_area_size_on_image[0], 0, 640), self.scaler(collection_area_size_on_image[1], 0, 640)])

        robot_observation.extend([self.scaler(self.__gps.getValues()[0], -6, 6), self.scaler(self.__gps.getValues()[0], -6, 6), self.scaler(self.__gps.getSpeed(), 0, 1.8),
                                  self.scaler(self.__containerSensor.getValue(), 0, 3.14), self.scaler(container_has_cacao, 0, 1)])

        for obs_object in current_observed_objects.values():
            robot_observation.extend([self.scaler((obs_object.get_position()[0]), 0, 10), self.scaler(obs_object.get_position()[1], -3, 3), self.scaler(obs_object.get_position()[2], -0.1, 3.5),
                                      obs_object.get_orientation()[0], obs_object.get_orientation()[1], obs_object.get_orientation()[
                2], self.scaler(obs_object.get_orientation()[3], -3.1416, 3.1416),
                self.scaler(obs_object.get_position_on_image()[0], 0, 640),  self.scaler(
                                          obs_object.get_position_on_image()[1], 0, 640),
                self.scaler(obs_object.get_size_on_image()[0], 0, 640), self.scaler(obs_object.get_size_on_image()[1], 0, 640)])

        for i in range(58 + self.__maxObjects * 12 - len(robot_observation)):
            robot_observation.append(-1)

        self.state = np.array(robot_observation)
        self.state[self.state == inf] = -1
        self.state[np.isnan(self.state)] = -1
        self.state[self.state < -1] = -1
        self.state[self.state > 1] = 1

        # print(robot_observation)
        print(f'Action: {action}, Reward: {self.__total_reward: .2f} Steps: {self.step_count} Collision: {self.in_collision}    Status: {self.__cacao_status}     Total Steps: {self.total_steps}/1000000')

        # if all cacaos are collected
        bonus = False  # bonus if task is completer
        if len(self.__cacao_status) > 4:
            for value in self.__cacao_status.values():
                if value != 2:
                    done = False
                    bonus = False
                    break
                else:
                    done = True
                    bonus = True
        if bonus:
            self.reward += 5000

        # Done  if reward or stepcount is reached or self.step_count == 5000
        if self.__total_reward > 9000 or self.__total_reward < -9000:
            done = True

        # Done if all cacaos are inside
        reward_multiplier = 1

        if self.in_collision == 1:
            self.reward -= 50
            done = True

        for value in self.__cacao_status.values():
            if value == 1:
                reward_multiplier += 0.05
            if value == 2:
                reward_multiplier += 0.1

        if self.reward > 0:
            self.reward *= reward_multiplier

        if done:
            self.step_count = 0

        self.__total_reward += self.reward

        return self.state.astype(np.float32), self.reward, done, {}


def main():

    log_path = os.path.join('Training', 'Logs')
    print(log_path)

    # Initialize the environment
    env = OpenAIGymEnvironment()
    check_env(env)

    PPO_Path = os.path.join('Training', 'Saved Models',
                            'PPO_Model_NoDisk_Trees')

    model = PPO('MlpPolicy', env, n_steps=2048,
                verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=1000000)
    model.save(PPO_Path)

    obs = env.reset()

    print("Done")
    #os.system("shutdown /s /t 1")
    for _ in range(100000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()
