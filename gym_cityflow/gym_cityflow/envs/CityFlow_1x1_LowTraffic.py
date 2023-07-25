import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import cityflow
import numpy as np
import os
import traceback

class CityFlow_1x1_LowTraffic(gym.Env):
    """
    Description:
        A single intersection with low traffic.
        8 roads, 1 intersection (plus 4 virtual intersections).

    State:
        Type: array[16]
        The number of vehicless and waiting vehicles on each lane.

    Actions:
        Type: Discrete(9)
        index of one of 9 light phases.

        Note:
            Below is a snippet from "roadnet.json" file which defines lightphases for "intersection_1_1".

            "lightphases": [
              0 {"time": 5, "availableRoadLinks": []},
              1 {"time": 30, "availableRoadLinks": [ 0, 4 ] },
              2 {"time": 30, "availableRoadLinks": [ 2, 7 ] },
              3 {"time": 30, "availableRoadLinks": [ 1, 5 ] },
              4 {"time": 30,"availableRoadLinks": [3,6]},
              5 {"time": 30,"availableRoadLinks": [0,1]},
              6 {"time": 30,"availableRoadLinks": [4,5]},
              7 {"time": 30,"availableRoadLinks": [2,3]},
              8 {"time": 30,"availableRoadLinks": [6,7]}]

            "lightphases": [
              0 {"time": 5, "availableRoadLinks": []},
              1 {"time": 30, "availableRoadLinks": [ 0, 4 ] },
              2 {"time": 30, "availableRoadLinks": [ 2, 7 ] },
              3 {"time": 30, "availableRoadLinks": [ 1, 5 ] },
              4 {"time": 30,"availableRoadLinks": [3,6,0,1]},
              5 {"time": 30,"availableRoadLinks": [0,1]},
              6 {"time": 30,"availableRoadLinks": [4,5]},
              7 {"time": 30,"availableRoadLinks": [2,3]},
              8 {"time": 30,"availableRoadLinks": [6,7]}]

    Reward:
        The total amount of time -- in seconds -- that all the vehicles in the intersection
        waitied for.

        Todo: as a way to ensure fairness -- i.e. not a single lane gets green lights for too long --
        instead of simply summing up the waiting time, we could weigh the waiting time of each car by how
        much it had waited so far.
    """

    metadata = {'render.modes':['human']}
    def __init__(self):
        #super(CityFlow_1x1_LowTraffic, self).__init__()
        # hardcoded settings from "config.json" file
        self.config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1x1_config")
        self.cityflow = cityflow.Engine(os.path.join(self.config_dir, "config.json"), thread_num=1)
        self.intersection_id = "intersection_1_1"

        self.sec_per_step = 1.0

        self.steps_per_episode = 2000
        self.current_step = 0
        self.sequence_index = 0
        

        self.sequence = [1, 3, 0, 2, 4, 0]
        # self.sequence = [1, 3, 2, 4]

        # self.min_green_time = 15
        # self.min_met = False

        self.zero_light_counter = 0
        

        self.is_done = False
        self.reward_range = (-float('inf'), float('inf'))
        self.start_lane_ids = \
            ["road_0_1_0_0",
             "road_0_1_0_1",
             "road_1_0_1_0",
             "road_1_0_1_1",
             "road_2_1_2_0",
             "road_2_1_2_1",
             "road_1_2_3_0",
             "road_1_2_3_1"]

        self.all_lane_ids = \
            ["road_0_1_0_0",
             "road_0_1_0_1",
             "road_1_0_1_0",
             "road_1_0_1_1",
             "road_2_1_2_0",
             "road_2_1_2_1",
             "road_1_2_3_0",
             "road_1_2_3_1",
             "road_1_1_0_0",
             "road_1_1_0_1",
             "road_1_1_1_0",
             "road_1_1_1_1",
             "road_1_1_2_0",
             "road_1_1_2_1",
             "road_1_1_3_0",
             "road_1_1_3_1"]

        """
        road id:
        ["road_0_1_0",
         "road_1_0_1",
         "road_2_1_2",
         "road_1_2_3",
         "road_1_1_0",
         "road_1_1_1",
         "road_1_1_2",
         "road_1_1_3"]
         
        start road id:
        ["road_0_1_0",
        "road_1_0_1",
        "road_2_1_2",
        "road_1_2_3"]
        
        lane id:
        ["road_0_1_0_0",
         "road_0_1_0_1",
         "road_1_0_1_0",
         "road_1_0_1_1",
         "road_2_1_2_0",
         "road_2_1_2_1",
         "road_1_2_3_0",
         "road_1_2_3_1",
         "road_1_1_0_0",
         "road_1_1_0_1",
         "road_1_1_1_0",
         "road_1_1_1_1",
         "road_1_1_2_0",
         "road_1_1_2_1",
         "road_1_1_3_0",
         "road_1_1_3_1"]
         
         start lane id:
         ["road_0_1_0_0",
         "road_0_1_0_1",
         "road_1_0_1_0",
         "road_1_0_1_1",
         "road_2_1_2_0",
         "road_2_1_2_1",
         "road_1_2_3_0",
         "road_1_2_3_1"]
        """
        self.action_mode = "disc"
        self.reward_wait_time = True
        self.mode = "all_all"
        assert self.mode == "all_all" or self.mode == "start_waiting", "mode must be one of 'all_all' or 'start_waiting'"
        """
        `mode` variable changes both reward and state.
        
        "all_all":
            - state: waiting & running vehicle count from all lanes (incoming & outgoing)
            - reward: waiting vehicle count from all lanes
            
        "start_waiting" - 
            - state: only waiting vehicle count from only start lanes (only incoming)
            - reward: waiting vehicle count from start lanes
        """
        """
        if self.mode == "all_all":
            self.state_space = len(self.all_lane_ids) * 2

        if self.mode == "start_waiting":
            self.state_space = len(self.start_lane_ids)
        """
        
        if self.action_mode == "disc":
            self.action_space = spaces.Discrete(5)
        else:
            self.action_space = spaces.Box(low=np.array([-20]),high=np.array([20]))

        if self.mode == "all_all":
            self.observation_space = spaces.MultiDiscrete([100]*32 + [5])
        else:
            self.observation_space = spaces.MultiDiscrete([100]*8 + [5])
    
    def vehicle_dict(self, vehicle_speeds, vehicle_distances):
        # bug if we include vehicles in waiting buffer
        for vehicle in self.cityflow.get_vehicles(include_waiting=False):
            speed = vehicle_speeds[vehicle]
            distance = vehicle_distances[vehicle]
            leader = self.cityflow.get_leader(vehicle)
            
            if vehicle in self.vehicle_wait_times and (speed == 0 or (leader and distance < 5)):
                self.vehicle_wait_times[vehicle] += self.sec_per_step
            elif vehicle not in self.vehicle_wait_times and (speed == 0 or (leader and distance < 5)):
                self.vehicle_wait_times[vehicle] = self.sec_per_step
            elif vehicle in self.vehicle_wait_times and (speed > 0 and not (leader and distance < 5)):
                del self.vehicle_wait_times[vehicle]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        if self.action_mode == "disc":
            self.steps_since_change += 1

            # Only change the light if the minimum green time has passed
            if self.steps_since_change >= 7:
                self.current_light = action
                if(self.previous_light != self.current_light):
                    # print(f"steps_since_change: {self.steps_since_change}   | current_light: {self.current_light}")
                    self.previous_light = self.current_light
                    self.steps_since_change = 0

                self.cityflow.set_tl_phase(self.intersection_id, self.previous_light)
        else:
            # Continuous
            self.steps_since_change += 1

            # If light phase '0' is active, count the steps
            if self.current_light == 0:
                self.zero_light_counter += 1
            else:
                self.zero_light_counter = 0  # Reset counter when other light phases are active

            # Check if we need to switch the light
            # If current light is 0 and zero_light_counter is 5, force a switch
            # Else, switch when phase_duration reaches 0
            if (self.current_light == 0 and self.zero_light_counter >= 5) or (self.current_light != 0 and self.phase_duration <= 0):

                print(f"steps_since_change: {self.steps_since_change}   | current_light: {self.current_light}")
                value = self.sequence[self.sequence_index]
                self.sequence_index = (self.sequence_index + 1) % len(self.sequence)

                self.current_light = value
                self.phase_duration = 10
                self.steps_since_change = 0

            # Decrease phase_duration by action only if current light is not 0
            if self.current_light != 0:
                self.phase_duration -= action

            self.cityflow.set_tl_phase(self.intersection_id, self.current_light)


        # Transition to the next state
        self.cityflow.next_step()

        state = self._get_state()
        reward = self._get_reward()
        self.current_step += 1

        if self.is_done:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. "
                        "You should always call 'reset()' once you receive 'done = True' "
                        "-- any further steps are undefined behavior.")
            reward = 0.0

        if self.current_step + 1 == self.steps_per_episode:
            print("\nFinal state: ", state, "\n")
            lane_waiting_vehicles_dict = self.cityflow.get_lane_waiting_vehicle_count()
            print("\nFinal Queue length: ", lane_waiting_vehicles_dict)
            self.is_done = True

        return state, reward, self.is_done, {}


    def reset(self):
        print("Reset called", self.current_step, "\n")
        # traceback.print_stack()
        self.vehicle_wait_times = {}
        self.cityflow.reset()
        self.is_done = False
        self.current_step = 0

        self.steps_since_change = 0
        self.current_light = 0
        self.previous_light = 0

        self.phase_duration = 30

        return self._get_state()

    def render(self, mode='human'):
        print("Current time: " + self.cityflow.get_current_time())

    def _get_state(self):
        lane_vehicles_dict = self.cityflow.get_lane_vehicle_count()
        lane_waiting_vehicles_dict = self.cityflow.get_lane_waiting_vehicle_count()

        state = None

        if self.mode=="all_all":
            state = np.zeros(len(self.all_lane_ids) * 2, dtype=np.float32)
            for i in range(len(self.all_lane_ids)):
                state[i*2] = lane_vehicles_dict[self.all_lane_ids[i]]
                state[i*2 + 1] = lane_waiting_vehicles_dict[self.all_lane_ids[i]]
        
                

        if self.mode=="start_waiting":
            state = np.zeros(len(self.start_lane_ids), dtype=np.float32)
            for i in range(len(self.start_lane_ids)):
                state[i] = lane_waiting_vehicles_dict[self.start_lane_ids[i]]
        
        state = np.append(state, self.current_light)

        return state

    def _get_reward(self):
        # Get the count of waiting vehicles
        lane_waiting_vehicles_dict = self.cityflow.get_lane_waiting_vehicle_count()
        lane_vehicles_dict = self.cityflow.get_lane_vehicle_count()
        
        # Define the trade-off coefficients
        b1 = 1.5  # Adjust as needed
        b2 = 1.0  # Adjust as needed

        # Calculate queue length (sum of waiting vehicles)
        # q = sum(lane_waiting_vehicles_dict.values())
        # # Calculate vehicle throughput (sum of all vehicles)
        # n = sum(lane_vehicles_dict.values())
        # # Time is simply the sec_per_step
        # T = self.sec_per_step
        # # Compute reward
        # reward = -b1 * q + (b2 * n / T)



        # # Calculate average queue length (average of waiting vehicles)
        q_avg = sum(lane_waiting_vehicles_dict.values()) / len(lane_waiting_vehicles_dict)
        # Calculate average vehicle throughput (average of all vehicles)
        n_avg = sum(lane_vehicles_dict.values()) / len(lane_vehicles_dict)
        # Time is simply the sec_per_step
        T = self.sec_per_step
        # Compute reward with averages
        reward = -b1 * q_avg + (b2 * n_avg / T)

        threshold = 15
        switch_penalty_weight = 1.5
        # Add a penalty for switching too quickly
        if self.current_light == 0:
            pass
        elif (0 < self.steps_since_change < threshold) :  # Skip this in the very first step
            switch_penalty = switch_penalty_weight / self.steps_since_change
            # print("Switch penalty: ", switch_penalty)
            reward -= (switch_penalty)
        
        # print("Reward: ", reward)
        return reward 





    def set_replay_path(self, path):
        self.cityflow.set_replay_file(path)

    def seed(self, seed=None):
        self.cityflow.set_random_seed(seed)

    def get_path_to_config(self):
        return self.config_dir

    def set_save_replay(self, save_replay):
        self.cityflow.set_save_replay(save_replay)