import numpy as np
import json
import random

def create_flow(lower_volume, upper_volume, num_intervals, interval_duration, total_steps, seed=0, distribution='normal'):
    np.random.seed(seed)
    random.seed(seed)

    intervals = np.round(np.linspace(0, total_steps, num_intervals)).astype(int)
    if distribution == 'normal':
        mean_volume = (lower_volume + upper_volume) / 2
        std_dev_volume = (upper_volume - lower_volume) / 4
        volumes = np.random.normal(loc=mean_volume, scale=std_dev_volume, size=num_intervals)
        volumes = np.maximum(volumes, 0).astype(int)
    elif distribution == 'uniform':
        volumes = [random.randint(lower_volume, upper_volume) for _ in range(num_intervals)]

    routes = [
        ["road_2_1_2", "road_1_1_2"],
        ["road_2_1_2", "road_1_1_3"],
        ["road_0_1_0", "road_1_1_0"],
        ["road_0_1_0", "road_1_1_1"],
        ["road_1_0_1", "road_1_1_1"],
        ["road_1_0_1", "road_1_1_2"],
        ["road_1_2_3", "road_1_1_3"],
        ["road_1_2_3", "road_1_1_0"]
    ]

    flows = []
    for i, volume in enumerate(volumes):
        for _ in range(volume):
            route = random.choice(routes)
            flow = {
                "vehicle": {
                    "length": 5.0,
                    "width": 2.0,
                    "maxPosAcc": 2.0,
                    "maxNegAcc": 4.5,
                    "usualPosAcc": 2.0,
                    "usualNegAcc": 4.5,
                    "minGap": 2.5,
                    "maxSpeed": 11.11,
                    "headwayTime": 2.0
                },
                "route": route,
                "interval": 5,
                "startTime": int(intervals[i]),
                "endTime":int(intervals[i])
            }
            flows.append(flow)

    with open('flow.json', 'w') as json_file:
        json.dump(flows, json_file, indent=4)

# Set your desired parameters
vehicles_per_hour = 1000
steps_per_hour = 3600
total_steps = steps_per_hour  # assuming you're running the simulation for 1 hour

# Calculate parameters for create_flow function
# Assume the traffic is evenly distributed over the hour
# Thus, one vehicle is generated approximately every interval_duration steps
interval_duration = total_steps / vehicles_per_hour 

# We will generate one vehicle per interval, so the number of intervals is the total number of vehicles
num_intervals = vehicles_per_hour

# Choose a reasonable range for the uniform distribution or a mean and std dev for the normal distribution
# If using uniform distribution
lower_volume = 1
upper_volume = 1
distribution = 'uniform'

create_flow(lower_volume, upper_volume, num_intervals=int(num_intervals), interval_duration=int(interval_duration), total_steps=total_steps, seed=0, distribution=distribution)
