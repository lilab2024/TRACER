import importlib
import numpy as np
import pandas as pd


def simulate(scenario):
    spacing, speed = importlib.import_module(f'simulators.{scenario}').simulate()
    target = pd.DataFrame({'Time': np.arange(len(speed)),
                           'Speed_Follower': speed[:, 1],
                           'Relative_speed': speed[:, 1] - speed[:, 0],
                           'Spacing': spacing[:, 1]})
    leader = pd.DataFrame({'Time': np.arange(len(speed)), 'Speed_Leader': speed[:, 0]})
    return target.round(3), leader.round(3)
