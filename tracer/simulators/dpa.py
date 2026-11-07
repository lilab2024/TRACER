"""Simulation extracted without changing numerical functions or random draws."""
import math
import numpy as np
from scipy.stats import norm

def IDM_Accel(params, v, v_l, s):
    v_0 = params[0]
    T = params[1]
    s_0 = params[2]
    delta = params[3]
    a = params[4]
    b = params[5]
    delta_V = v_l - v
    s_star = s_0 + max([0.0, v * T - v * delta_V / (2 * np.sqrt(a * b))])
    accel = a * (1 - (v / v_0) ** delta - (s_star / max(s, 0.1)) ** 2)
    return accel

def makeSim_ring_mix_attack(numSteps, numVeh, params_IDM, params_OVRV, delta_T, initVel_Follower, initGap, attacked_Vehicles, attack_start_time, attack_end_time, slope_1, slope_2):
    spacing_gap_Profile = np.zeros((numSteps, numVeh))
    spacing_gap_Profile[0, :] = np.squeeze(initGap)
    velProfile_Follower = np.zeros((numSteps, numVeh))
    velProfile_Follower[0, :] = initVel_Follower
    accelVal = np.zeros((numSteps, numVeh))
    sigma = 0.5
    for t in range(1, numSteps):
        for i in range(numVeh):
            noise = norm.rvs(0, sigma)
            if i > 0:
                real_rel_speed = velProfile_Follower[t - 1, i - 1] - velProfile_Follower[t - 1, i]
                real_spacing = spacing_gap_Profile[t - 1, i]
            else:
                real_rel_speed = 0.0
                real_spacing = spacing_gap_Profile[t - 1, i]
            if i == 0:
                accelVal[t - 1, i] = IDM_Accel(params_IDM[i, :], velProfile_Follower[t - 1, i], velProfile_Follower[t - 1, numVeh - 1], spacing_gap_Profile[t - 1, i]) + noise
                accelVal[t - 1, i] = max(min(accelVal[t - 1, i], 2), -5)
                velProfile_Follower[t, i] = max(velProfile_Follower[t - 1, i] + accelVal[t - 1, i] * delta_T, 0.5)
                spacing_gap_Profile[t, i] = max(spacing_gap_Profile[t - 1, i] + (velProfile_Follower[t - 1, numVeh - 1] - velProfile_Follower[t - 1, i]) * delta_T, 0)
            elif attacked_Vehicles[0, i] == 2:
                if attack_start_time <= t < attack_end_time:
                    lambda1 = norm.rvs(0, 5)
                    lambda2 = norm.rvs(0, 5)
                    noisy_spacing = real_spacing + lambda1
                    perceived_leader_speed = velProfile_Follower[t - 1, i - 1] + lambda2
                    accelVal[t - 1, i] = IDM_Accel(params_IDM[i, :], velProfile_Follower[t - 1, i], perceived_leader_speed, noisy_spacing)
                    accelVal[t - 1, i] = max(min(accelVal[t - 1, i], 2), -5)
                    velProfile_Follower[t, i] = max(velProfile_Follower[t - 1, i] + accelVal[t - 1, i] * delta_T, 0.5)
                    spacing_gap_Profile[t, i] = max(real_spacing + (perceived_leader_speed - velProfile_Follower[t - 1, i]) * delta_T, 0)
                else:
                    accelVal[t - 1, i] = IDM_Accel(params_IDM[i, :], velProfile_Follower[t - 1, i], velProfile_Follower[t - 1, i - 1], spacing_gap_Profile[t - 1, i])
                    accelVal[t - 1, i] = max(min(accelVal[t - 1, i], 2), -5)
                    velProfile_Follower[t, i] = max(velProfile_Follower[t - 1, i] + accelVal[t - 1, i] * delta_T, 0.5)
                    spacing_gap_Profile[t, i] = max(spacing_gap_Profile[t - 1, i] + (velProfile_Follower[t - 1, i - 1] - velProfile_Follower[t - 1, i]) * delta_T, 0)
            elif attacked_Vehicles[0, i] == 1:
                accelVal[t - 1, i] = IDM_Accel(params_IDM[i, :], velProfile_Follower[t - 1, i], velProfile_Follower[t - 1, i - 1], spacing_gap_Profile[t - 1, i])
                accelVal[t - 1, i] = max(min(accelVal[t - 1, i], 2), -5)
                velProfile_Follower[t, i] = max(velProfile_Follower[t - 1, i] + accelVal[t - 1, i] * delta_T, 0.5)
                spacing_gap_Profile[t, i] = max(spacing_gap_Profile[t - 1, i] + (velProfile_Follower[t - 1, i - 1] - velProfile_Follower[t - 1, i]) * delta_T, 0)
            else:
                accelVal[t - 1, i] = IDM_Accel(params_IDM[i, :], velProfile_Follower[t - 1, i], velProfile_Follower[t - 1, i - 1], spacing_gap_Profile[t - 1, i])
                accelVal[t - 1, i] = max(min(accelVal[t - 1, i], 2), -5)
                velProfile_Follower[t, i] = max(velProfile_Follower[t - 1, i] + accelVal[t - 1, i] * delta_T, 0.5)
                spacing_gap_Profile[t, i] = max(spacing_gap_Profile[t - 1, i] + (velProfile_Follower[t - 1, i - 1] - velProfile_Follower[t - 1, i]) * delta_T, 0)
    return (spacing_gap_Profile, velProfile_Follower, accelVal)

def simulate():
    np.random.seed(42)
    # Preserve original pre-simulation random draws, including unused values.
    L = 300
    numVeh = 6
    numSteps = 10 * 60 * 8
    deltaT = 1 / 10
    car_length = 5
    ACC_params = [39.12, 0.96, 10.0, 20.0, 1.18, 8.9]
    Human_params = [30, 1.26, 3.4, 4, 1.06, 2]
    ACC_indices = [1]
    Attack_indices = [1]
    IDM_parameters = np.tile(Human_params, (numVeh, 1))
    attacked_Vehicles = np.zeros((1, numVeh))
    total_vehicle_length = car_length * numVeh
    initial_spacing = (L - total_vehicle_length) / numVeh
    S_E = np.full((numVeh, 1), initial_spacing)
    initial_velocity = np.zeros((1, numVeh))
    attacked_Vehicles[0, ACC_indices] = 1
    attacked_Vehicles[0, Attack_indices] = 2
    IDM_parameters[ACC_indices, :] = ACC_params
    ACC_override_params = [0.08, 0.54, 1.03, 11.88, 1.06]
    OVRV_parameters = np.tile(ACC_override_params, (numVeh, 1))
    attack_start_time = 10 * 60 * 6
    attack_end_time = 10 * 60 * 6.5
    slope_params = [0, -0.1, -0.3, -0.5, -0.7, -0.9, -2, 2, 5, -5, -3, -1.5]
    slope_1 = slope_params[0]
    slope_2 = slope_params[0]
    simSpacing, simFollowVel, simAcc = makeSim_ring_mix_attack(numSteps, numVeh, IDM_parameters, OVRV_parameters, deltaT, initial_velocity, S_E, attacked_Vehicles, attack_start_time, attack_end_time, slope_1, slope_2)
    return simSpacing, simFollowVel
