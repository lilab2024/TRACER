import numpy as np
np.random.seed(42)
import pandas as pd
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

def OVRV_Accel(params, v, v_l, s):
    k1 = params[0]
    k2 = params[1]
    eta = params[2]
    tau = params[3]
    delta_V = v_l - v
    accel = k1 * (s - eta - tau * v) + k2 * (v_l - v)
    return accel

def makeSim_ring_mix(numSteps, numVeh, params_IDM, params_OVRV, delta_T, initVel_Follower, initGap, attacked_Vehicles):
    spacing_gap_Profile = np.zeros((numSteps, numVeh))
    spacing_gap_Profile[0, :] = np.squeeze(initGap)
    velProfile_Follower = np.zeros((numSteps, numVeh))
    velProfile_Follower[0, :] = initVel_Follower
    accelVal = np.zeros((numSteps, numVeh))
    sigma = 0.5
    for t in range(1, numSteps):
        for i in range(numVeh):
            noise = norm.rvs(0, sigma)
            noi_attack = norm.rvs(0, 5)
            if i == 0:
                accelVal[t - 1, i] = IDM_Accel(params_IDM[i, :], velProfile_Follower[t - 1, i], velProfile_Follower[t - 1, numVeh - 1], spacing_gap_Profile[t - 1, i]) + noise
                accelVal[t - 1, i] = max(min(accelVal[t - 1, i], 2), -5)
                velProfile_Follower[t, i] = max(velProfile_Follower[t - 1, i] + accelVal[t - 1, i] * delta_T, 0.5)
                spacing_gap_Profile[t, i] = max(spacing_gap_Profile[t - 1, i] + (velProfile_Follower[t - 1, numVeh - 1] - velProfile_Follower[t - 1, i]) * delta_T, 0)
            elif attacked_Vehicles[0, i] == 2:
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

def OVRV_Accel_attack(params, v, v_l, s, slope_1, slope_2):
    k1 = params[0]
    k2 = params[1]
    eta = params[2]
    tau = params[3]
    delta_V = v_l - v
    attack_spacing = slope_1 * s
    attack_speed_relative = slope_2 * delta_V
    accel = k1 * (s + attack_spacing - eta - tau * v) + k2 * (v_l - v + attack_speed_relative)
    return accel
import math
noise_attack = norm.rvs(0, 5)

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
            noise_attack = norm.rvs(0, 5)
            if i == 0:
                accelVal[t - 1, i] = IDM_Accel(params_IDM[i, :], velProfile_Follower[t - 1, i], velProfile_Follower[t - 1, numVeh - 1], spacing_gap_Profile[t - 1, i]) + noise
                accelVal[t - 1, i] = max(min(accelVal[t - 1, i], 2), -5)
                velProfile_Follower[t, i] = max(velProfile_Follower[t - 1, i] + accelVal[t - 1, i] * delta_T, 0.5)
                spacing_gap_Profile[t, i] = max(spacing_gap_Profile[t - 1, i] + (velProfile_Follower[t - 1, numVeh - 1] - velProfile_Follower[t - 1, i]) * delta_T, 0)
            elif attacked_Vehicles[0, i] == 2:
                if attack_start_time <= t < attack_end_time:
                    accelVal[t - 1, i] = IDM_Accel(params_IDM[i, :], velProfile_Follower[(math.floor((t - 1) / 10) - 9) * 10, i - 1], velProfile_Follower[(math.floor((t - 1) / 10) - 9) * 10, i - 1 - 1], spacing_gap_Profile[(math.floor((t - 1) / 10) - 9) * 10, i - 1])
                    accelVal[t - 1, i] = max(min(accelVal[(math.floor((t - 1) / 10) - 9) * 10, i - 1], 2), -5)
                    velProfile_Follower[t, i] = max(velProfile_Follower[(math.floor((t - 1) / 10) - 9) * 10, i - 1] + accelVal[(math.floor((t - 1) / 10) - 9) * 10, i - 1] * delta_T, 0.5)
                    spacing_gap_Profile[t, i] = max(spacing_gap_Profile[(math.floor((t - 1) / 10) - 9) * 10, i - 1] + (velProfile_Follower[(math.floor((t - 1) / 10) - 9) * 10, i - 1 - 1] - velProfile_Follower[(math.floor((t - 1) / 10) - 9) * 10, i - 1]) * delta_T, 0)
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
import numpy as np
L = 300
numVeh = 6
numSteps = 10 * 60 * 8
deltaT = 1 / 10
time_series = np.arange(0, numSteps * deltaT, deltaT)
car_length = 5
ACC_params = [39.12, 0.96, 10.0, 20.0, 1.18, 8.9]
Human_params = [30, 1.26, 3.4, 4, 1.06, 2]
Theoretical_ACC_params = [33.34, 1.63, 2.02, 4.02, 2.01, 8.97]
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
(simSpacing, simFollowVel, simAcc) = makeSim_ring_mix_attack(numSteps, numVeh, IDM_parameters, OVRV_parameters, deltaT, initial_velocity, S_E, attacked_Vehicles, attack_start_time, attack_end_time, slope_1, slope_2)
last_vehicle_spacing = simSpacing[:, numVeh - 1]
remaining_vehicle_spacing = simSpacing[:, :numVeh - 1]
simSpacing_reshaped = np.hstack((last_vehicle_spacing[:, None], remaining_vehicle_spacing))
cumulative_spacing = np.cumsum(simSpacing_reshaped, axis=1)
non_negative_follow_vel = np.maximum(simFollowVel, 0)
first_vehicle_vel = non_negative_follow_vel[:, 0]
first_vehicle_travel_dist = first_vehicle_vel * deltaT
cumulative_first_vehicle_dist = np.cumsum(first_vehicle_travel_dist)
all_vehicle_cumulative_dist = np.tile(cumulative_first_vehicle_dist, (numVeh, 1)).T
position_vehicles_2_to_20 = all_vehicle_cumulative_dist - cumulative_spacing
first_vehicle_position = position_vehicles_2_to_20[:, numVeh - 1]
remaining_vehicles_position = position_vehicles_2_to_20[:, :numVeh - 1]
combined_positions = np.hstack((first_vehicle_position[:, None], remaining_vehicles_position))
travel_distance_all_vehicles = simFollowVel * deltaT
cumulative_travel_distance_all_vehicles = np.cumsum(travel_distance_all_vehicles, axis=0)
initial_position = L
spacing_step_length = S_E[0]
number_of_vehicles = numVeh
initial_vehicle_positions = initial_position - spacing_step_length * np.arange(number_of_vehicles)
initial_positions_all_steps = np.tile(initial_vehicle_positions, (numSteps, 1))
pos_new = cumulative_travel_distance_all_vehicles + initial_positions_all_steps
import numpy as np
vehicle_idx = 1
lead_idx = 0
numSteps = simFollowVel.shape[0]
output_file = 'ma_acc_1_raw.txt'
with open(output_file, 'w') as f:
    f.write(f"{'Time':>10} {'Speed_Follower':>10} {'Relative_speed':>10} {'Spacing':>10}\n")
    for t in range(numSteps):
        velocity = simFollowVel[t, vehicle_idx]
        relative_speed = velocity - simFollowVel[t, lead_idx]
        spacing = simSpacing[t, vehicle_idx]
        f.write(f'{t:10d} {velocity:10.3f} {relative_speed:10.3f} {spacing:10.3f}\n')
print(f'第2辆车的数据已保存到 {output_file}，包含时间步、速度、相对速度和间距，列对齐')
output_file_leader = 'ma_leader_0_raw.txt'
with open(output_file_leader, 'w') as f:
    f.write(f"{'Time':>10} {'Speed_Leader':>15}\n")
    for t in range(numSteps):
        v_leader = simFollowVel[t, lead_idx]
        f.write(f'{t:10d} {v_leader:15.3f}\n')
print(f'✅ 前车数据已保存到 {output_file_leader}，包含时间步和速度。')
