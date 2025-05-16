# algorithm_parecon.py
import numpy as np
import subprocess

def algorithm_parecon(V, t, Delta_t, Total_queue, Total_general_info, Total_model_info, mobile_GPU):
    queue_F = Total_queue[t, 0]
    queue_G = Total_queue[t, 1]
    queue_H = Total_queue[t, 2]

    kappa_val = 1 / (queue_H / V) - 0.05
    kappa = min(max(kappa_val, 0), 1)
    
    input_frames = Total_general_info['Total_general_info'][0, 1][t]
    uplink_network = Total_general_info['Total_general_info'][1, 1]['uplink_network'][t]

    uplink_speed_mbit = uplink_network.flatten()[0] / 1e6
    uplink_speed_mbit = uplink_speed_mbit[0][0]
    uplink_speed_str = f"{uplink_speed_mbit}mbit"

    cmd = [
        'sudo', 'tc', 'class', 'change', 'dev', 'wlan0',
        'parent', '1:1', 'classid', '1:11', 'htb',
        'rate', uplink_speed_str
    ]
    
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Network changed to {uplink_speed_str}.")
    except subprocess.CalledProcessError as e:
        print(f"네트워크 속도 변경에 실패했습니다: {e}")
    
    
    
    server_GPU = Total_general_info['Total_general_info'][2,1][1][1][0,0]
    w = 14159416
    mobile_rho = Total_model_info['Total_model_info'][0, 1]
    server_rho = Total_model_info['Total_model_info'][1, 1]
    mobile_sigma = Total_model_info['Total_model_info'][2, 1]
    partition_point = len(mobile_rho)
    resizing_factor_number = 10
    resizing_factors = Total_general_info['Total_general_info'][3,1]
    
    def IBS(iter_i, b_optimal, iter_s, iter_s_number, queue_F, queue_G, queue_H, w, Delta_t, mobile_rho, mobile_sigma, mobile_GPU, server_GPU, uplink_value, input_frames):
        term1 = (w * iter_s * mobile_rho[iter_i, iter_s_number]) / mobile_GPU        
        term2 = (w * iter_s * mobile_sigma[iter_i, iter_s_number] * Delta_t) / uplink_value
        term3 = (w * iter_s * server_rho[iter_i, iter_s_number]) / server_GPU
        F_component = 1.8 * queue_F * (term1 + term2 + term3)
        
        if iter_i == 9 or iter_i == 0:
            F_component = queue_F * (term1 + term2 + term3)
        
        accuracy_t = Total_general_info['Total_general_info'][6,1][iter_s_number][0]
                
        G_component = queue_G * accuracy_t
        H_component = queue_H * (b_optimal / input_frames)
        return F_component - 2 * G_component - H_component

    x = np.inf
    IBS_values = np.zeros((partition_point, resizing_factor_number))
    for iter_i in range(partition_point):
        for iter_s_number in range(resizing_factor_number):
            iter_s = resizing_factors[iter_s_number][0]
            uplink_value = uplink_network[0][0][0]
            b_optimal = np.floor(np.min([
                input_frames[0],
                0.7 * uplink_value / (w * iter_s * mobile_sigma[iter_i, iter_s_number]),
                0.7 * (mobile_GPU * Delta_t) / (w * iter_s * mobile_rho[iter_i, iter_s_number])
                #temp = (w * iter_s * mobile_rho[iter_i, iter_s_number])
                    
            ]))
            if iter_i == 9 or iter_i == 0:
                b_optimal = np.floor(np.min([
                input_frames[0],
                uplink_value / (w * iter_s * mobile_sigma[iter_i, iter_s_number]),
                (mobile_GPU * Delta_t) / (w * iter_s * mobile_rho[iter_i, iter_s_number])
            ]))
            IBS_result = IBS(iter_i, b_optimal, iter_s, iter_s_number, queue_F, queue_G, queue_H, w, Delta_t,
                             mobile_rho, mobile_sigma, mobile_GPU, server_GPU, uplink_value, input_frames)
            IBS_values[iter_i, iter_s_number] = IBS_result
            
            
            if IBS_result < x:
                x = IBS_result
                optimal_values = {
                    'IBS_result': IBS_result,
                    'input_frames': input_frames[0],
                    'uplink_value': uplink_value,
                    'kappa': kappa,
                    'point': iter_i,
                    'processed_fps': b_optimal,
                    'resizing_factor': iter_s,
                    'resizing_number': iter_s_number,
                    'mobile_process_density': mobile_rho[iter_i, iter_s_number],
                    'server_process_density': server_rho[iter_i, iter_s_number],
                    'bit_conversion_ratio': mobile_sigma[iter_i, iter_s_number]
                }

    return optimal_values
