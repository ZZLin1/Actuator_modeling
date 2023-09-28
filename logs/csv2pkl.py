import pandas as pd
import pickle
import numpy as np

# 读取CSV文件
df = pd.read_csv('XY/hexapod-joint_state.csv').astype(str)
# 定义新顺序
new_order = [15, 16, 17, 9, 10, 11, 12, 13, 14, 6, 7, 8, 0, 1, 2, 3, 4, 5]
# 将数据转换为字典列表
data_list = []
for _, row in df.iterrows():
    pos_cmd_str = row['pos_cmd']
    pos_fdb_str = row['pos_fdb']
    dof_vel_str = row['dof_vel']
    torque_str = row['torque']
    # 将字符串转换为NumPy数组
    pos_cmd_array = np.array([float(val) for val in pos_cmd_str.strip('()').split(', ')])
    pos_fdb_array = np.array([float(val) for val in pos_fdb_str.strip('()').split(', ')])
    dof_vel_array = np.array([float(val) for val in dof_vel_str.strip('()').split(', ')])
    torque_array = np.array([float(val) for val in torque_str.strip('()').split(', ')])
    # 调整顺序
    pos_cmd_array = pos_cmd_array[new_order]
    pos_fdb_array = pos_fdb_array[new_order]
    dof_vel_array = dof_vel_array[new_order]
    torque_array = torque_array[new_order]

    data_dict = {
        # 'time': row['.header.stamp.nsecs']*0.000000001,
        'joint_pos_target': pos_cmd_array,
        'joint_pos': pos_fdb_array,
        'joint_vel': dof_vel_array,
        'tau_est': torque_array
    }
    data_list.append(data_dict)

# 将字典列表保存为pkl文件
with open('XY/log.pkl', 'wb') as pkl_file:
    pickle.dump(data_list, pkl_file)
