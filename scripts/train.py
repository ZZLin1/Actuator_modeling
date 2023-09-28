from utils import train_actuator_network_and_plot_predictions
from glob import glob

log_dir_root = "../logs/"
log_dir = "XY"
load_pretrained_model = True # True:评估 False:训练

if load_pretrained_model:
    actuator_network_path = "../actuator_nets/XY/unitree_XY_1500.pt"
else:
    actuator_network_path = "../actuator_nets/XY/unitree_XY.pt"

log_dirs = glob(f"{log_dir_root}{log_dir}/", recursive=True)

if len(log_dirs) == 0: raise FileNotFoundError(f"No log files found in {log_dir_root}{log_dir}/")

for log_dir in log_dirs:
    try:
        train_actuator_network_and_plot_predictions(log_dir[:11], log_dir[11:],
                                                    actuator_network_path=actuator_network_path,
                                                    load_pretrained_model=load_pretrained_model)
    except FileNotFoundError:
        print(f"Couldn't find log.pkl in {log_dir}")
    except EOFError:
        print(f"Incomplete log.pkl in {log_dir}")


# 8e-4 1500 4.7016 4.6916 1.2962
# 1.5e-3 500 4.7528 4.8126 1.3418
# 2.5e-3 500 4.7637 4.7883 1.3172