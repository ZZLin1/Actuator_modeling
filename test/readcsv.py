import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csv_file = '../logs/A1/rlc_hardware_2023-05-23-21-19-06-hexapod-joint_state_fdb.csv'
data = pd.read_csv(csv_file)

plt.plot(data['.header.stamp.secs'], data['.position'][0])
plt.show()

