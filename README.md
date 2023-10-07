# Actuator_modeling
# HIT zzl

1.Data Processing
```python logs/csv2pkl.py```
    - The data is stored in `logs/<robot_name>/<data>`
    - Process .csv as .pkl
2.Train
```python scripts/train.py```
    - To train/eval an actuator network
    - The trained policy is saved in `actuator/<robot_name>/<model>.pt`