# Actuator_modeling

1.Data Processing
```python scripts/csv2pkl.py```
    - The data is stored in `logs/<robot_name>/<data>`
    - Process .csv as .pkl
2.Train
```python scripts/train.py```
    - To train/eval an actuator network
    - The trained policy is saved in `actuator/<robot_name>/<model>.pt`


# Citation
If you use this code, please cite the following paper:

```bibtex
@ARTICLE{10752370,
  author={Zhang, Zhelin and Liu, Tie and Ding, Liang and Wang, Haoyu and Xu, Peng and Yang, Huaiguang and Gao, Haibo and Deng, Zongquan and Pajarinen, Joni},
  journal={IEEE Robotics and Automation Letters}, 
  title={Imitation-Enhanced Reinforcement Learning With Privileged Smooth Transition for Hexapod Locomotion}, 
  year={2025},
  volume={10},
  number={1},
  pages={350-357},
  doi={10.1109/LRA.2024.3497754}}