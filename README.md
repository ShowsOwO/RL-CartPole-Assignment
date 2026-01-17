# DQN CartPole-v1 Project

## 1. 项目介绍 (Introduction)
本项目是《人工智能》课程综合作业（方向B：强化学习项目）。
项目基于 Deep Q-Network (DQN) 算法，使用 PyTorch 框架在 Gymnasium 的 `CartPole-v1` 环境中训练智能体，使其能够控制倒立摆保持平衡。

## 2. 环境依赖 (Dependencies)
* **Python 版本**: 3.8+
* **核心库**: PyTorch, Gymnasium, Numpy, Matplotlib

安装依赖命令：
```bash
pip install -r requirements.txt
```
## 3. 运如何行 (Usage)
本项目包含完整的训练与可视化测试流程。
启动命令：
```bash
python train.py
```

* **运行流程说明**:

**1.**
程序首先会在后台进行 250 个 Episode 的训练。

**2.**
训练完成后，会自动保存模型权重至 dqn_cartpole_model.pth 并绘制奖励曲线。

**3.**
终端会提示是否进行可视化演示，输入 y 即可查看智能体控制平衡车的动态效果。

## 4. 参考引用(References)
**本项目参考了以下开源资源及文献**：

[1] DQN Algorithm: Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature.

[2] Gymnasium Documentation: https://gymnasium.farama.org/

[3] DQN深度强化学习：CartPole倒立摆任务 https://zhuanlan.zhihu.com/p/21975146686

代码框架参考：基于 PyTorch 的 DQN 基础实现。