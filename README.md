## Action-conditional Scene Rendering for RL agents in 3D Games



<p align="center"> 
<img height="250" width="500" src="assets/pyramids_render.gif">
</p>

<p align="center"> 
Left: ground truth visual observations. Right: rendered observations conditioned on the agent's actions. 
</p>

For AI to learn complex vision tasks in the 3D world, an effective representation of the 3D visual environment is critical. However, it's often hard to train representations of complex 3D environments based on visual inputs, and agents trained on raw pixels often do not perform well.  One common alternative is to replace visual inputs with sensor inputs, and another is to train a separate network to transform visual inputs to handcrafted features. These appoarches, combined with model-free reinforcement learning algorithms, can achieve good results, but are unlikely to scale to more complex visual environments.  

Humans (and other animals), on the other hand, use intrinsic predictive models of the 3D environment to navigate in the real world. Such models not only encode the current visual inputs but are also able to predict what future scenes would look like after performing different actions. Here, we develops a predictive vision model for agents to effectively encode 3D game environments and render future scenes conditioned on current actions. 

For game engine and training data, we use 3D Unity games from the the [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). Game builds and training data can be found [here](https://github.com/yueqiw/gqn-world-model/releases/). Example notebook for scene rendering can be found [here](notebooks/unity_predict_pyramids_video.ipynb). 

This work leverages the [Generative query network (GQN)](https://deepmind.com/blog/neural-scene-representation-and-rendering/) and ideas from [World Models](https://worldmodels.github.io). GQN provides a novel approach for learning robust representation conditional rendering of 3D environments without human labeling. World Models, as the name suggests, is the idea to learn a dynamic model of the agent's game environment, and is a promising direction in model-based reinforcement learning. 

<p align="center"> 
<img height="360" src="assets/model_flow_unity.png">
</p>

<p align="center"> 
The qenrative query network for action-conditional scene rendering
</p>

### Credits:

@yueqiw @GilbertZhang @DorisHYC

GQN code template: [generative-query-network-pytorch](<https://github.com/wohlert/generative-query-network-pytorch>)

This project started during the COMS4995 Deep Learning course at Columbia, advised by Prof. Iddo Drori. 

