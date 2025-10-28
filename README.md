# CathSim: An Open-source Simulator for Endovascular Intervention

### [[Project Page](https://airvlab.github.io/cathsim/)] [[Paper](https://arxiv.org/abs/2208.01455)]

<div align="center">
    <a href="https://"><img height="auto" src="/misc/cathsim_dn.gif"></a>
</div>

## Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Quickstart](#quickstart)
4. [Training](#training)
5. [Manual Control](#manual-control)
6. [Mesh Processing](#mesh-processing)
7. [Adding Elements](#adding-elements)

## Requirements

1. Ubuntu (tested with Ubuntu 22.04 LTS)
2. Miniconda (tested with Miniconda 23.5)
3. Python 3.9

If `miniconda` is not installed run the following for a quick Installation. Note: the script assumes you use `bash`.

```bash
# installing miniconda
mkdir -p ~/.miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh -O ~/.miniconda3/miniconda.sh
bash ~/.miniconda3/miniconda.sh -b -u -p ~/.miniconda3
rm -rf ~/.miniconda3/miniconda.sh
~/.miniconda3/bin/conda init bash
source .bashrc
```

## Installation

1. Create a `conda environment`:

```bash
conda create -n cathsim python=3.9
conda activate cathsim
```

2. Install the environment:

```bash
git clone git@github.com:airvlab/cathsim
cd cathsim
pip install -e .
```

## Quickstart

A quick way to have the environment run with gym is to make use of the `make_dm_env` function and then wrap the resulting environment into a `DMEnvToGymWrapper` resulting in a `gym.Env`.

```python
import cathsim.gym.envs
import gymnasium as gym

task_kwargs = dict(
    dense_reward=True,
    success_reward=10.0,
    delta=0.004,
    use_pixels=False,
    use_segment=False,
    image_size=64,
    phantom="phantom3",
    target="bca",
)

env = gym.make("cathsim/CathSim-v0", **task_kwargs)


obs = env.reset()
for _ in range(1):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    for obs_key in obs:
        print(obs_key, obs[obs_key].shape)
    print(reward)
    print(terminated)
    print(truncated)
    for info_key in info:
        print(info_key, info[info_key])
```

Being a `gym` interface, it is compatible with RL libraries such as `stable_baselines3`:

```python
import cathsim.gym.envs
from stable_baselines3 import SAC

model = SAC("MultiInputPolicy", "cathsim/CathSim-v0").learn(10000)
```

For a list of the environment libraries at the current time, see the accompanying `environment.yml`

## Training

In order to train the models available run:

```bash
bash ./scripts/train.bash
```

The script will create a `results` directory on the `cwd`. The script saves the data in `<trial-name>/<phantom>/<target>/<model>` format. Each model has three subfolders `eval`, `models` and `logs`, where the evaluation data contains the `Trajectory` data resulting from the evaluation of the policy, the `models` contains the `pytorch` models and the `logs` contains the `tensorboard` logs.

##### Path Comparison

<img width="1604" alt="path comparison between human and ENN" src="./misc/path.png">

## Manual Control

For a quick visualisation of the environment run:

```bash
run_env
```

You will now see the guidewire and the aorta along with the two sites that represent the targets. You can interact with the environment using the keyboard arrows.

## Mesh Processing

In order to use a custom aorta, it has to be processed using convex decomposition. This can be done using V-HACD or a preferred method. To do so, you can use stl2mjcf, available [here](https://github.com/tudorjnu/stl2mjcf). You can quickly install the tool with:

```bash
pip install git+git@github.com:tudorjnu/stl2mjcf.git
```

After the installation, you can use `stl2mjcf --help` to see the available commands. The resultant files can be then added to `cathsim/assets`. The `xml` will go in that folder and the resultant meshes folder will go in `cathsim/assets/meshes/`.

Note: You will probably have to change the parameters of V-HACD for the best results.

## Adding Elements

### Adding a Phantom

Following the steps from [mesh processing](#mesh-processing), the easiest way is to add the files to the correct directory, namely `src/cathsim/dm/components/phantom_assets/`. From here, you can simply select the phantom based on its name. For example, assuming your phantom is named `my_phantom.xml`, you would simply call:

```python
import cathsim.gym.envs
import gymnasium as gym

task_kwargs = dict(
    phantom="my_phantom",
    target=[0.1, 0.1, 0.1],  # select a target based on the mesh or embed it into the xml
)

env = gym.make("cathsim/CathSim-v0", **task_kwargs)
```

For more control, you could set the aorta using `mjcf`. See `src/cathsim/dm/components/phantom.py` and `src/cathsim/dm/components/base_models.py` for an example on how to do this. You can then just add the phantom to the task as such:

```python
from cathsim.gym.envs import CathSim

phantom = MyPhantom()
tip = Tip(n_bodies=4)
guidewire = Guidewire(n_bodies=80)
task = Navigate(
    phantom=phantom,
    guidewire=guidewire,
    tip=tip,
    target=target,
    **kwargs,
)
env = composer.Environment(
    task=task,
    random_state=random_state,
    strip_singleton_obs_buffer_dim=True,
)

env = CathSim(dm_env=env)
```

## Adding a guidewire

A guidewire can be created similarly to the phantom and then embedded into the task like above, using the MJCF model as follows:

#### Creating an MJCF model

In PyMJCF, the basic building block of a model is an `mjcf.Element`. This
corresponds to an element in the generated XML. However, user code _cannot_
instantiate a generic `mjcf.Element` object directly.

A valid model always consists of a single root `<mujoco>` element. This is
represented as the special `mjcf.RootElement` type in PyMJCF, which _can_ be
instantiated in user code to create an empty model.

```python
from dm_control import mjcf

mjcf_model = mjcf.RootElement()
print(mjcf_model)  # MJCF Element: <mujoco/>
```

#### Adding new elements

Attributes of the new element can be passed as kwargs:

```python
my_box = mjcf_model.worldbody.add('geom', name='my_box',
                                  type='box', pos=[0, .1, 0])
print(my_box)  # MJCF Element: <geom name="my_box" type="box" pos="0. 0.1 0."/>
```

Please see more information on `mjcf` [here](https://github.com/google-deepmind/dm_control/tree/main/dm_control/mjcf).*

## TODO's

- [x] Code refactoring
- [x] Add fluid simulation
- [x] Add VR/AR interface through Unity
- [x] Implement multiple aortic models
- [x] Update to `gymnasium`
- [x] Add guidewire representation
- [ ] Create tests for the environment

## Maintainers (full list of [contributors](contributors.md))

- [Tudor Jianu](https://tudorjnu.github.io/)
- [Baoru Huang](https://baoru.netlify.app)
- [Tung Ta](https://tungtd.com/)
- [Anh Nguyen](https://cgi.csc.liv.ac.uk/~anguyen/)

## Terms of Use

Please review our [Terms of Use](TERMS.md) before using this project.

## License

Please feel free to copy, distribute, display, perform or remix our work but for non-commercial purposes only.

## Citation

If you find our paper useful in your research, please consider citing:

``` bibtex
@article{jianu2024cathsim,
  title={Cathsim: an open-source simulator for endovascular intervention},
  author={Jianu, Tudor and Huang, Baoru and Vu, Minh Nhat and Abdelaziz, Mohamed EMK and Fichera, Sebastiano and Lee, Chun-Yi and Berthet-Rayne, Pierre and y Baena, Ferdinando Rodriguez and Nguyen, Anh},
  journal={IEEE Transactions on Medical Robotics and Bionics},
  year={2024},
  publisher={IEEE}
}
```
