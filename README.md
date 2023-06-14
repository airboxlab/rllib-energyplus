# Ray RLlib - EnergyPlus Python API sample

An example of how to train a control policy using Ray RLlib and EnergyPlus Python API.

Requires Python 3.8+, EnergyPlus 9.3+

## Setup

### Using docker image

Look for a pre-built docker image in [packages](https://github.com/airboxlab/rllib-energyplus/pkgs/container/rllib-energyplus) and follow instructions to pull it.

Alternatively, build the docker image

```shell
docker build -t rllib-energyplus .
```

Run the container

```shell
docker run --rm --name rllib-energyplus -it rllib-energyplus
```

Inside the container, run the experiment

```shell
cd /root/rllib-energyplus
python3 run.py --idf model.idf --epw LUX_LU_Luxembourg.AP.065900_TMYx.2004-2018.epw --framework torch
```

### Using virtual environment

#### Package dependencies

Edit `requirements.txt` and add the deep learning framework of your choice (TensorFlow or PyTorch)

```shell
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

#### Path dependencies

Add EnergyPlus folder to `PYTHONPATH` environment variable:

```shell
export PYTHONPATH="/usr/local/EnergyPlus-23-1-0/:$PYTHONPATH"
```

Make sure you can import EnergyPlus API by printing its version number

```shell
$ python3 -c 'from pyenergyplus.api import EnergyPlusAPI; print(EnergyPlusAPI.api_version())'
0.2
```

## Run example

```shell
python3 run.py \
  --idf /path/to/model.idf \
  --epw /path/to/LUX_LU_Luxembourg.AP.065900_TMYx.2004-2018.epw
```

Example of episode reward stats obtained training with PPO, 1e6 timesteps, 2 workers, untuned. Experiment took ~1h45min

![PPO stats](images/ppo_untuned.png "PPO training - Single AHU model")

## Tracking an experiment

Tensorboard is installed with requirements. 
To track an experiment running in a docker container, the container must be started with `--network host` parameter. 

Start tensorboard with:

```shell
tensorboard --logdir ~/ray_results --bind_all
```
