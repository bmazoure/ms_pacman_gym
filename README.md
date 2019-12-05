# PocMan_gym
Partially observable PacMan game in OpenAI Gym format


## Install

To install, do either

```
git clone https://github.com/bmazoure/PocMan_gym.git
cd PocMan_gym
pip install -e .
```

or directly
```
pip install git+https://github.com/bmazoure/PocMan_gym.git
```

## Environment

Two observation modes are available for now (as explained and coded by Hamilton et al. 2014):
* Sparse scalar observation (Binary number 0-2^16-1)
* Sparse observation vector (16 bits)
* TODO: Fully observable vector (16 bits)
* TODO: RGB tensor of (H,W,3)