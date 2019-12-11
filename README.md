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
* Sparse scalar observation (Binary number 0-2^16-1), some bits zeroed out
* Sparse observation vector (16 bits), some bits zeroed out
* Full scalar observation (Binary number 0-2^16-1)
* Full observation vector (16 bits)
* Fully observable ASCII (21 x 19 str)
* Fully observable RGB tensor of (21 x 19 x 3 uint8)