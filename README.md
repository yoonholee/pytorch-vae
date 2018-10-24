# pytorch-vae

A minimal implementation of [VAE](https://arxiv.org/abs/1312.6114), [IWAE](https://arxiv.org/abs/1509.00519), and [MIWAE](https://arxiv.org/abs/1802.04537).
We followed the experimental setup of the [IWAE paper](https://arxiv.org/abs/1509.00519) as closely as possible.

## Usage

You should be able to run experiments right away.
First create a virtual environment using [pipenv](https://github.com/pypa/pipenv):

```pipenv install```

To run experiments, you simply have to use:

```pipenv run python main.py <options>```

## Example commands

For original VAE:

```pipenv run python main.py ```

To also make figures (reconstruction, samples):

```pipenv run python main.py --figs ```

For IWAE with 5 importance samples:

```pipenv run python main.py --importance_num=5 ```

For MIWAE(16, 4):

```pipenv run python main.py --mean_num=16 --importance_num=4 ```

See [the config file](https://github.com/yoonholee/pytorch-generative/blob/master/utils/config.py) for more options.

## Results
