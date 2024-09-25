# GPU Deconvolution

## What

This repo contain experiments investigating how a deconvoultion method performs on data
relevant for climate simulation output.

## Install

The project is developed using [pixi]. To install, clone the repo and let [pixi] do the
rest.

```bash
git clone https://github.com/engeir/gpu-deconv
cd gpu-deconv || exit
pixi install
```

[Pixi] is similar to [conda], and is able to produce environment files that [conda] can
use to install the exact same project environment. To install this project using [conda]
rather than [pixi]:

```bash
pixi project export conda-environment environment.yml
conda env create -n new-env-name -f environment.yml
```

## Usage

Run the scripts as usual as `python src/<file>.py`, or in some cases [pixi] tasks are
available: `pixi run script1`.

[conda]: https://docs.conda.io/en/latest/index.html
[pixi]: https://pixi.dev/latest/
