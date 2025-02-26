# GPU Deconvolution

## What

This repo contain experiments investigating how a deconvolution method performs on data
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

## Troubleshooting

Some issues may arise when running this code.

- **Path to the GPU libraries**

  The default path used to look for libraries is not necessarily the correct one, and
  can be overridden with the `LD_LIBRARY_PATH` environment variable. When using [pixi],
  this is set in the _tasks_ in the [pyproject.toml](./pyproject.toml) file. See the
  [fpp-analysis-tools README](https://github.com/uit-cosmo/fpp-analysis-tools) for how
  to find your path.

- **Plotting on a remote server**

  The scripts in this repo do some plotting, and its nice to be able to view them
  interactively, not just save the image files and inspect those. This need an `ssh`
  connection that uses the `-X` flag (`ssh -X user@servername`) or that it is set in the
  `ssh` configuration at `$HOME/.ssh.config`:

  ```txt
  Host somecustomname
      ForwardX11 yes
      ForwardX11Timeout 0
      Hostname 1.2.3.4
      User user
  ```

  Note that there might be a timeout on the X11 forwarding, so setting this to zero (no
  timeout) in the configuration is the simplest solution.

  Further, the `DISPLAY` environment variable must be set once you are logged into the
  server. Try first `echo "$DISPLAY"`, and if it is empty do
  `export DISPLAY=localhost:10.0`. In this repo, this value is set in the
  [.mise.toml](./.mise.toml) file, but this depend on having [mise] installed to work.

[conda]: https://docs.conda.io/en/latest/index.html
[mise]: https://mise.jdx.dev/
[pixi]: https://pixi.dev/latest/
