# DeepCV README.md (_Work In Progress_)

__By Paul-Emmanuel SOTIR <paulemmanuel.sotir@oultook.com>__  
_This project is under Open Source MIT License, see [./LICENSE](./LICENSE) or more details._

__WIP: This Project is still under active development, and at an early stage of development__  

DeepCV is a Kedro PyTorch project which aims to simplify the implementation of simple vision tasks. DeepCV allows you to easily define vision processing pipelines by leveraging recent DeepLearning algorithms along with the usual OpenCV tools.  

Some of DeepCV's main features are:
- Custom lossless Image and video compression codec using learned arithmetic encoder policies to minimize image and video sizes (which means faster streaming, faster reading from disk, lower storage size) 
- Feature extraction and matching using lightweight CNNs, improving the reliability and reproducibility of image processing pipelines compared to implementations that rely on classical feature extractors such as SIFT, ORB, ...
- The [`deepcv.meta`](./src/deepcv/meta) python module contains various utilities to make it easier to [define models](./src/deepcv/meta/base_module.py), [train models with ignite](./src/deepcv/meta/ignite_training.py), [search hyperparameters with NNI](./src/deepcv/meta/hyperparams.py), follow and visualize training experiments with MLFlow/kedro/TensorboardX..., [preprocess](./src/deepcv/meta/data/preprocess.py) and [augment data](./src/deepcv/meta/data/augmentation.py), schedule learning rate(s) with [One Cycle policy](./src/deepcv/meta/one_cycle.py), perform meta-learning thanks to various tools like [`HyperparametersEmbedding`](./src/deepcv/meta/hyperparams.py) and as well as meta deep-learning abstractions (e.g. [`Experiment`, `DatasetStats`, `Task`, `Hyperparameters`, `HyperparameterSpace`](./src/deepcv/meta/data/training_metadata.py)) stored for each experiments in a 'metadataset', ...
- [`deepcv.meta.base_module.DeepcvModule`](./src/deepcv/meta/base_module.py) A base class for easier DeepCV model definition: model sub-modules (NN blocks or layers) can be defined in a simple and generic manner in [`./conf/base/parameters.yml`](./conf/base/parameters.yml) and a shared image embedding block of a few convolution layers can allow learning to be transferred between any DeepCV image models by sharing, training, forking and/or merging these shared weights.
- [`./conf/base/parameters.yml`] can also specify data augmentation and preprocessing recipes.
- ...

## Install instructions

### Method #1: Install from this repository
``` shell
git clone ...
...
# You can then run tests scripts to verify successfull installation of DeepCV:
python tests/test1.py
```

### Method #2: Install our package from Anaconda repository
TODO: ...

## Usage example

``` python
import deepcv

def main():
    pass

if __name__ == "__main__":
    main()
```

## Documentation
TODO: more in depth description

### See [Sphinx documentation](www.deepcv.com/sphinx/index.html) for more details
Alternatively, if you need documentation from a specific branch or updated with your contributions, you can build sphinx documentation by following these instructions:

``` shell
git clone ...
```

## Contribution guide

TODO: bla bla

## TODO List

__Interesting third party projects which could be integrated into this project__

Eventually create submodule(s) for the following github projects under third_party directory (see https://git-scm.com/book/fr/v2/Utilitaires-Git-Sous-modules + script to update submodule to latest release commit?):
- ImageNetV2  
- Detectron2  
- Apex (https://github.com/NVIDIA/apex): needs to be installed manually along with PyProf (optional)  
- Use pytorch-OpCounter for pytorch model memory/FLOPs profiling: https://github.com/Lyken17/pytorch-OpCounter
- SinGAN pytorch implementation: https://github.com/tamarott/SinGAN
- https://github.com/MegviiDetection/video_analyst
- https://github.com/rwightman/pytorch-image-models

__External dependencies__ _TODO: Remove this section_
kedro/mlflow/ignite/pytorch/tensorboard//NNI/Apex/Scikit-learn/Numpy/pandas/Jupyter/.../Python/Conda/CUDA + DeepCV with ffmpeg(+ faster/hardware-accelerated video h264/VP9/AV1 decompression lib?) + DeepCV docker image with or without GPU acceleration + keep in mind portability (e.g. to android, ARM, jetson, ...)

<details>
  <summary><b> DeepCV Features TODO list</b></summary>
<ul>
    <li> Implement continuous integration using Travis CI</li>
    <li> Create or find an ECA implementation: channel attention gate on convolution gate using sigmoid of 1D convolution output as attention gate (element-wise multiplication of each channels with their respective gating scale) (kernel size of 1D conv: k << ChannelCount with k=Func(C)) </li>
    <li> Add image completion/reconstruction/generation/combination (could be used as data augmentation trick) to DeepCV (see paper about one shot image completion/combination/reconstruction and distill+quantize it and combine it with usual and simple augmentation recipes when used for data augmentation)</li>
    <li> Implement basic image feature matching and compare it against shitty approaches like SIFT, ORB, ...</li>
    <li> Implement a pipeline for video stiching and add support for video stabilization, audio-and/or-visual synchronization, image compression (lossless or lossy), watermark removal, visual tracking, pose estimation + simplify usage: (scikit compliant models, warpers over pipelined models for easier usage along with DeepCV, package it like a plugin to DeepCV, fine-tuning the framework for easier training of whole pipelines on custom data)</li>
    <li> Implement (fork lepton and L3C for better AC and DC compression using deterministic shallow-NN prediction from context) or add better jpeg and mpeg compression codecs (for use cases where storage-size/bandwidth is the priority, e.g. for faster video/image processing or streaming pipelines, or smaller media storage (= priority to size, then, prioritize decompression time vs compression time)) and/or look for algorithms which could be applied directly on compressed images/frames (see [Lepton](https://dropbox.tech/infrastructure/lepton-image-compression-saving-22-losslessly-from-images-at-15mbs) and [L3C](https://arxiv.org/pdf/1811.12817v3.pdf) ) + utilities to convert files to our codec for faster processing:
    <ul>
        <li> must be lossless to preserve visual quality when encoding back to jpeg, but should match the benefits from any existing lossy jpeg compression (e.g. lossless algorithm built on top of jpeg's tiles)</li>
        <li> keep in mind the possibility of progressive image/frame loading/streaming in a future implementation</li>
        <li> benchmark performances on imagenet, compare speed and size with L3C (use benchmarking code from https://github.com/fab-jul/L3C-PyTorch) </li></ul></li>
    <li> implement distillation/quantization  + Apex</li>
    <li> add a simple open-source implementation of wave function collapsing, optimize it
        -> Future work : Procedural Content Generation: Use a GAN to generates slots (learn scenes manifold by semantic clusters) used by Wave Function Collapse (+ Growing Grids as space filling algorithm to determine tile shapes) </li>
    <li> add uncertainty estimation tools on deep learning models</li>
    <li> Implement unit and feature tests </li>
</ul>
</details>

Kedro project generated using `Kedro 0.15.7`
```
```

Take a look at the [documentation](https://kedro.readthedocs.io) to get started.

## Rules and guidelines

In order to get the best out of the template:
 * Please don't remove any lines from the `.gitignore` file provided
 * Make sure your results can be reproduced by following a data engineering convention, e.g. the one we suggest [here](https://kedro.readthedocs.io/en/stable/06_resources/01_faq.html#what-is-data-engineering-convention)
 * Don't commit any data to your repository
 * Don't commit any credentials or local configuration to your repository
 * Keep all credentials or local configuration in `conf/local/`

## Temporary: README generated by Kedro:

### Installing dependencies

Dependencies should be declared in `src/requirements.txt` for pip installation and `src/environment.yml` for conda installation.

``` shell
kedro install
```

### Running Kedro

You can run your Kedro project with:

``` shell
kedro run
```

### Testing Kedro

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests with the following command:

``` shell
kedro test
```

To configure the coverage threshold, please have a look at the file `.coveragerc`.

#### Working with Kedro from notebooks

In order to use notebooks in your Kedro project, you need to install Jupyter:

``` shell
pip install jupyter
```

For using Jupyter Lab, you need to install it:

``` shell
pip install jupyterlab
```

After installing Jupyter, you can start a local notebook server:

``` shell
kedro jupyter notebook
```

You can also start Jupyter Lab:

``` shell
kedro jupyter lab
```

And if you want to run an IPython session:

``` shell
kedro ipython
```

Running Jupyter or IPython this way provides the following variables in
scope: `proj_dir`, `proj_name`, `conf`, `io`, `parameters` and `startup_error`.

##### Converting notebook cells to nodes in a Kedro project

Once you are happy with a notebook, you may want to move your code over into the Kedro project structure for the next stage in your development. This is done through a mixture of [cell tagging](https://jupyter-notebook.readthedocs.io/en/stable/changelog.html#cell-tags) and Kedro CLI commands.

By adding the `node` tag to a cell and running the command below, the cell's source code will be copied over to a Python file within `src/<package_name>/nodes/`.

``` shell
kedro jupyter convert <filepath_to_my_notebook>
```

> *Note:* The name of the Python file matches the name of the original notebook.

Alternatively, you may want to transform all your notebooks in one go. To this end, you can run the following command to convert all notebook files found in the project root directory and under any of its sub-folders.

``` shell
kedro jupyter convert --all
```

##### Ignoring notebook output cells in `git`

In order to automatically strip out all output cell contents before committing to `git`, you can run `kedro activate-nbstripout`. This will add a hook in `.git/config` which will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be left intact locally.

### Package the project

In order to package the project's Python code in `.egg` and / or a `.wheel` file, you can run:

``` shell
kedro package
```

After running that, you can find the two packages in `src/dist/`.

### Building API documentation

To build API docs for your code using Sphinx, run:

``` shell
kedro build-docs
```

See your documentation by opening `docs/build/html/index.html`.

### Building the project requirements

To generate or update the dependency requirements for your project, run:

``` shell
kedro build-reqs
```

This will copy the contents of `src/requirements.txt` into a new file `src/requirements.in` which will be used as the source for `pip-compile`. You can see the output of the resolution by opening `src/requirements.txt`.

After this, if you'd like to update your project requirements, please update `src/requirements.in` and re-run `kedro build-reqs`.
