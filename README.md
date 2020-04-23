# DeepCV README.md (_Work In Progress_)

_This project is under Open Source MIT License, see [./LICENSE](./LICENSE) or more details._  

__By Paul-Emmanuel SOTIR <paulemmanuel.sotir@oultook.com>__  
__WIP: This Project is still under active development, and at an early stage of development__  

DeepCV is a Kedro PyTorch project which aims to simplify the implementation of simple vision tasks. DeepCV allows you to easily define vision processing pipelines by leveraging recent DeepLearning algorithms along with the usual OpenCV tools.  

Some of DeepCV's main features are:
- The [`deepcv.meta`](./src/deepcv/meta) python module contains various utilities to make it easier to [define models](./src/deepcv/meta/base_module.py), [train models with ignite](./src/deepcv/meta/ignite_training.py), [search hyperparameters with NNI](./src/deepcv/meta/hyperparams.py), follow and visualize training experiments with MLFlow/kedro/TensorboardX..., [preprocess](./src/deepcv/meta/data/preprocess.py) and [augment data](./src/deepcv/meta/data/augmentation.py), schedule learning rate(s) with [One Cycle policy](./src/deepcv/meta/one_cycle.py), perform meta-learning thanks to various tools like [`HyperparametersEmbedding`](./src/deepcv/meta/hyperparams.py) and as well as meta deep-learning abstractions (e.g. [`Experiment`, `DatasetStats`, `Task`, `Hyperparameters`, `HyperparameterSpace`](./src/deepcv/meta/data/training_metadata.py)) stored for each experiments in a 'metadataset', ...
- [`deepcv.meta.base_module.DeepcvModule`](./src/deepcv/meta/base_module.py) A base class for easier DeepCV model definition: model sub-modules (NN blocks or layers) can be defined in a simple and generic manner in [`./conf/base/parameters.yml`](./conf/base/parameters.yml) and a shared image embedding block of a few convolution layers can allow learning to be transferred between any DeepCV image models by sharing, training, forking and/or merging these shared weights.
- [`./conf/base/parameters.yml`] can also specify data augmentation and preprocessing recipes.
- ...

## Install instructions

In order to handle dependencies, DeepCV requires a conda distribution (e.g. [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)) to be installed in your working environment.  
You will then need to activate conda environment using `conda activate deepcv` if you want to run DeepCV code.  
DeepCV is a project based on [Kedro machine learning project template](https://github.com/quantumblacklabs/kedro). We modified `kedro install` command in order to better support conda environments. Thus, during kedro installation, a conda environment for DeepCV will be created according to [.src/environment.yml conda env file](.src/environment.yml); Feel free to modify it according to your needs (e.g. add dependencies, ...) and then run `kedro install` to either update or create a new conda environment.  

Once installed you can use DeepCV as a project template or a dependency in your code, and run it either througt 
 `kedro run` command to run machine learning pipelines or directly from your python source file. You can also run a deepcv source file to test it: e.g. `python -o ./src/deepcv/synchronization/audio.py` will run some unit tests related to audio synchronization tasks.

### Method #1: Install from this repository

``` shell
git clone https://github.com/PaulEmmanuelSotir/DeepCV.git
cd ./DeepCV

# Install DeepCV dependencies and conda environment (if you want to customize conda environment you can either modify default YAML conda env file at ./src/environment.yml or specify a new env file (see `--conda-yml` option in `kedro install -h`))
kedro install

# You then need to activate conda environment (make sure to have a conda distro installed)
conda activate deepcv

# You can then run tests of any deepcv module to verify successfull installation of DeepCV (Won't work for now, stay tuned 📡):
python deepcv/detection/object.py
```

### Method #2: Install our package from Anaconda repository

TODO: Package DeepCV project and upload it to a conda cloud repository
``` shell
conda install deepcv

# Install DeepCV dependencies and conda environment (if you want to customize conda environment you can either modify default YAML conda env file at ./src/environment.yml or specify a new env file (see `--conda-yml` option in `kedro install -h`))
kedro install

# You then need to activate conda environement (make sure to have a conda distro installed)
conda activate deepcv

# You can then run tests of any deepcv module to verify successfull installation of DeepCV (Won't work for now, stay tuned 📡):
python deepcv/detection/object.py
```

## Usage example

Make sure to run `conda activate deepcv` (or activate your own conda env with deepcv as a dependency) before runing DeepCV code.  
Here is an example usage of deepcv from your own Python source:  

``` python
import deepcv
import deepcv.meta as meta

def main():
    # TODO: Show example usage of deepcv
    raise NotImplementedError

if __name__ == "__main__":
    main()
```

## Documentation
DeepCV is a Kedro PyTorch project which aims to simplify the implementation of simple vision tasks. DeepCV allows you to easily define vision processing pipelines by leveraging recent DeepLearning algorithms along with the usual OpenCV tools.  
DeepCV is a project based on [Kedro machine learning project template](https://github.com/quantumblacklabs/kedro), which enforce and simplifies the definition, configuration, training and inference of machine learning pipelines.  
Moreover DeepCV uses [MLFlow](https://mlflow.org/) for better experiment versionning and visualization.  

__See [hosted Sphinx documentation](www.deepcv.com/sphinx/index.html) for more details, local version at [./docs/build/html/index.html](./docs/build/html/index.html) (WIP: Not hosted for now nor very informative, stay tunned 📡)__

Alternatively, if you need documentation from a specific branch or updated documentation with your own contributions, you can build sphinx documentation by following these instructions:

``` shell
git clone https://github.com/PaulEmmanuelSotir/DeepCV.git
cd ./DeepCV
kedro install
kedro build-docs
# Once documentation have been sucessfully built, you can browse to ./docs/build/html/index.html
```

## Contribution guide

Any contribution are welcome, but keep in mind this project is still at a very early stage of development.  
Feel free to submit issues if you have any feature suggestion, ideas, application areas, or if you have difficulties to reuse it.

## 📝TODO List📝

__DeepCV Features and code refactoring TODO List__ 💥(☞ﾟヮﾟ)☞💥

👍 = DONE; ♻ = WIP; 💤: TODO  

- 👍 Implement conda activate when kedro is called in kedro_cli.py (WIP: testing and debug) 
- ♻Finalize object detection model definition + move generic code of ObjectDetector to DeepCVModule base class
- ♻Improve Hyperparameters/HyperparameterSpace/HyperparameterEmbedding/GeneralizationAcrossScalesPredictor implementations + integrate with NNI (remove any hyperopt usage)
- ♻parse and process [deepcv.meta.data.preprocess](./src/deepcv/meta/data/preprocess.py) recipes from parameters.yml
- ♻parse and process [deepcv.meta.data.augmentation](./src/deepcv/meta/data/augmentation.py) recipes from parameters.yml
- ♻refactor augmentation operators of AugMix on PIL images in [deepcv.meta.data.augmentation](./src/deepcv/meta/data/augmentation.py)
- 👍make possible to specify dense and/or residual links in NN architecture configuration ([parameters.yml](./conf/bas/parameters.yml)) and process it accordingly in forward method of [deepcv.meta.base_module.DeepcvModule](./src/deepcv/meta/base_module.py) model base class
- ♻Improve dense/residual link support by reducing its memory footprint: store [deepcv.meta.base_module.DeepcvModule](./src/deepcv/meta/base_module.py) sub-modules output features only if they are actually needed by a residual/dense link deeper in NN architecture
- ♻Improve [deepcv.meta.base_module.DeepcvModule](.src/deepcv/meta/base_module.py) model base class to parse YAML NN architecture definition of [parameters.yml](./conf/base/parameters.yml) in a more powerfull/generic way to allow siamese NNs, residual/dense links down/up-sampling, attention gates, multi-scale/depth inputs/outputs, ... (for now, [DeepcvModule](.src/deepcv/meta/base_module.py) only support YAML NN architecture definition made of a list of sub-modules with eventual residual-dense links.
- 💤Setup and download various torchvision datasets ( at least CIFAR10/100 and ImageNet32 datasets)
- ♻fix tests/deepcv module imports (make 'tests' like a third party module appart from deepcv or move 'tests' into deepcv module)
- ♻fix code and YAML config files in order to be able to run basic kedro pipelines and build documentation
- ♻Run and Debug whole object detector pipeline
- ♻Look into possible implementation of AugMix into deepcv (+ see any improved versions of AugMix)
- ♻MLFlow integration... (including hyperprarmeter search integration if needed, dashboard, versionning, custom experiments storage for meta-learning, ...)
- ♻Fully implement HybridConnectivityGatedNet model (+ refactor it to make usage of newest version of  [deepcv.meta.base_module.DeepcvModule](./src/deepcv/meta/base_module.py) model base class)
- 💤Create jupyter notebook(s) for basic prototyping and training results visualization + implement utility tools for jupyter notebooks
- ♻NNI Hyperparameter search integration
- 💤Train object detection model + its hp search + Human detection model
- 💤Start Ensembling and stacking utilities module implementation
- ♻Implement OneCycle Policy along with optional learning rate scales varying for each layers or conv blocks + momentum and eventually investigate similar policies for other hyperprarmeters (e.g. dropout_prob, L2, ...) + consider to integrate fastai to deepcv dependencies in order to reuse its OneCycle policy implemetation?
- ♻Implement architectures templates/patterns for multiscale neural net inputs and outputs + eventually gaussian blur kernels applied to convolutions activations with decreasing blur kernel size during training steps (+ rapid SOTA review from citing papers of these techniques)
- 💤Implement Uncertainty estimation utilities in [deepcv.meta.uncertainty.estimation module](./src/deepcv/meta/uncertainty/estimation.py), see: https://ai.googleblog.com/2020/01/can-you-trust-your-models-uncertainty.html
- 💤Implement or integrate distillation with optionnal quantization tools + distillation from ensembles of teacher networks (see NNI, Apex and built-in PyTorch quantization/compression tooling)
- ♻Train, vizualize and investigate the effect of various contrastive losses like [deepcv.meta.contrastive.JensenShannonDivergenceConsistencyLoss or deepcv.meta.contrastive.TripletMarginLoss](./src/deepcv/meta/contrastive.py) for embeddings and supervised training setups (à la contrastive learning for pretraining or additional loss term in supervised training setup) -> look for other contrastive or generative and combine thoose approaches with classical supervised losses
- 💤Setup Continuous Integration using Travis CI
- 💤Create or find an ECA implementation: channel attention gate on convolution gate using sigmoid of 1D convolution output as attention gate (element-wise multiplication of each channels with their respective gating scale) (kernel size of 1D conv: k << ChannelCount with k=Func(C))
- 💤Add image completion/reconstruction/generation/combination to DeepCV with a custom model distilled and/or quantized from [SinGAN](https://arxiv.org/abs/1905.01164) + for data augmentation setup: integrate it to AugMix augmentation transforms
- 💤Use [UDA (UnsupervisedDataAugmentation)](https://arxiv.org/pdf/1904.12848.pdf) and replace/append-to its underlying image augmentation method (RandAugment) with a custom model distilled from [SinGAN](https://arxiv.org/abs/1905.01164) (i.e. use [SinGAN](https://arxiv.org/abs/1905.01164) under [UDA](https://arxiv.org/pdf/1904.12848.pdf) framework) 🎓🧪:
    - [official UDA implemetantion](https://github.com/google-research/uda)
    - [pytorch implementation (more recent and  slightly more popular but applied to text)](https://github.com/SanghunYun/UDA_pytorch)
    - [pytorch implementation 2 (seems a bit sketchy but applied to images)](https://github.com/vfdev-5/UDA-pytorch)
- 💤Implement basic image feature matching and compare it against classical/non-ML vision approaches like SIFT, ORB, ...: Feature extraction and matching using lightweight CNNs, improving the reliability and reproducibility of image processing pipelines compared to implementations that rely on classical feature extractors such as SIFT, ORB, ...
- 💤Implement a pipeline for video stiching and add support for video stabilization, audio-and/or-visual synchronization, image compression (lossless or lossy), watermark removal, visual tracking, pose estimation
- 💤implement warpers over DeepCV model pipelines to allow scikit model interface usage and better integration along with OpenCV code + fine-tuning tooling of whole pipelines on small amount of custom data 
- 💤Implement a mechanism to choose which pipelines/models/third-party projects/dependencies to enable or not (i.e. optional plugins to DeepCV)
- 💤Custom lossless Image and video compression codec using learned arithmetic encoder policies to minimize image and video sizes (which means faster streaming, faster reading from disk, lower storage size) : Implement (fork lepton and L3C for better AC and DC compression using deterministic shallow-NN prediction from context) or add better JPEG and MPEG compression codecs (for use cases where storage-size/bandwidth is the priority, e.g. for faster video/image processing or streaming pipelines, or smaller media storage (= priority to size, then, prioritize decompression time vs compression time)) and/or look for algorithms which could be applied directly on compressed images/frames (see [Lepton](https://dropbox.tech/infrastructure/lepton-image-compression-saving-22-losslessly-from-images-at-15mbs) and [L3C](https://arxiv.org/pdf/1811.12817v3.pdf) ) + utilities to convert files to our codec for faster processing:
    - must be lossless to preserve visual quality when encoding back to jpeg, but should match the benefits from any existing lossy jpeg compression (e.g. lossless algorithm built on top of jpeg's tiles)
    - keep in mind the possibility of progressive image/frame loading/streaming in a future implementation<
    - benchmark performances on imagenet, compare speed and size with L3C (use benchmarking code from https://github.com/fab-jul/L3C-PyTorch)
- 💤add a simple open-source implementation of wave function collapsing, optimize it -> Future work : Procedural Content Generation: Use a GAN to generates slots (learn scenes manifold by semantic clusters) used by Wave Function Collapse (+ Growing Grids as space filling algorithm to determine tile shapes)
- ♻Implement unit tests and sanity checks


__Interesting third party projects which could be integrated into DeepCV__

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

________________________________________________________________________________________________________________________________________

# Temporary: README.md generated by Kedro:

Take a look at the [documentation](https://kedro.readthedocs.io) to get started.

## Rules and guidelines

In order to get the best out of the template:
 * Please don't remove any lines from the `.gitignore` file provided
 * Make sure your results can be reproduced by following a data engineering convention, e.g. the one we suggest [here](https://kedro.readthedocs.io/en/stable/06_resources/01_faq.html#what-is-data-engineering-convention)
 * Don't commit any data to your repository
 * Don't commit any credentials or local configuration to your repository
 * Keep all credentials or local configuration in `conf/local/`

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
