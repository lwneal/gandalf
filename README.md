## GANdalf

GANdalf trains auto-encoding generative adversarial networks.

Autoencoding GANs are trained with two objectives:

* Generate realistic-looking fake examples based on an input vector of random noise from a low-dimensional "latent space".
* Compress real examples by encoding them into points in that latent space, then decoding them back to their original form

GANdalf adds a third objective:

* Given a point in the latent space, predict attributes about it with a classifier network

For any example, we can compute the gradient of a classifier's output with respect to the example's encoding in latent space.
By gradient descent, the example can be perturbed in such a way that it resembles the original example as much as possible, while being classified in a different desired way.
We term such a perturbed example a "counterfactual" example.

By choosing an appropriate classifier and target, counterfactual examples can be generated for a variety of useful tasks.

### Code Structure

Each model and its results is saved in a result directory, referred to as a `result_dir`.
A `result_dir` includes a configuration file `params.json`, saved weights with the extension `.pth`, and a variety of `.jpg`, `.mp4`, and `.png` files created during evaluation and testing.

Each file in `experiments/` is an executable job that takes the parameter `--result_dir`.

The experiment `train_aac.py` is special: it creates the `result_dir` if it does not already exist.

Other experiments require `result_dir` to exist already, and perform some task using it.

### Getting Started

To add an experiment, first run:

    `./test.sh`

Then edit the things that say "example" in `experiments/example.py` and run `test.sh` again.

### Requirements

Runs on Ubuntu, under Python 3.5. Requires CUDA and PyTorch 0.2 or higher. For other requirements, run `install_requirements.sh`.

#### Datasets

The input to each experiment is a `.dataset` file. These files can be generated by the scripts in `datasets/`.

#### Experiments

Each experiment is an executable Python script. Run `experiments/train_aac.py` to start training a manifold.

#### Counterfactuals

After a network has trained for at least one epoch, use it to generate counterfactuals with `experiments/generate_counterfactual.py`

## Hacks

The following requirements are hacks that should be removed or turned into configuration:

* All datasets should be placed in `/mnt/data` using `datasets/download_mnist.sh` et al
* The `--result_dir` option should always point to a directory in `/mnt/results/`
* If you `mv` a result\_dir, you have to manually edit its `params.json` or else the evaluate scripts will break
* If you stop train\_aac and then restart it, the `--epochs` parameter in params.json won't match up with the actual number of epochs

