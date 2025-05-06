A novel approach to 3D shape generation using a VAE-like network that leverages layers from the Point Net++ architecture as an encoder network and a DeepSDF-inspired decoder network.

To install all dependencies please activate the conda environment in `environment.yaml`. 

The training loop is implemented in `pointnet_sdfvae_simple.ipynb`. We already used this notebook to train a model on a larger dataset, and saved its weights in `working_model.pth`.

To generate new cars and visualize the latent space, see `latent-visualizations`
