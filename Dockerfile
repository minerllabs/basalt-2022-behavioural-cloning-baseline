# In case you don't want to use Dockerfile and build from
# scratch, you can delete simply "Dockerfile" from your repository.
# Additional details on specifying dependencies are available in the README.

# Pre-installed conda and apt based MineRL runtime.
# This is done to save time during the submissions and faster debugging for you.
FROM aicrowd/base-images:minerl-22-base

# Install needed apt packages
ARG DEBIAN_FRONTEND=noninteractive
USER root
COPY apt.txt apt.txt
RUN apt -qq update && xargs -a apt.txt apt -qq install -y --no-install-recommends \
 && rm -rf /var/cache/*

# Set the user and conda environment paths
USER aicrowd
ENV HOME_DIR /home/$USER
ENV CONDA_DEFAULT_ENV="minerl"
ENV PATH /home/aicrowd/.conda/envs/minerl/bin:$PATH
ENV FORCE_CUDA="1"

# Use MineRL environment
SHELL ["conda", "run", "-n", "minerl", "/bin/bash", "-c"]

# Conda environment update
COPY environment.yml environment.yml
RUN conda env update --name minerl -f environment.yml --prune

# Copy the files
COPY --chown=1001:1001 . /home/aicrowd