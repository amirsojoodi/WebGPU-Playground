# Install NVIDIA Driver and CUDA Sdk

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# where $distro/$arch should be replaced by one of the following:
# ubuntu1604/x86_64
# ubuntu1804/cross-linux-sbsa
# ubuntu1804/ppc64el
# ubuntu1804/sbsa
# ubuntu1804/x86_64
# ubuntu2004/cross-linux-aarch64
# ubuntu2004/arm64
# ubuntu2004/cross-linux-sbsa
# ubuntu2004/sbsa
# ubuntu2004/x86_64
# ubuntu2204/sbsa
# ubuntu2204/x86_64

sudo apt update

```

To install CUDA Sdk:

```bash
sudo apt install cuda-toolkit

sudo apt install cuda-drivers
# If a specific driver version is required
sudo apt install cuda-drivers-535

sudo reboot

# Verify the installation
nvidia-smi
```

Maybe you need to set some environment variables:

```bash
export CUDA_HOME="/usr/local/cuda"
export PATH=$PATH:"$CUDA_HOME/bin"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$CUDA_HOME/lib64"
# This is required for CUDA driver API
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$CUDA_HOME/lib64/stubs"
```
