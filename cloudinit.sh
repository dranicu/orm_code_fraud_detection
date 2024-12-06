#!/bin/bash

#AI_financial_test
echo "running cloudinit.sh script"

dnf install -y dnf-utils zip unzip gcc
dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo
dnf remove -y runc

echo "INSTALL DOCKER"
dnf install -y docker-ce --nobest

echo "ENABLE DOCKER"
systemctl enable docker.service

echo "INSTALL NVIDIA CONT TOOLKIT"
dnf install -y nvidia-container-toolkit

echo "START DOCKER"
systemctl start docker.service

echo "PYTHON packages"
python3 -m pip install --upgrade pip wheel oci
python3 -m pip install --upgrade setuptools
python3 -m pip install oci-cli
python3 -m pip install langchain
python3 -m pip install python-multipart
python3 -m pip install pypdf
python3 -m pip install six

echo "GROWFS"
/usr/libexec/oci-growfs -y


echo "Export nvcc"
sudo -u opc bash -c 'echo "export PATH=\$PATH:/usr/local/cuda/bin" >> /home/opc/.bashrc'
sudo -u opc bash -c 'echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> /home/opc/.bashrc'

echo "Add docker opc"
usermod -aG docker opc

echo "CUDA toolkit"
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf clean all
dnf -y install cuda-toolkit-12-4
dnf -y install cudnn

echo "Python 3.10.6"
dnf install curl gcc openssl-devel bzip2-devel libffi-devel zlib-devel wget make -y
wget https://www.python.org/ftp/python/3.10.6/Python-3.10.6.tar.xz
tar -xf Python-3.10.6.tar.xz
cd Python-3.10.6/
./configure --enable-optimizations
make -j $(nproc)
sudo make altinstall
python3.10 -V
cd ..
rm -rf Python-3.10.6*

echo "Git"
dnf install -y git

echo "Conda"
mkdir -p /home/opc/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/opc/miniconda3/miniconda.sh
bash /home/opc/miniconda3/miniconda.sh -b -u -p /home/opc/miniconda3
rm -rf /home/opc/miniconda3/miniconda.sh
/home/opc/miniconda3/bin/conda init bash
chown -R opc:opc /home/opc/miniconda3
su - opc -c "/home/opc/miniconda3/bin/conda init bash"

# Ensure the .bashrc is reloaded
sudo -u opc bash -c "source /home/opc/.bashrc"

echo "Creating Conda environment"

# Create the YAML file and redirect it to env_triton.yaml
su - opc -c "cat <<EOF > /home/opc/env_triton.yaml
name: triton_example
channels:
  - conda-forge
  - nvidia
  - rapidsai
dependencies:
  - cudatoolkit=11.4
  - cudf=21.12
  - cuml=21.12
    #  - cupy
  - jupyter
  - kaggle
  - matplotlib
  - numpy=1.21.5
  - pandas
  - pip
  - python=3.8
  - scikit-learn=0.24.2
  - pip:
      - tritonclient[all]
      - xgboost>=1.5,<1.6
      - numba==0.55.1
      - cython==0.29.34
      - cupy-cuda11x==11.5.0
      - pynvml
      - altair
      - category_encoders
      - umap
      - umap-learn
      - treelite
      - treelite_runtime
      - tensorflow
      - protobuf==3.20.0
EOF"

# Create the YAML file and redirect it to env_domino.yaml
su - opc -c "cat <<EOF > /home/opc/env_domino.yaml
name: domino_example
channels:
  - conda-forge
  - nvidia
  - rapidsai
dependencies:
  - cudatoolkit=11.4
  - cudf=21.12
  - cuml=21.12
  - kaggle
  - matplotlib
  - numpy=1.21.5
  - pandas
  - python=3.8
  - scikit-learn=1.0.2
  - pip
  - pip:
      - cupy-cuda11x==11.5.0
      - imbalanced-learn==0.12.3
      - tritonclient[all]
      - xgboost>=1.5,<1.6
      - numba==0.55.1
      - cython==0.29.34
      - pynvml
      - seaborn
      - imblearn
      - altair
      - category_encoders
      - umap
      - umap-learn
      - treelite
      - treelite_runtime
      - tensorflow
      - protobuf==3.20.0
EOF"


# Create the Conda environment using the newly created YAML file
su - opc -c "/home/opc/miniconda3/bin/conda env create -f /home/opc/env_triton.yaml"


echo "source activate triton_example" >> /home/opc/.bashrc
echo "Starting Jupyter Notebook server as opc and triton_example env"

# Start Jupyter Notebook server
su - opc -c "source /home/opc/.bashrc && \
             conda activate triton_example && \
             nohup jupyter notebook --ip=0.0.0.0 --port=8888 > /home/opc/jupyter.log 2>&1 &"


echo "Creating kernel execution for Jupyter domino_example"

# Create the Conda env_domino environment
su - opc -c "/home/opc/miniconda3/bin/conda env create -f /home/opc/env_domino.yaml"

# Install ipykernel in the created environments
su - opc -c "/home/opc/miniconda3/bin/conda run -n domino_example /home/opc/miniconda3/bin/conda install ipykernel"
su - opc -c "/home/opc/miniconda3/bin/conda run -n triton_example /home/opc/miniconda3/bin/conda install ipykernel"

# Install the Jupyter kernel
su - opc -c "/home/opc/miniconda3/bin/conda run -n domino_example python -m ipykernel install --user --name domino_example --display-name 'domino_example_kernel'"
su - opc -c "/home/opc/miniconda3/bin/conda run -n triton_example python -m ipykernel install --user --name triton_example --display-name 'triton_example_kernel'"

# Get the model files:
su - opc -c "wget https://objectstorage.il-jerusalem-1.oraclecloud.com/p/pyGEGe7DpwPRzMjT6arIwe7O1aH5rjxMsyeC7Y8Z17Dt9ai4Nk8Re8UYHELVB5nT/n/orasenatdpltintegration03/b/bucket_for_bogdan/o/AI/Notebook/models_for_test_final.zip"
su - opc -c "unzip models_for_test_final.zip"
su - opc -c "docker pull nvcr.io/nvidia/tritonserver:23.03-py3-sdk"
su - opc -c "docker pull nvcr.io/nvidia/tritonserver:21.12-py3"

date