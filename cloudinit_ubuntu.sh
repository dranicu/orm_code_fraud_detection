#!/bin/bash

# AI_financial_test
echo "running cloudinit.sh script"

apt-get update -y
apt-get install -y dnf-utils zip unzip gcc curl openssl libssl-dev libbz2-dev libffi-dev zlib1g-dev wget make git

echo "INSTALL NVIDIA CUDA + TOOLKIT + drivers"
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update -y
apt-get -y install cuda-toolkit-12-5
apt-get install -y nvidia-driver-555
apt-get -y install cudnn
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit

# Add Docker repository and install Docker
apt-get remove -y runc
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io

echo "ENABLE DOCKER"
systemctl enable docker.service

echo "START DOCKER"
systemctl start docker.service


echo "PYTHON packages"
apt-get install -y python3-pip
python3 -m pip install --upgrade pip wheel oci
python3 -m pip install --upgrade setuptools
python3 -m pip install oci-cli langchain python-multipart pypdf six

echo "GROWFS"
growpart /dev/sda 1
resize2fs /dev/sda1

echo "Export nvcc"
echo "export PATH=\$PATH:/usr/local/cuda/bin" >> /home/ubuntu/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> /home/ubuntu/.bashrc

echo "Add docker ubuntu"
usermod -aG docker ubuntu

echo "Python 3.10.6"
wget https://www.python.org/ftp/python/3.10.6/Python-3.10.6.tar.xz
tar -xf Python-3.10.6.tar.xz
cd Python-3.10.6/
./configure --enable-optimizations
make -j $(nproc)
make altinstall
python3.10 -V
cd ..
rm -rf Python-3.10.6*

echo "Conda"
mkdir -p /home/ubuntu/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/ubuntu/miniconda3/miniconda.sh
bash /home/ubuntu/miniconda3/miniconda.sh -b -u -p /home/ubuntu/miniconda3
rm -rf /home/ubuntu/miniconda3/miniconda.sh
/home/ubuntu/miniconda3/bin/conda init bash
chown -R ubuntu:ubuntu /home/ubuntu/miniconda3
chown ubuntu:ubuntu /home/ubuntu/.bashrc

echo "Creating Conda environment"

# Create the YAML file and redirect it to env_triton.yaml
su - ubuntu -c "cat <<EOF > /home/ubuntu/env_triton.yaml
name: triton_example
channels:
  - conda-forge
  - nvidia
  - rapidsai
dependencies:
  - cudatoolkit=11.4
  - cudf=21.12
  - cuml=21.12
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
      - ipykernel
EOF"


# Create the YAML file and redirect it to env_domino.yaml
su - ubuntu -c "cat <<EOF > /home/ubuntu/env_domino.yaml
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
      - ipykernel
EOF"

# Create the YAML file and redirect it to env_triton_final
su - ubuntu -c "cat <<EOF > /home/ubuntu/env_triton_final.yaml
name: triton_example_final
channels:
  - nvidia/label/cuda-12.5.1
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - _libgcc_mutex=0.1=conda_forge
  - _openmp_mutex=4.5=2_gnu
  - _sysroot_linux-64_curr_repodata_hack=3=h69a702a_16
  - binutils_impl_linux-64=2.40=ha1999f0_7
  - binutils_linux-64=2.40=hb3c18ed_0
  - bzip2=1.0.8=h4bc722e_7
  - ca-certificates=2024.7.4=hbcca054_0
  - cuda-cccl=12.5.39=0
  - cuda-cccl_linux-64=12.5.39=0
  - cuda-command-line-tools=12.5.1=0
  - cuda-compiler=12.5.1=0
  - cuda-crt-dev_linux-64=12.5.82=ha770c72_0
  - cuda-crt-tools=12.5.82=ha770c72_0
  - cuda-cudart=12.5.82=0
  - cuda-cudart-dev=12.5.82=0
  - cuda-cudart-dev_linux-64=12.5.82=0
  - cuda-cudart-static=12.5.82=0
  - cuda-cudart-static_linux-64=12.5.82=0
  - cuda-cudart_linux-64=12.5.82=0
  - cuda-cuobjdump=12.5.39=0
  - cuda-cupti=12.5.82=0
  - cuda-cupti-dev=12.5.82=0
  - cuda-cuxxfilt=12.5.82=0
  - cuda-driver-dev=12.5.82=0
  - cuda-driver-dev_linux-64=12.5.82=0
  - cuda-gdb=12.5.82=0
  - cuda-libraries=12.5.1=0
  - cuda-libraries-dev=12.5.1=0
  - cuda-nsight=12.5.82=0
  - cuda-nvcc=12.5.82=0
  - cuda-nvcc-dev_linux-64=12.5.82=ha770c72_0
  - cuda-nvcc-impl=12.5.82=hd3aeb46_0
  - cuda-nvcc-tools=12.5.82=hd3aeb46_0
  - cuda-nvcc_linux-64=12.5.82=0
  - cuda-nvdisasm=12.5.39=0
  - cuda-nvml-dev=12.5.82=0
  - cuda-nvprof=12.5.82=0
  - cuda-nvprune=12.5.82=0
  - cuda-nvrtc=12.5.82=0
  - cuda-nvrtc-dev=12.5.82=0
  - cuda-nvtx=12.5.82=0
  - cuda-nvvm-dev_linux-64=12.5.82=ha770c72_0
  - cuda-nvvm-impl=12.5.82=h59595ed_0
  - cuda-nvvm-tools=12.5.82=h59595ed_0
  - cuda-nvvp=12.5.82=0
  - cuda-opencl=12.5.39=0
  - cuda-opencl-dev=12.5.39=0
  - cuda-profiler-api=12.5.39=0
  - cuda-sanitizer-api=12.5.81=0
  - cuda-toolkit=12.5.1=0
  - cuda-tools=12.5.1=0
  - cuda-version=12.5=3
  - cuda-visual-tools=12.5.1=0
  - dbus=1.13.18=hb2f20db_0
  - expat=2.6.2=h59595ed_0
  - fontconfig=2.14.2=h14ed4e7_0
  - freetype=2.12.1=h267a509_2
  - gcc_impl_linux-64=13.3.0=hfea6d02_0
  - gcc_linux-64=13.3.0=hc28eda2_0
  - gds-tools=1.10.1.7=0
  - glib=2.80.3=h8a4344b_1
  - glib-tools=2.80.3=h73ef956_1
  - gmp=6.3.0=hac33072_2
  - gxx_impl_linux-64=13.3.0=hffce095_0
  - gxx_linux-64=13.3.0=h6834431_0
  - icu=75.1=he02047a_0
  - kernel-headers_linux-64=3.10.0=h4a8ded7_16
  - ld_impl_linux-64=2.40=hf3520f5_7
  - libcublas=12.5.3.2=0
  - libcublas-dev=12.5.3.2=0
  - libcufft=11.2.3.61=0
  - libcufft-dev=11.2.3.61=0
  - libcufile=1.10.1.7=0
  - libcufile-dev=1.10.1.7=0
  - libcurand=10.3.6.82=0
  - libcurand-dev=10.3.6.82=0
  - libcusolver=11.6.3.83=0
  - libcusolver-dev=11.6.3.83=0
  - libcusparse=12.5.1.3=0
  - libcusparse-dev=12.5.1.3=0
  - libexpat=2.6.2=h59595ed_0
  - libffi=3.4.2=h7f98852_5
  - libgcc-devel_linux-64=13.3.0=h84ea5a7_100
  - libgcc-ng=14.1.0=h77fa898_0
  - libglib=2.80.3=h8a4344b_1
  - libgomp=14.1.0=h77fa898_0
  - libiconv=1.17=hd590300_2
  - libnpp=12.3.0.159=0
  - libnpp-dev=12.3.0.159=0
  - libnsl=2.0.1=hd590300_0
  - libnvfatbin=12.5.82=0
  - libnvfatbin-dev=12.5.82=0
  - libnvjitlink=12.5.82=0
  - libnvjitlink-dev=12.5.82=0
  - libnvjpeg=12.3.2.81=0
  - libnvjpeg-dev=12.3.2.81=0
  - libpng=1.6.43=h2797004_0
  - libsanitizer=13.3.0=heb74ff8_0
  - libsqlite=3.46.0=hde9e2c9_0
  - libstdcxx-devel_linux-64=13.3.0=h84ea5a7_100
  - libstdcxx-ng=14.1.0=hc0a3c3a_0
  - libuuid=2.38.1=h0b41bf4_0
  - libxcb=1.16=hd590300_0
  - libxcrypt=4.4.36=hd590300_1
  - libxkbcommon=1.7.0=h2c5496b_1
  - libxml2=2.12.7=he7c6b58_4
  - libzlib=1.3.1=h4ab18f5_1
  - ncurses=6.5=h59595ed_0
  - nsight-compute=2024.2.1.2=2
  - nspr=4.35=h27087fc_0
  - nss=3.103=h593d115_0
  - openssl=3.3.1=h4bc722e_2
  - pcre2=10.44=h0f59acf_0
  - pip=24.2=pyhd8ed1ab_0
  - pthread-stubs=0.4=h36c2ea0_1001
  - python=3.8.19=hd12c33a_0_cpython
  - readline=8.2=h8228510_1
  - setuptools=72.1.0=pyhd8ed1ab_0
  - sysroot_linux-64=2.17=h4a8ded7_16
  - tk=8.6.13=noxft_h4845f30_101
  - wheel=0.43.0=pyhd8ed1ab_1
  - xkeyboard-config=2.42=h4ab18f5_0
  - xorg-kbproto=1.0.7=h7f98852_1002
  - xorg-libx11=1.8.9=hb711507_1
  - xorg-libxau=1.0.11=hd590300_0
  - xorg-libxdmcp=1.1.3=h7f98852_0
  - xorg-xextproto=7.3.0=h0b41bf4_1003
  - xorg-xproto=7.0.31=h7f98852_1007
  - xz=5.2.6=h166bdaf_0
  - pip:
      - absl-py==2.1.0
      - aiohappyeyeballs==2.3.4
      - aiohttp==3.10.0
      - aiosignal==1.3.1
      - asttokens==2.4.1
      - astunparse==1.6.3
      - async-timeout==4.0.3
      - attrs==23.2.0
      - backcall==0.2.0
      - brotli==1.1.0
      - cachetools==5.4.0
      - certifi==2024.7.4
      - charset-normalizer==3.3.2
      - comm==0.2.2
      - cramjam==2.8.3
      - cuda-python==12.3.0
      - debugpy==1.8.2
      - decorator==5.1.1
      - executing==2.0.1
      - fastparquet==2024.2.0
      - flatbuffers==24.3.25
      - frozenlist==1.4.1
      - fsspec==2024.6.1
      - gast==0.4.0
      - gevent==24.2.1
      - geventhttpclient==2.0.2
      - google-auth==2.32.0
      - google-auth-oauthlib==1.0.0
      - google-pasta==0.2.0
      - greenlet==3.0.3
      - grpcio==1.65.2
      - h5py==3.11.0
      - idna==3.7
      - importlib-metadata==8.2.0
      - ipykernel==6.29.5
      - ipython==8.12.3
      - jedi==0.19.1
      - jupyter-client==8.6.2
      - jupyter-core==5.7.2
      - keras==2.13.1
      - libclang==18.1.1
      - markdown==3.6
      - markupsafe==2.1.5
      - matplotlib-inline==0.1.7
      - multidict==6.0.5
      - nest-asyncio==1.6.0
      - numpy==1.24.3
      - oauthlib==3.2.2
      - opt-einsum==3.3.0
      - packaging==24.1
      - pandas==2.0.3
      - parso==0.8.4
      - pexpect==4.9.0
      - pickleshare==0.7.5
      - platformdirs==4.2.2
      - prompt-toolkit==3.0.47
      - protobuf==4.25.4
      - psutil==6.0.0
      - ptyprocess==0.7.0
      - pure-eval==0.2.3
      - pyarrow==17.0.0
      - pyasn1==0.6.0
      - pyasn1-modules==0.4.0
      - pygments==2.18.0
      - python-dateutil==2.9.0.post0
      - python-rapidjson==1.19
      - pytz==2024.1
      - pyzmq==26.0.3
      - requests==2.32.3
      - requests-oauthlib==2.0.0
      - rsa==4.9
      - six==1.16.0
      - stack-data==0.6.3
      - tensorboard==2.13.0
      - tensorboard-data-server==0.7.2
      - tensorflow==2.13.1
      - tensorflow-estimator==2.13.0
      - tensorflow-io-gcs-filesystem==0.34.0
      - termcolor==2.4.0
      - tornado==6.4.1
      - traitlets==5.14.3
      - tritonclient==2.48.0
      - typing-extensions==4.5.0
      - tzdata==2024.1
      - urllib3==2.2.2
      - wcwidth==0.2.13
      - werkzeug==3.0.3
      - wrapt==1.16.0
      - yarl==1.9.4
      - zipp==3.19.2
      - zope-event==5.0
      - zope-interface==6.4.post2
EOF"

# Create the Conda environment using the newly created YAML file
sudo -u ubuntu -i bash -c "/home/ubuntu/miniconda3/bin/conda env create -f /home/ubuntu/env_triton.yaml"

# echo "source activate triton_example" >> /home/ubuntu/.bashrc
echo "Starting Jupyter Notebook server as ubuntu and triton_example env"

# Initialize Conda for the ubuntu user
sudo -u ubuntu -i bash -c "/home/ubuntu/miniconda3/condabin/conda init"

sudo -u ubuntu -i bash -c "source /home/ubuntu/miniconda3/bin/activate triton_example && \
                           nohup jupyter notebook --ip=0.0.0.0 --port=8888 > /home/ubuntu/jupyter.log 2>&1 &"

echo "Creating kernel execution for Jupyter domino_example"

# Create the Conda env_domino environment
sudo -u ubuntu -i bash -c "/home/ubuntu/miniconda3/bin/conda env create -f /home/ubuntu/env_domino.yaml"

# # Install ipykernel in the created environments
# sudo -u ubuntu -i bash -c "/home/ubuntu/miniconda3/bin/conda run -n domino_example /home/ubuntu/miniconda3/bin/conda install ipykernel"
# sudo -u ubuntu -i bash -c "/home/ubuntu/miniconda3/bin/conda run -n triton_example /home/ubuntu/miniconda3/bin/conda install ipykernel"

# Install the Jupyter kernel
sudo -u ubuntu -i bash -c "/home/ubuntu/miniconda3/bin/conda run -n domino_example python -m ipykernel install --user --name domino_example --display-name 'domino_example_kernel'"
sudo -u ubuntu -i bash -c "/home/ubuntu/miniconda3/bin/conda run -n triton_example python -m ipykernel install --user --name triton_example --display-name 'triton_example_kernel'"
sudo -u ubuntu -i bash -c "/home/ubuntu/miniconda3/bin/conda run -n triton_example_final python -m ipykernel install --user --name triton_example --display-name 'triton_example_final_kernel'"

# Get the model files
sudo -u ubuntu -i bash -c "wget https://objectstorage.il-jerusalem-1.oraclecloud.com/p/pyGEGe7DpwPRzMjT6arIwe7O1aH5rjxMsyeC7Y8Z17Dt9ai4Nk8Re8UYHELVB5nT/n/orasenatdpltintegration03/b/bucket_for_bogdan/o/AI/Notebook/models_for_test_final.zip"
apt-get install -y unzip
sudo -u ubuntu -i bash -c "unzip models_for_test_final.zip"

sudo -u ubuntu -i bash -c "docker pull nvcr.io/nvidia/tritonserver:21.12-py3"
sudo -u ubuntu -i bash -c "docker pull nvcr.io/nvidia/tritonserver:23.03-py3-sdk"

su - ubuntu -c "sudo nvidia-smi"
date