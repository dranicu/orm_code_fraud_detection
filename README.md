# **ORM_stack_a10_gpu_ai_fin**

# **ORM Stack to deploy an A10 shape, one GPU different scenarios for financial services**

- [Intallation](#installation)
- [Note](#note)
- [Systems_monitoring](#systems_monitoring)
- [Jupyter_access](#jupyter_access)
- [Fraud_Detection_models](#fraud_detection_models)

## Installation
- **you can use Resource Manager from OCI console to upload the code from here**
- **once the instance is created, wait the cloud init completion and then you can allow firewall access to be able launch the jupyter notebook interface, commands detailed on both Oracle Linux and Ubuntu in the [Jupyter_access](#jupyter_access)**

- **Jupyter notebook has already configured triton_example_kernel and domino_example_kernel environments** 
- **to switch between them you can open each notebook then go to Kernel -> Change kernel -> Select [domino_example_kernel or triton_example_kernel]**

## NOTE
- **the code deploys an A10 shape with one GPU Shape**
- **based on your need you have the option to either create a new VCN and subnet or you ca use an existing VCN and a subnet where the VM will be deployed**
- **it will add a freeform TAG : "GPU_TAG"= "A10-1"**
- **the boot vol is 500 GB**
- **the cloudinit will do all the steps needed to download and install all Jupyter notebooks and needed Python packages**

## Systems_monitoring
- **Some commands to check the progress of cloudinit completion and GPU resource utilization:**
```
monitor cloud init completion: tail -f /var/log/cloud-init-output.log
monitor single GPU: nvidia-smi dmon -s mu -c 100
                    watch -n 2 'nvidia-smi'
monitor the processes from python for the created environments: watch -n 5 'ps aux | grep python | egrep "triton|domino"'
monitor the system in general: sar 3 1000
```
## Jupyter_access
### Enable access to Jupyter on both Oracle Linux and Ubuntu:

- **Oracle Linux:**
```
sudo firewall-cmd --zone=public --permanent --add-port 8888/tcp
sudo firewall-cmd --reload
sudo firewall-cmd --list-all

!!! in case that you reboot the system you will need to manually start jupyter notebook:
make sure you execute the following commands from /home/opc
conda activate triton_example
> jupyter.log
nohup jupyter notebook --ip=0.0.0.0 --port=8888 > /home/opc/jupyter.log 2>&1 &
then cat jupyter.log to collect the token for the access
```
- **Ubuntu:**
```
sudo iptables -L
sudo iptables -F
sudo iptables-save > /dev/null

If this does not work do also this:
sudo systemctl stop iptables
sudo systemctl disable iptables

sudo systemctl stop netfilter-persistent
sudo systemctl disable netfilter-persistent

sudo iptables -F
sudo iptables-save > /dev/null

!!! in case that you reboot the system you will need to manually start jupyter notebook:
make sure you execute the following commands from /home/ubuntu
conda activate triton_example
> jupyter.log
nohup jupyter notebook --ip=0.0.0.0 --port=8888 > /home/ubuntu/jupyter.log 2>&1 &
then cat jupyter.log to collect the token for the access
```
## Fraud_Detection_models
### The instance contains Jupyter notebooks from the following fraud detection models:
https://domino.ai/blog/credit-card-fraud-detection-using-xgboost-smote-and-threshold-moving
https://www.nvidia.com/en-us/launchpad/data-science/deploy-a-fraud-detection-xgboost-model-using-triton/

#### If you want to execute/modify the notebooks you can use following environments based on the examples location (the notebooks are already saved with the kernel components):
- **Models_for_test_final/Fraud_Tests/ works with triton_example_kernel**
- **Models_for_test_final/XGBOOST_SMOTE_Domino Credit_Card_Fraud_Detection_using_XGBoost_GPU works with domino_example_kernel** 
# **Lab Overview**
- **Once the deployment is successful, the customer can connect to Jupyter Notebook and review the step by step analysis. These lab files guide users through various stages of implementing, deploying, and analyzing financial fraud detection models using advanced machine learning techniques and GPU acceleration. The labs demonstrate the use of XGBoost, SMOTE, Triton Inference Server, and RAPIDS to achieve high-performance fraud detection.**

# **Lab summary info and used files description:**
- **Models_for_test_final/Fraud_Tests/Fraud_Detection_Example presents how to train and deploy an XGBoost fraud detection model using Triton-FIL backend for optimized performance analysis and real-time serving**

- **Models_for_test_final/Fraud_Tests/07-triton-client-inference presents inference with NVIDIA Triton Inference FIL backend: setup, server connection, inference requests, and handling categorical variables**

- **Models_for_test_final/Fraud_Tests/06-accelerating-inference presents installation of Triton client, monitoring server logs, ensuring correct categorical variable handling, and using the same data frame for consistency**

- **Models_for_test_final/Fraud_Tests/05-explaining-predictions presents XGBoost model predictions using SHAP values for better model interpretability, feature importance, and interactions**

- **Models_for_test_final/Fraud_Tests/04-rapids-gpu-pipeline presents how to train and to  evaluate XGBoost models on NVIDIA GPUs with RAPIDS, including data encoding, model storage, performance comparison with CPU, and hyperparameter tuning**

- **Models_for_test_final/Fraud_Tests/03-model-rules-engine presents how to create and evaluate a rule-based fraud detection system, focusing on defining rules, calculating performance metrics, and comparing results with more advanced models**

- **Models_for_test_final/Fraud_Tests/02-visualization presents how to visualize multidimensional data using PCA and UMAP, leveraging RAPIDS and NVIDIA GPUs for dimensionality reduction and interactive exploration**

- **Models_for_test_final/Fraud_Tests/01-eda presents how to explore data through exploratory analysis, including examining variable distributions, identifying patterns, and comparing features like transaction types and amounts to uncover insights before transitioning to more advanced analysis techniques**

- **Models_for_test_final/Fraud_Tests/00-getting-started introduces Jupyter notebooks, covering how to edit and execute cells, use built-in Python functions for help, restart and clean up the kernel, and manage GPU memory**

- **Models_for_test_final/XGBOOST_SMOTE_Domino/Credit_Card_Fraud_Detection_using_XGBoost_GPU explores a credit card fraud detection approach using XGBoost, SMOTE, and threshold moving. It begins with preprocessing data to handle class imbalance with SMOTE, followed by training an XGBoost model to detect fraud. The key enhancement involves adjusting the decision threshold to improve performance metrics like precision and recall, resulting in better fraud detection accuracy. The approach emphasizes the importance of threshold tuning for optimizing model performance in fraud detection scenarios.**

## **Lab files and detailed description**
- **1. Lab Models_for_test_final/Fraud_Tests/Fraud_Detection_Example: utilizes IEEE-CIS Fraud Detection dataset and trains XGBoost models, supporting categorical variables. Models are serialized for Triton Inference Server, deployed on GPUs for higher throughput and lower latency. Triton's tools like perf_analyzer are used for tuning and comparing CPU vs GPU performance.**

- **2. Lab Models_for_test_final/06-accelerating-inference: install Triton client requirements. Connect to Triton server, ensuring it is online and error-free. Deploy models and submit requests, handling categorical variables correctly.**

- **3. Lab Models_for_test_final/05-explaining-predictions: explain model predictions and feature importance using SHAP values. Compute feature importance on GPU and identify feature interactions. Plot feature importance and interactions, comparing GPU vs CPU performance.**

- **4. Lab Models_for_test_final/04-rapids-gpu-pipeline: import RAPIDS libraries, load and split data. Perform one-hot and target encoding for categorical features, impute and scale numerical features. Train XGBoost model on GPU, store and serialize model. Compare GPU vs CPU training time and accuracy, evaluate metrics, and adjust sample size. Experiment with XGBoost parameters to optimize performance and reduce false positives.**

- **5. Lab Models_for_test_final/Fraud_Tests/03-model-rules-engine: define simple rules for fraud detection (e.g., amount > $200), evaluate performance using precision and recall, refine rules, and compare rule-based performance with advanced models.**

- **6. Lab Models_for_test_final/Fraud_Tests/02-visualization: reduce dimensions, emphasize variance, requiring scaled numeric features and encoded categorical data. Nonlinear dimensionality reduction with key hyperparameters like n_neighbors and min_dist.**

- **7. Lab Models_for_test_final/Fraud_Tests/01-eda: examine data structure and column types, identify categorical and continuous variables. Analyze distribution of transaction types and foreign vs. domestic transactions. Evaluate transaction amounts, explore distributions, and assess time gaps between transactions.**

- **8. Lab Models_for_test_final/Fraud_Tests/00-getting-started: different types (prose, code, UI elements), editing and execution. How to restart the kernel and run up to the selected cell.**

- **9. Lab Models_for_test_final/XGBOOST_SMOTE_Domino/Credit_Card_Fraud_Detection_using_XGBoost_GPU: XGBoost, SMOTE, and threshold moving for classification. Handle class imbalance with SMOTE, train XGBoost classifier, and adjust thresholds for better performance. Improved accuracy, effective handling of imbalance, and optimized performance through threshold tuning.**
