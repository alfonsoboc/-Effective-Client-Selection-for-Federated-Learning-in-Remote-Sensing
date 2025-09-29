# Effective Client Selection for Federated Learning in Remote Sensing

The official implementation of the Paper: *FedCor: Correlation-Based Active Client Selection Strategy for Heterogeneous Federated Learning* (CVPR 22') and our FedSignal strategy.
IMPORTANT!!!
 Most of these scripts come from the Github Yoruko-Tang/FedCor, with some changes for the correct implementation of FedSignal, called here danfonis.
danfonis.py is the only 100% original script, and the modified ones by us where federated_main.py utils.py sampling.py and options.py

**Abstract:** Federated learning (FL) enables the collaboration
of multiple deep learning models to learn from decentralized
data archives (i.e., clients) without accessing data on clients.
Although FL offers ample opportunities in knowledge discovery
from distributed image archives, it is seldom considered in
remote sensing (RS). In this paper, we consider client selection
strategies as a tool to mitigate the adverse effects of non-IID
data distributions. In particular, we examine the implementation
of the FedCor algorithm in the context of RS and introduce a
new client selection strategy: FedSignal. After explaining these
strategies within the theoretical framework, we present our
experimental results on the EuroSAT dataset, divided into various
decentralization scenarios. Our results demonstrate the effectiveness of FedSignal in improving convergence rates and model
accuracy under different levels of data heterogeneity. Based on
our comprehensive analysis, we derive guidelines for selecting
suitable client selection strategies in FL for RS applications.


## Required Environment

1. python 3.8
2. pytorch 1.7.0
6. cvxopt 1.2.0

## Running Experiments

Cd to the root directory of the repository and run the following command lines to start a training on Dir setting with different client selection strategies. If you want to train with 2SPC or 1SPC setting, you can replace the option ```--alpha=0.2``` with ```--shards_per_client=2``` or ```--shards_per_client=1``` respectively.

For more details about each option, see ```/src/options.py```.

Even if the code includes more datasets (FMNIST, CIFAR10, CIFAR100...), none of them are useful for the experiments, so only take in consideration the eurosat experiments (only dataset suitable for remote sensing context).


### EUROSAT(RGB)
 These are examples for running the code
 IMPORTANT: Before running the code, change in utils.py the direction of the data for the use of eurosat. In the moment, the data is stored in 'C:\\Users\\a\\OneDrive\\Escritorio\\PCVRS_submission\\src\\data'
Modify the variable data_dir!


#### Random selection

```shell
python3 src/federated_main.py --gpu=0 --dataset=eurosat --model=resnet --epochs=2000 --num_user=10 --frac=0.05 --local_ep=5 --local_bs=50 --lr=0.01 --lr_decay=1.0 --optimizer=sgd --reg=3e-4 --iid=0 --unequal=0 --alpha=0.2 --verbose=1 --seed 1 
```

#### FedCor

```shell
python3 src/federated_main.py --gpu=0 --gpr_gpu=0 --dataset=eurosat --model=resnet  --epochs=2000 --num_user=10 --frac=0.05 --local_ep=5 --local_bs=50 --lr=0.01 --lr_decay=1.0 --optimizer=sgd --reg=3e-4 --iid=0 --unequal=0 --alpha=0.2 --verbose=1 --seed 1 --gpr --discount=0.9 --GPR_interval=50 --group_size=500 --warmup=20
```

#### FedSignal

```shell
python3 src/federated_main.py --gpu=0 --dataset=cifar --danfonis --model=resnet  --epochs=2000 --num_user=10 --frac=0.05 --local_ep=5 --local_bs=50 --lr=0.01 --lr_decay=1.0 --optimizer=sgd --reg=3e-4 --iid=0 --unequal=0 --alpha=0.2  --seed 1 
```


Instructions for Utilizing Decentralization Scenarios
To employ the various decentralization scenarios described in this report, please adhere to the following instructions:

Eurosat_iid Scenario:

Set the iid argument to 1 by using the command: --iid=1.
Eurosat_noniid Scenario:

Set the iid argument to 0 by using the command: --iid=0.
Specify the number of shards per client with the argument: --shards_per_client=N, where 
𝑁
N represents the desired number of shards, as detailed in Appendix I of this report. This adjustment facilitates the creation of different decentralization scenarios.
If the iid argument is set to 0 without specifying the shards_per_client argument, the system will default to a scenario where each client receives images from only one class.
Important Considerations:

The number of clients should be set to 10 to effectively manage non-IID (non-independent and identically distributed) data scenarios.
It is crucial to maintain consistent hyperparameters across experiments to ensure that the focus remains on evaluating the performance of different strategies under similar conditions.

