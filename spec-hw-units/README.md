# Install
We suggest to create a new python virtual environment and install the required packages using the following commands:
```bash
# Create a new python virtual environment and activate it
python3 -m venv darkside_venv
source darkside_venv/bin/activate

# Install the required packages
python3 -m pip install -r requirements.txt

# Install the current package
python3 -m pip install .
```

# Usage

## CIFAR-10
First, navigate to the `exp/icl` directory.
Then, use the `run.sh` script to run the *Warmup*, *Search* and *Final Training* phases of ODiMO using the Darkside SoC as Hardware target.
```bash
source run.sh {strength} mbv1_search_32 half fine now 42 {metric}
```
where `{strength}` is the regularization strength (λ in the manuscript) and `{metric}` is the cost metric used to evaluate the models, it can be either `darkside` (i.e., latency) or `darkside-power`.

N.B., the script will automatically download the CIFAR-10 dataset.

## CIFAR-100
First, navigate to the `exp/cifar100` directory.
Then, use the `run_ft.sh` script to run the *Warmup*, *Search* and *Final Training* phases of ODiMO using the Darkside SoC as Hardware target.
```bash
source run_ft.sh {strength} mbv1_search_32 half fine now 42 {metric} 1 no_ckp_wup no_ckp_search no_ckp_ft
```
where `{strength}` is the regularization strength (λ in the manuscript) and `{metric}` is the cost metric used to evaluate the models, it can be either `darkside` (i.e., latency) or `darkside-power`.

N.B., the script will automatically download the CIFAR-100 dataset.


## ImageNet
First, navigate to the `exp/imagenet` directory.
Then, use the `run_ft.sh` script to run the *Warmup*, *Search* and *Final Training* phases of ODiMO using the Darkside SoC as Hardware target.
```bash
source run_ft.sh {strength} mbv1_search_32 half fine now 42 {metric} 2 no_ckp_wup no_ckp_search no_ckp_ft
```
where `{strength}` is the regularization strength (λ in the manuscript) and `{metric}` is the cost metric used to evaluate the models, it can be either `darkside` (i.e., latency) or `darkside-power`.

N.B., in this case the data need to be manually downloaded from the [ImageNet website](http://image-net.org/download) and the path to the dataset should be specified in the `run_ic.sh` script at line 4.

