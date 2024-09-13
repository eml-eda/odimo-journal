# Install
We suggest to create a new python virtual environment and install the required packages using the following commands:
```bash
# Create a new python virtual environment and activate it
python3 -m venv diana_venv
source diana_venv/bin/activate

# Install the required packages
pip install -r requirements.txt

# Install the current package
python setup.py install
```

# Usage

## CIFAR-10
First, navigate to the `image_classification/cifar10` directory.
Then, use the `run_ic_fulltest.sh` script to run the *Search* and *Final Training* phases of ODiMO using the DIANA SoC as Hardware target.
```bash
source run_ic_fulltest.sh {strength} res20_pow2_diana_full {metric} now search ft
```
where `{strength}` is the regularization strength (λ in the manuscript) and `{metric}` is the cost metric used to evaluate the models, it can be either `latency` or `power`.

N.B., the script will automatically download the CIFAR-10 dataset.

## CIFAR-100
First, navigate to the `image_classification/cifar100` directory.
Then, use the `run_ic.sh` script to run the *Search* and *Final Training* phases of ODiMO using the DIANA SoC as Hardware target.
```bash
source run_ic.sh {strength} res18_pow2_diana_full_c100 {metric} now search ft
```
where `{strength}` is the regularization strength (λ in the manuscript) and `{metric}` is the cost metric used to evaluate the models, it can be either `latency` or `power`.

N.B., the script will automatically download the CIFAR-100 dataset.


## ImageNet
First, navigate to the `image_classification/imagenet` directory.
Then, use the `run_ic.sh` script to run the *Search* and *Final Training* phases of ODiMO using the DIANA SoC as Hardware target.
```bash
source run_ic.sh {strength} res18_diana_full {metric} now search ft
```
where `{strength}` is the regularization strength (λ in the manuscript) and `{metric}` is the cost metric used to evaluate the models, it can be either `latency` or `power`.

N.B., in this case the data need to be manually downloaded from the [ImageNet website](http://image-net.org/download) and the path to the dataset should be specified in the `run_ic.sh` script at line 4.

