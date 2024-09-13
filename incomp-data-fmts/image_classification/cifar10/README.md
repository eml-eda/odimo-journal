# Install
We suggest to create a new python environment and install the required packages using the following commands:
```bash
# Create and activate a new python environment
python3 -m venv diana_env
source diana_env/bin/activate

# Install the required packages
pip install -r requirements.txt

# Install the current package
python setup.py install
```

# Usage

## CIFAR-10
First, move to the `/image_classification/cifar10` directory.
Then, run the following shell script to perform *Search* phase and *Final Training* phase over the CIFAR-10 dataset using DIANA SoC as target hardware:
```bash
source run_ic_fulltest.sh {strength} res20_pow2_diana_full {metric} now search ft
```
where `{strength}` is the regularization strength (λ in the manuscrpit), `{metric}` is the cost metric used to drive the mapping process. It could be either `latency` or `power`.

N.B., the dataset is automatically downloaded by the script.

## CIFAR-100
First, move to the `/image_classification/cifar100` directory.
Then, run the following shell script to perform *Search* phase and *Final Training* phase over the CIFAR-100 dataset using DIANA SoC as target hardware:
```bash
source run_ic.sh {strength} res18_pow2_diana_full_c100 {metric} now search ft
```
where `{strength}` is the regularization strength (λ in the manuscrpit), `{metric}` is the cost metric used to drive the mapping process. It could be either `latency` or `power`.

N.B., the dataset is automatically downloaded by the script.

## ImageNet
First, move to the `/image_classification/imagenet` directory.
Then, run the following shell script to perform *Search* phase and *Final Training* phase over the CIFAR-100 dataset using DIANA SoC as target hardware:
```bash
source run_ic.sh {strength} res18_diana_full {metric} now search ft
```
where `{strength}` is the regularization strength (λ in the manuscrpit), `{metric}` is the cost metric used to drive the mapping process. It could be either `latency` or `power`.

N.B., the dataset **need to be downloaded** independently from the [ImageNet website](http://www.image-net.org/). Then, the path where data are located need to be specified in the `run_ic.sh` script at line 4 `data="{path-goes-here}"`.

