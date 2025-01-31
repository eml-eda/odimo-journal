# Table of Contents
- [Install](#install)
- [Usage](#usage)
    - [CIFAR-10](#cifar-10)
    - [CIFAR-100](#cifar-100)
    - [ImageNet](#imagenet)
- [Hardware Models](#hardware-models)
- [Training Protocol](#training-protocol)
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

# Hardware Models
The DIANA latency models take into account the cycles required for loading weights and executing GEMMs/Convolutions (accounting for each CU's parallelism dimension and unrolling factor). Non-idealities such as programming overheads and memory stalls are ignored. Similarly, input loading and output storage latencies are ignored, given that they could be usually double-buffered with computation. Nonetheless, as experimentally shown in Sec. V of the paper, this model correlates well with actual latencies measured on the hardware. While the error is, in general, high, both the models for the digital and the analog accelerators neglect the same latency component, therefore showing an almost constant relative underestimation error.

Considering a Convolutional layer, without loss of generality, the latency model for the digital accelerator is:

$$
    LAT_{dig}^{(l)}(\theta) = \lceil \frac{C_{out, dig}^{(l)}(\theta)}{16} \rceil \lceil \frac{o_y^{(l)}}{16} \rceil \times C_{in}^{(l)} \times o_x^{(l)} \times f_x^{(l)} \times f_y^{(l)} + \\
                    C_{in}^{(l)} \times C_{out, dig}^{(l)}(\theta) \times f_x^{(l)} \times f_y^{(l)}
$$
where $C_{in}^{(l)}$, $o_x^{(l)}$/$o_y^{(l)}$ and $f_x^{(l)}$/$f_y^{(l)}$ are the layer's input channels, output spatial dimensions, and kernel sizes respectively. $C_{out, dig}^{(l)}$ is the number of output channels assigned by ODiMO to the digital CU (as a function of the learned $\theta$ parameters). The two addends in the latency model simply compute the number of MAC and DMA cycles respectively, where the former considers the spatial parallelism of the accelerator (16x16 PEs).

The model for the AIMC CU is:
$$
    LAT_{aimc}^{(l)}(\theta) = \lceil \frac{C_{in}^{(l)} \times f_x^{(l)} \times f_y^{(l)}}{1152} \rceil \lceil \frac{C_{out, aimc}^{(l)}(\theta)}{512} \rceil \times o_x^{(l)} \times o_y^{(l)}  \\ +
                      2 \times 4 \times C_{in}^{(l)} \times \lceil \frac{C_{out, aimc}^{(l)}(\theta)}{512} \rceil
$$
also in this case the first term models the number of MAC operations as function of the CU's parallelism (1152$\times$512) and the second one accounts for DMA cycles.

The two detailed latency models can be used to build the energy consumption optimization target of Eq. 4 of the paper. In particular, we only need the average power consumption of the two CUs when active and idle. Then, for a single layer $l$, and making the dependency from $\theta$ implicity for clarity, the regularization target becomes:
$$
    \mathcal{C}_{en} = \: P_{act, \: dig} \cdot LAT_{dig} + P_{act, \: ana} \cdot LAT_{ana} \\
    + P_{idle} \cdot \max(LAT_{dig}, LAT_{ana})
$$
The overall regularization term of the DNN will be obtained by summing a contribution in this form  for each layer $l$.

# Training Protocol
Here, we detail how the general training protocol introduced in Sec. IV of the paper is customized for the case of DIANA.
The Warmup consists of training the given DNN in floating-point. Then, since the DIANA accelerators do not implement Batch Normalization (BN) in hardware, the BN layers are folded with Conv/FC.
To simulate the effect of quantization during the optimization, we follow the scheme of [FQ-Conv](https://arxiv.org/abs/1912.09356):
$$
    Q(x) = \frac{e^{s}}{2^{n-1}-1} \cdot \mathrm{round}(2^{n-1} - 1 \cdot \mathrm{clip}(x, -1, 1))
$$
where $s$ is a trainable scale parameter and $n$ is the bit-width.
With reference to above equation, we use $n=8$ for the digital accelerator weights, while $n = 2$ is used to perform ternarization i.e., the quantization format of DIANA's AIMC accelerator weights. Concerning activations, the AIMC and digital blocks have slightly different formats on 7- and 8-bit respectively. During the optimization phase, we use the worst case of the two (7-bit) as fake-quantization bit-width for layers' inputs/outputs.
As long as the DNN is appropriately fine-tuned (see below), we found this approximation not to degrade our results.

During the Search phase, the fake-quantized DNN is optimized until convergence, with an early-stop mechanism.
Then, after discretizing the final channel assignment to each CU, the model undergoes the final Final Training step. In this phase, we use the exact quantization format also for activations, i.e., shared data are stored on 8-bit but the AIMC accelerator D/A and A/D converters are on 7-bit, effectively truncating the LSB of inputs/outputs.