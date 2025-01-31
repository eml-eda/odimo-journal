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

# Hardware Models
Darkside includes a cluster of eight SIMD-enabled general-purpose RISC-V cores coupled with an HW accelerator, the DWE, to execute Depthwise Convolutions with high arithmetic intensity.
The latency model of the DWE is defined as:
$$
    LAT_{DWE}^{(l)}(\theta) = \lfloor \frac{C_{out, DWE}^{(l)}(\theta) + 15}{16} \rfloor  \times (o_x o_y \tau_{\text{comp}} + o_x \tau_i + \tau_w)
$$
where $C_{out, DWE}^{(l)} = C_{in, DWE}^{(l)}$ denotes the number of output/input channels of the layer. $\tau_{\text{comp}}, \tau_i, \tau_w$ are constants set respectively to $4, 9,$ and $9$ that take into account the number of cycles necessary to compute the fraction of the output pixels corresponding to the loaded input, load a new input portion and load the necessary weights.
This model only neglects minor latency contributions, such as programming overheads and conflicts in the memory interconnect.

Conversely, the model for the RISC-V cluster is derived by analyzing the cluster structure proposed in the reference paper, characterized by cores with MAC-and-Load units and a dedicated data mover for data transposition, starting from the original software routine that implements Conv layers, proposed in [PULP-NN](https://royalsocietypublishing.org/doi/10.1098/rsta.2019.0155). The model is:
$$
    LAT_{\text{clust}}^{(l)}(\theta) = \lfloor \frac{o_{y}^{(l)}+1}{2} \rfloor \lfloor \frac{o_x^{(l)} + 7}{8} \rfloor \times \\
                    \lfloor \frac{C_{out, clust}^{(l)}(\theta)+3}{4} \rfloor (15 + 8 \lfloor \frac{f_x^{(l)} f_y^{(l)} C_{in}^{(l)} + 3}{4} \rfloor)
$$
where the factor within round brackets takes into account the cycles required to perform matrix multiplications and output activation operations. The term multiplied to it is the number of iterations required to produce the complete output feature map, which accounts for the parallelization over the eight cores performed on the ox dimension and for the inner loop unrolling (with a factor of 2) over oy.
While the model is accurate in estimating the cluster performance, the real software implementation, which is not open-source, can slightly deviate from this model.

The energy consumption optimization target for the Darkside platform can be derived starting from Eq. 4 of the paper by plugging in the analytical latency model as done for the DIANA platform in its [README](../incomp-data-fmts/README.md).

# Training Protocol
The SuperNet used by ODiMO to optimize the layer selection for Darkside includes a normal Conv and a DW Conv as alternatives for each layer, matching the cluster and DWE CUs respectively. Before Warmup, the weights of both alternatives are randomly initialized. To ensure that all paths of the supernet are properly trained, in each iteration of the Warmup phase, we uniformly sample a random number of channels from each of the two alternatives, independently for every layer.

During the Search phase, to avoid the fast convergence of the optimization algorithm towards solutions with low cost but degraded performance, we found it beneficial to implement a strength-scheduling approach, similar to the one proposed [DUCCIO](https://ieeexplore.ieee.org/document/10278089), where the target value of the regularization strength $\lambda$ is divided by $100$ and linearly increased over the epochs until the original value is reached, then it is kept constant.

At the end of the search phase, the mapping is discretized and the final discovered network is trained *from-scratch* by optimizing only the task loss $\mathcal{L}$ with respect to the weights $W$. Differently for DIANA, where fine-tuning was sufficient, in this case, we foundwthat training from scratch is beneficial.