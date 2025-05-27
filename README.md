# [AI4MAT-ICLR-2025 Spotlight] Towards Extrapolation in Deep Material Property Rregression
This is the official implementation code for AI4MAT-ICLR-2025 Spotlight paper MEX.

Here is the visualization of the Matching-based EXtrapolation (MEX) framework.

![overview](https://github.com/panmianzhi/Matching-based-EXtrapolation/blob/main/imgs/method.png)

## Datasets
All datasets used in this work are available in the [dataset](https://github.com/panmianzhi/Matching-based-EXtrapolation/tree/main/data) directory.

## Model Architecture
The backbone Equiformer-V2 model is in [equiformer_v2_oc20.py](https://github.com/panmianzhi/Matching-based-EXtrapolation/blob/main/ocp/ocpmodels/models/equiformer_v2/equiformer_v2_oc20.py).

The training and inference algorithms are implemented in [mex_trainer.py](https://github.com/panmianzhi/Matching-based-EXtrapolation/blob/main/ocp/ocpmodels/trainers/mex_trainer.py).

## Acknowledgement
The structure of this repository is based on [fairchem](https://github.com/FAIR-Chem/fairchem).