# Metaheuristic-Guided Active learning for Optimizing Reaction Conditions of High-Performance Nonoxidative Methane Conversion
Converting methane into value-added compounds is of great interest due to its abundance in natural and biomass gases. However, since the conventional engineering process of methane conversion inevitably produces CO<sub>2</sub>, one of the greenhouse gases, an efficient methane conversion method with low CO<sub>2</sub> emission is mandatory for green manufacturing systems. However, a data-driven and automated optimization of the reaction conditions for high-performance nonoxidative direct conversion of methane remains a challenging problem because we should conduct expensive chemical experiments to collect prior data or knowledge for the optimization. To avoid the expensive costs of chemical experiments, we propose a method to perform active learning without pre-defined unlabeled constructed by expensive chemical experiments. To this end, we combine the active learning method with the metaheuristic algorithms to perform active learning with statistically augmented data. We applied the proposed method to a high-throughput screening task to discover new reaction conditions of high-performance nonoxidative methane, and the high-throughput screening error was significantly reduced by 69.11%.

Reference: Not available

# Run
This repository provides an implementation of active learning with metaheuristic search (ALMS) to construct a prediction model that predicts the C<sub>2</sub> yields of the input reaction conditions. You can construct a predictiono model based on ALMS by executing ``exec_alms.py``.

# Datasets
The reference of the nonoxidative methane converision dataset is https://doi.org/10.1039/D0RE00378F.
We uploaded the initial training, unexplored, and test datasets of the epxierments in the ``datasets`` folder.

- ``dataset.xlsx``: The original dataset.
- **ESTM dataset:** It is a refined thermoelectric materials dataset for machine learning. ESTM dataset contains 5,205 experimental observations of thermoelectric materials and their properties (reference: https://doi.org/10.xxxx/xxxxxxxxx).

# Notes
- This repository contains only a subset of the source Starry dataset due to the dataset license. Please visit [Starrydata](https://www.starrydata2.org) to download the full data of the source Starry dataset.
- The full data of the ESTM dataset is provided in the ``dataset folder`` of this repository.
- The ``results folder`` provides the extrapolation results on the full data of the Starry and ESTM dataset. You can check the extrapolation results reported in the paper.
