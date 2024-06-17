# multitarget-drug-design

## Overview
This repository houses the code, data, and results for "Several Birds with One Stone: Exploring the Potential of AI Methods for Multi-Target Drug Design." The focus is on leveraging artificial intelligence to develop multi-target drug designs efficiently.

## Repository Contents
Source code for each ML model used in experiments:
- **/FragVAE_DEL**: Code for FragVAE+DEL implementation
  - **/single_target**: Code for experiments targeting a single protein.
  - **/dual_target**: Code for experiments targeting two proteins.
  - **/triple_target**: Code for experiments targeting three proteins.
- **/JTVAE_DEL**: Code for JTVAE+DEL implementation
  - **/single_target**
  - **/dual_target**
  - **/triple_target**
- **/AAE_DEL**: Code for AAE+DEL implementation
  - **/single_target**
  - **/dual_target**
  - **/triple_target**
- **/DEL_Counter_Docking**: Code for DEL with counter docking (CD) implementation
  - **/AAE-CD**
  - **/FragVAE-CD**
  - **/JTVAE-CD**
- **/DEL_Without_Docking**: Code for DEL without implementation
  - **/DEL-SAS-LogP**: DEL with objective set <SAS, LogP>
  - **/DEL-SAS-LogP-QED**: DEL with objective set <SAS, LogP, QED>
  - **/DEL-SAS-LogP-QED-TPSA**: DEL with objective set <SAS, LogP, QED, TPSA>
- **/data/**: Experiment ZINC RAW data.

## Getting Started
1. Clone this repository.
2. Follow the setup instructions in `README.md` within each directory to prepare your environment.

## Citation
Please cite our paper if you use this repository for your research.
