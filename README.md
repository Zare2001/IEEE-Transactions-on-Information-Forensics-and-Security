# Privacy-Robustness Trade-off in Decentralized Federated Learning

This repository contains the official source code for the research paper:

**"A Sample Article Using IEEEtran.cls for IEEE Journals and Transactions"**
*by Zarè Palanciyan, Qiongxiu Li, and Richard Heusdens*
*(Replace with your actual paper title and authors once finalized)*

This project rigorously investigates the fundamental trade-off between privacy preservation and adversarial robustness in fully decentralized federated learning (FL). Specifically, we analyze the impact of various privacy mechanisms on the detectability of Byzantine attacks within a Primal-Dual Method of Multipliers (PDMM) based FedAVG framework.

## Overview

The core objective of this research is to demonstrate that strengthening privacy inevitably degrades the ability to detect malicious (Byzantine) behavior in decentralized FL. We achieve this by:
*   Implementing a decentralized FedAVG scheme utilizing PDMM for model aggregation.
*   Integrating three distinct privacy-preserving mechanisms: Differential Privacy (DP), Secure Multi-Party Computation (SMPC), and Subspace Perturbation (SP).
*   Simulating a range of active Byzantine attacks to corrupt model updates.
*   Employing a Median Absolute Deviation (MAD) based detection algorithm to identify adversarial nodes.
*   Quantifying the privacy-robustness trade-off using False Alarm Rate (FAR) and Missed Detection Rate (MDR) as key metrics.

## Repository Contents

This repository is organized as follows:

*   `APPENDIX_A_IEEE_Transactions_on_Informational_Privacy_and_Byzantine_Robustness.pdf`: The full research paper (or supplementary material PDF).
*   `AverageFed.py`: Core implementation of the PDMM-based FedAVG aggregation logic.
*   `Corruption.py`: Definitions and implementations of various Byzantine attack models.
*   `MDR_and_FAR_Plotter.ipynb`: Jupyter notebook for generating the False Alarm Rate (FAR) and Missed Detection Rate (MDR) plots, as seen in the paper's results section.
*   `ModelPerformancePlotter.ipynb`: Jupyter notebook for analyzing and plotting model performance metrics, such as training and testing loss/accuracy.
*   `README.md`: This file.
*   `UtilityGraph.py`: Utilities for generating and managing the decentralized network graph.
