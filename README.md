# Fault Diagnosis

This page contains all the codes related to the following paper "Improving convolutional neural networks for fault diagnosis in chemical processes by incorporating global correlations," published by myself, Samuel Abiola, Myisha A. Chowdhury and my Ph.D. advisor Qiugang Lu.

## Abstract:

Fault diagnosis (FD) has received attention because of its importance in maintaining safe operations of industrial processes. Recently, modern data-driven FD approaches such as deep learning have shown encouraging performance. Particularly, convolutional neural networks (CNNs) offer an alluring capacity to deal with multivariate time-series data converted into images. Nonetheless, existing CNN techniques focus on capturing local correlations. However, global spatiotemporal correlations often prevail in multivariate time-series data from industrial processes. Hence, extracting global correlations using CNNs from such data requires deep architectures that incur many trainable parameters. This paper proposes a novel local–global scale CNN (LGS-CNN) that directly accounts for local and global correlations. Specifically, the proposed network incorporates local correlations through traditional square kernels and global correlations are collected utilizing spatially separated one-dimensional kernels in a unique arrangement. FD performance on the benchmark Tennessee Eastman process dataset validates the proposed LGS-CNN against CNNs, and other state-of-the-art data-driven FD approaches.

## LGS-CNN:

![image](https://github.com/SaifAlWahaibi/FaultDiagnosis/assets/106843163/3335d0eb-3c39-4bb7-b9b2-a0522272ba29)

## Global Correlation Extraction:

![Dummy 3x3 GIF](https://github.com/SaifAlWahaibi/FaultDiagnosis/assets/106843163/5227a882-7132-48ee-9c2e-d89419fa78f0.gif)
