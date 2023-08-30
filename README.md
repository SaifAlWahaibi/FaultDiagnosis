# Fault Diagnosis

NOTE: I am trying to figure out how to upload the dataset used in the paper. It's a 65 MB zip file

This page contains all the codes related to the following paper "Improving convolutional neural networks for fault diagnosis in chemical processes by incorporating global correlations," published by myself, Samuel Abiola, Myisha A. Chowdhury and my Ph.D. advisor Qiugang Lu.

## Abstract:

Fault diagnosis (FD) has received attention because of its importance in maintaining safe operations of industrial processes. Recently, modern data-driven FD approaches such as deep learning have shown encouraging performance. Particularly, convolutional neural networks (CNNs) offer an alluring capacity to deal with multivariate time-series data converted into images. Nonetheless, existing CNN techniques focus on capturing local correlations. However, global spatiotemporal correlations often prevail in multivariate time-series data from industrial processes. Hence, extracting global correlations using CNNs from such data requires deep architectures that incur many trainable parameters. This paper proposes a novel local–global scale CNN (LGS-CNN) that directly accounts for local and global correlations. Specifically, the proposed network incorporates local correlations through traditional square kernels and global correlations are collected utilizing spatially separated one-dimensional kernels in a unique arrangement. FD performance on the benchmark Tennessee Eastman process dataset validates the proposed LGS-CNN against CNNs, and other state-of-the-art data-driven FD approaches.

## LGS-CNN:

![image](https://github.com/SaifAlWahaibi/FaultDiagnosis/assets/106843163/3335d0eb-3c39-4bb7-b9b2-a0522272ba29)

## Global Correlation Extraction:

![Dummy 3x3 GIF](https://github.com/SaifAlWahaibi/FaultDiagnosis/assets/106843163/5227a882-7132-48ee-9c2e-d89419fa78f0.gif)

## Results:

### Comparison with CNNs:

![image](https://github.com/SaifAlWahaibi/FaultDiagnosis/assets/106843163/6cf142d0-9c6f-4e63-9d72-3427fe3fc5bc)

![image](https://github.com/SaifAlWahaibi/FaultDiagnosis/assets/106843163/3bd95b9c-5d24-4186-8acd-a5649f4b71aa)


*LGS-CNN and CNN Models 1-5 are trained and validated on the same dataset
**LGS-CNN and CNN Models 6 and 7 are trained and tested on the same dataset
***LGS-CNN and CNN Models 1-5 have a similar number of trainable parameters with the CNN madels having slightly more
****LGS-CNN and CNN Models 6 and 7 have a similar number of trainable parameters with the CNN madels having slightly more
*****Details on Models 1-7 for LGS-CNN and CNN can be found in the paper

### LGS-CNN Local Receptive Fields:

![image](https://github.com/SaifAlWahaibi/FaultDiagnosis/assets/106843163/0bd68cc2-ae01-45d1-a289-6bb77edd20f1)

### t-SNE Plots:

#### After feature extraction with LGS-CNN:

![image](https://github.com/SaifAlWahaibi/FaultDiagnosis/assets/106843163/fd0ee2a7-6cdc-4deb-804f-de0d411e9953)

#### Before feature extraction with LGS-CNN:

![image](https://github.com/SaifAlWahaibi/FaultDiagnosis/assets/106843163/9fc75997-4ec5-42df-a4ea-7855e31052d3)

### Comparison with other Models:

![image](https://github.com/SaifAlWahaibi/FaultDiagnosis/assets/106843163/2fc5b534-b64a-40d5-9ee3-a6418ed353d9)
