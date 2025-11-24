HyperCASR: Spectral-spatial Open-Set Recognition With Category-Aware Semantic Reconstruction for Hyperspectral Imagery, TIP, 2025.
==
[Bobo Xi](https://b-xi.github.io/), [Wenjie Zhang](https://github.com/WenjieZhang-cyber), [Jiaojiao Li](https://scholar.google.com/citations?user=Ccu3-acAAAAJ&hl=zh-CN&oi=sra), [Rui Song](https://scholar.google.com/citations?user=_SKooBYAAAAJ&hl=zh-CN), and [Yunsong Li](https://dblp.uni-trier.de/pid/87/5840.html).
***

Code for the paper: [HyperCASR: Spectral-spatial Open-Set Recognition With Category-Aware Semantic Reconstruction for Hyperspectral Imagery](http://doi.org/10.1109/TIP.2025.3630327).

<div align=center><img src="/backbone.png" width="90%" height="90%"></div>
Fig. 1: The architecture of the proposed HyperCASR for HSI OSR. 

## Abstract
Open-set recognition (OSR) in hyperspectral imagery (HSI) focuses on accurately classifying known classes while effectively rejecting unknown negative samples. Most existing reconstruction-based approaches are susceptible to noise interference in the input images, and known classes can easily lead to inter-class confusion during the reconstruction process. Moreover, effectively utilizing the abundant spectral-spatial information in HSI within an open-set context presents significant challenges. To address these issues, we propose HyperCASR, an innovative framework for HSI OSR that integrates a grouped spectral-spatial retentive transformer (GSSRT) and a class-aware semantic reconstruction (CASR) module. This method begins by designing the GSSRT to extract features from HSI, enhancing the extraction capability of spatial-spectral information by introducing a grouped pixel embedding (GPE) module and a novel spatial retentive attention (SRA) mechanism. Subsequently, an independent autoencoder (AE) is assigned to each known class to reconstruct semantic features, which helps to mitigate noise interference and inter-class confusion. Additionally, by minimizing reconstruction errors to estimate class affiliation, the framework effectively identifies unknown classes. Experimental results across three benchmark datasets indicate that the HyperCASR framework significantly enhances classification performance for both known and unknown classes when compared to existing state-of-the-art methods. The code is available at https://github.com/B-Xi/TIP_2025_HyperCASR.

## Training and Test Process
1. Please prepare the training and test data as operated in the paper.
2. Run the 'main_UP.py' to reproduce the HyperCASR results on UP data set.

## References
--
If you find this code helpful, please kindly cite:

[1] Xi, B., Zhang, W., Li, J., Song, R., & Li, Y., "HyperCASR: Spectral-spatial Open-Set Recognition With Category-Aware Semantic Reconstruction for Hyperspectral Imagery," in IEEE Transactions on Image Processing, doi: 10.1109/TIP.2025.3630327.

Citation Details
--
BibTeX entry:
```
@ARTICLE{11247869,
  author={Xi, Bobo and Zhang, Wenjie and Li, Jiaojiao and Song, Rui and Li, Yunsong},
  journal={IEEE Transactions on Image Processing}, 
  title={HyperCASR: Spectral-spatial Open-Set Recognition With Category-Aware Semantic Reconstruction for Hyperspectral Imagery}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIP.2025.3630327}}
```
 
Licensing
--
Copyright (C) 2025 Bobo Xi, Wenjie Zhang

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.