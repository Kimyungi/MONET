# MONET: Modality-Embracing Graph Convolutional Network and Target-Aware Attention for Multimedia Recommendation
This repository provides a reference implementation of *MONET* as described in the following paper:
> MONET: Modality-Embracing Graph Convolutional Network and Target-Aware Attention for Multimedia Recommendation<br>
> Yungi Kim, Taeri Kim, Won-Yong Shin and Sang-Wook Kim<br>
> 17th ACM Int'l Conf. on Web Search and Data Mining (ACM WSDM 2024)<br>

### Overview of MONET
![monet](https://github.com/Kimyungi/MONET/assets/28508383/a386478c-d3d6-4c13-abef-fca83e95ae71)

### Authors
- Yungi Kim (gozj3319@hanyang.ac.kr)
- Taeri Kim (taerik@hanyang.ac.kr)
- Won-Yong Shin (wy.shin@yonsei.ac.kr)
- Sang-Wook Kim (wook@hanyang.ac.kr)

### Requirements
The code has been tested running under Python 3.6.13. The required packages are as follows:
- ```gensim==3.8.3```
- ```pytorch==1.10.2+cu113```
- ```torch_geometric==2.0.3```
- ```sentence_transformers==2.2.0```
- ```pandas```
- ```numpy```
- ```tqdm```
- ```torch-scatter```
- ```torch-sparse```
- ```torch-cluster```
- ```torch-spline-conv```
- ```torch-geometric```

### Dataset Preparation
#### Dataset Download
*Men Clothing and Women Clothing*: Download Amazon product dataset provided by [MAML](https://github.com/liufancs/MAML). Put data folder into the directory data/.

*Beauty and Toys & Games*: Download 5-core reviews data, meta data, and image features from [Amazon product dataset](http://jmcauley.ucsd.edu/data/amazon/links.html). Put data into the directory data/{folder}/meta-data/.

#### Dataset Preprocessing
Run ```python build_data.py --name={Dataset}```

### Usage
#### For simplicity, we provide usage for the WomenClothing dataset.
------------------------------------
- For MONET in RQ1,
```
python main.py --agg=concat --n_layers=2 --alpha=1.0 --beta=0.3 --dataset=WomenClothing --model_name=MONET_2_10_3
```
------------------------------------
- For RQ2, refer the second cell in "Preliminaries.ipynb".
------------------------------------
- For MONET_w/o_MeGCN and MONET_w/o_TA in RQ3,
```
python main.py --agg=concat --n_layers=0 --alpha=1.0 --beta=0.3 --dataset=WomenClothing --model_name=MONET_wo_MeGCN
python main.py --target_aware --agg=concat --n_layers=2 --alpha=1.0 --beta=0.3 --dataset=WomenClothing --model_name=MONET_wo_TA
```
------------------------------------
- For MONET_sum, MONET_weighted_sum, and MONET_fc_layer in RQ4,
```
python main.py --agg=sum --n_layers=2 --alpha=1.0 --beta=0.3 --dataset=WomenClothing --model_name=MONET_sum
python main.py --agg=weighted_sum --n_layers=2 --alpha=1.0 --beta=0.3 --dataset=WomenClothing --model_name=MONET_weighted_sum
python main.py --agg=fc --n_layers=2 --alpha=1.0 --beta=0.3 --dataset=WomenClothing --model_name=MONET_fc_layer
```
------------------------------------
- For RQ5 (hyperparameters $\alpha$, $\beta$ sensitivity),
```
python main.py --agg=concat --n_layers=2 --alpha={value} --beta=0.3 --dataset=WomenClothing --model_name=MONET_2_{alpha}_3
python main.py --agg=concat --n_layers=2 --alpha=1.0 --beta={value} --dataset=WomenClothing --model_name=MONET_2_10_{beta}
```

### Cite
We encourage you to cite our paper if you have used the code in your work. You can use the following BibTex citation:
```
@inproceedings{kim24wsdm,
  author   = {Yungi Kim and Taeri Kim and Won{-}Yong Shin and Sang{-}Wook Kim},
  title     = {MONET: Modality-Embracing Graph Convolutional Network and Target-Aware Attention for Multimedia Recommendation},
  booktitle = {ACM International Conference on Web Search and Data Mining (ACM WSDM 2024)},      
  year      = {2024}
}
```

### Acknowledgement
The structure of this code is largely based on [LATTICE](https://github.com/CRIPAC-DIG/LATTICE). Thank for their work.
