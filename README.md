# FIETA: Few-shot Inductive Knowledge Graph Completion via Task-Aware Graph Neural Networks
This is the official implementation of the paper **"FIETA: Few-shot Inductive Knowledge Graph Completion via Task-Aware Graph Neural Networks"** accepted to the 35th ACM International Conference on Information and Knowledge Management (CIKM 2026).

All codes are written by **Chanyoung Chung** ([chanyoung.chung@kaist.ac.kr](mailto:chanyoung.chung@kaist.ac.kr)).

If you use this repository, please cite our paper.
```
@inproceedings{fieta,
  author={Chanyoung Chung and Minsung Hwang and Joyce Jiyoung Whang},
  title={FIETA: Few-shot Inductive Knowledge Graph Completion via Task-Aware Graph Neural Networks},
  year={2026},
	booktitle={Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
	doi={10.1145/3799682.3840780},
}
```

## Prerequisites
We used Python 3.10.13, PyTorch 2.0.1, and CUDA 11.7 to run all experiments. Install the required packages with the following command:
```
pip install -r requirements.txt
```
Also, unzip `data.zip` to use the datasets. You can download the datasets from [here](https://drive.google.com/file/d/1hQba5br53x0JiBWe355g1PGkls7VKzd7/view?usp=sharing).
## Preprocessing
Use the following codes to compute positional encodings (LapPE):
```
python compute_LapPE.py [DATA_NAME]
```
## Training & Inference
To train and test FIETA, please run the following command:
```
python main.py [DATA_NAME] -K [K] -d [DIM] -nlr [NLR] -nls [NLS] -nle [NLE] -pos [POS] -p [DROP] -nh [NH] -lr [LR]
```
The following table describes the arguments of `main.py`:
| Name | Description | Default |
| :-: | :- | :-: |
| `DATA_NAME` | Name of the dataset | - |
| `K` | Number of support triplets | 1 |
| `DIM` | Number of dimensions of representations | 32 |
| `NLR` | Number of layers in $$\phi_\text{rel}$$ | 3 |
| `NLS` | Number of layers in $$\phi_\text{sup}$$ | 6 |
| `NLE` | Number of layers in $$\phi_\text{ent}$$ | 6 |
| `POS` | Positional encodings (none or lappe) | none |
| `P` | Dropout rate | 0.0 |
| `LR` | Learning rate | 0.001 |

We tuned only the dropout rate `P` and use the default values for all other hyperparameters. The dropout rates used for each dataset are as follows:

|  | NL-Ind-T | NL-Ind-S | NL-Ind-I | FB-Ind-T | FB-Ind-S | FB-Ind-I | NELL-One |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1-Shot | 0.3 | 0.3 | 0.3 | 0.2 | 0.0 | 0.0 | 0.0 |
| 5-Shot | 0.2 | 0.3 | 0.0 | 0.0 | 0.1 | 0.2 | 0.0 |

## Data Processing
We provide the codes for generating few-shot inductive KGC datasets. Unzip `dataset_generation.zip` to see the data generation steps.

## License

This code is released under the **CC BY-NC-SA 4.0 license**.
