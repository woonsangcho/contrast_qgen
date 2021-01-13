## Contrastive Multi-Document Question Generation

This repo contains a complete reproducible code and dataset for the arXiv paper: https://arxiv.org/abs/1911.03047, accepted to appear at EACL 2021. It is based on repositories from [Huggingface Transformers](https://github.com/huggingface/transformers) and [OpenAI GPT-2](https://github.com/openai/gpt-2).

In the paper, we propose a novel generating coordinator model that leverage reinforcement learning using signals from multiple documents. We also develop a principled contrastive-learning based regularization to promote specificity of generated questions. 

### Recommended Environment
- Linux Ubuntu 18.04
- GPU with at least 12G memory

At both training time and evaluation time, it requires loading the GPT-2 generator block model, Transformer-based coordinator model and the pre-trained ranker model from which we derive reinforcement learning signals. 

Model training has been done using 8 NVidia Tesla V100 GPU in parallel. We recommend running our codebase with multiple GPUs.

### Dependencies
- [PyTorch](https://pytorch.org/) (1.4.0)
- [PyLucene](https://lucene.apache.org/pylucene/) (7.7.1)
- [nltk](https://www.nltk.org/install.html) (3.4.5)
- [nlg-eval](https://github.com/Maluuba/nlg-eval) (2.2)
- [transformers](https://github.com/huggingface/transformers) (however, the repo is self-contained)
- numpy
- scipy
- boto3
- requests
- tqdm
- regex

### Setup

```git clone https://github.com/woonsangcho/contrast_qgen```

For pre-processing and constructing challenging negative samples, first download the raw [MS-MARCO Conversational Search](https://github.com/microsoft/MSMARCO-Conversational-Search), and follow the preprocessing code. For convenience, download the dataset from [here](https://drive.google.com/open?id=1zjea_-B3zaHRn-RS382ccIEK7HnS17jj) and place it under your ```$DATA_PATH/```. We randomly splits the publicly available MS-MARCO-QA dataset into train/dev/test sets. Due to the large collection of dataset, we random sampled a subset of the dev set to expedite the training. 

Download the pre-trained ranker, converted into PyTorch for convenience [here](https://drive.google.com/open?id=1bfi2z_QekANaO-2uqdqIfloLJPEmx_c9).

Download *pre-indexed* Lucene files [here](https://drive.google.com/open?id=14MDH_5AJvlAbMuAmhixqxApQDDEEHadE) for ranking, and [here](https://drive.google.com/open?id=1DICXy9YkECo6jrBbn1NIsixbXIAW1Qie) for the retrieval-based baselines.

### 1. Fine-tuning of the pre-trained GPT-2 generator block model on the MS-MARCO domain

Download a pre-trained GPT-2 model (small) from this [link](https://github.com/openai/gpt-2). 

Download the public MS-MARCO dataset, formatted to our codebase [here](https://drive.google.com/open?id=1m6Fe31ntceowXhte62p0iXokHCWYOmyf).

``` python src/train_gpt2_distributed.py --config $CONFIG_PATH ```

```$CONFIG_PATH``` contains the path to the model configuration file: ```config_file/config_domain_tune.json```.
Modify your data file paths under ```$DATA_PATH```.

If you would like to train the generator block from scratch rather than using the pre-trained GPT-2 model, append ``` --init_checkpoint 'None' ```. However, we observed we can obtain a better generating block by fine-tuning a pre-trained GPT-2 model (based on validation dataset). To bypass this step for your convenience, you can download our fine-tuned GPT-2 model [here](www.google.com).

### 2. Training the coordinator using RL and Set-induced Contrastive Regularization (SCR)

First, build and install PyLucene following the commands [here](https://lucene.apache.org/pylucene/install.html).

We distributed the training across multiple GPUs (8 GPUs) using the following command.

``` python -m torch.distributed.launch --nproc_per_node=8 src/train_gpt2_distributed_rl.py --config $CONFIG_PATH ```

```$CONFIG_PATH``` contains the path to the model configuration file: ```config_file/config_coordinator_rl.json```. This contains the default configuration for training the full model. Modify the arguments to fit your environment. For other parameter options, see comments.

On 8 NVidia Tesla V100 GPUs, the training takes about 2 days to complete. For a pre-trained coordinator, download [here](https://drive.google.com/open?id=1jEwUMt-BNsmPLRVb_lu71VJUBWAUYHzl).

The coordinator with default configuration has 14,501,377 parameters.

### 3. Evaluating generated questions via automatic metrics

```python src/evaluate_coordinator.py --config config_file/config_domain_tune.json --coordinator_model <path-to-the-coordinator-model> ```

```python src/evaluate_coordinator_embedding.py --config config_file/config_domain_tune.json --coordinator_model <path-to-the-coordinator-model> ```

## Contact

Please email all inquiries to Woon Sang Cho at: woonsang __at__ princeton.edu.

## Disclaimer

This repository aims to promote further research in multi-document question generation. This source code provided here contains the research pipeline, including the modeling code needed to produce a model weight file, as well as generation code. This repository can be adapted to users' own data to generate outputs. We are not responsible for any generation from the 3rd party utilization of the shared files included herein, including the pretrained system or the generation code.

## Citation

For citation, please use the following bibtex entry:

```
@article{cho2020contrastqgen,
  title={Contrastive Multi-Document Question Generation},
  author={Cho, Woon Sang and Zhang, Yizhe and Rao, Sudha and Celikyilmaz, Asli and Xiong, Chenyan and Gao, Jianfeng and Wang, Mengdi and Dolan, Bill},
  year={2020}
  eprint={1911.03047},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
