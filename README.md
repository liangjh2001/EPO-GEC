## Environment Setup

To set up the environment, ensure you have the following versions installed:

- CUDA: 12.1
- Python: 3.10

To install the required Python dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Download data：

### English Data

- BEA-2019 Shared Task: [Download Here](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data)
- CoNLL-2014 Shared Task: [Download Here](https://www.comp.nus.edu.sg/~nlp/conll14st.html)

### Chinese Data

- FCGEC Dataset: [Download Here](https://github.com/xlxwalex/FCGEC)
- NaCGEC Dataset: [Download Here](https://github.com/masr2000/NaCGEC)

## Data Preprocessing

You need to preprocess the datasets into a unified format, compatible with the training pipeline.

### Required Format

Ensure that the data is structured in the same format as the provided examples:

- `data/epo_data_sample.json`
- `data/sft_data_example.json`

Additionally, you will need to modify the `data/dataset_info.json` file to match the specifics of your dataset configuration.

## Training Pipeline

### SFT Stage 1: 

```bash
bash bash/train_gec_sft_stage1.sh
```

```bash
bash bash/export_model.sh  # merge lora weight
```

### SFT Stage 2: 

```bash
bash bash/train_gec_sft_stage2.sh
```

### Sampling

```bash
bash bash/gec_pairwise_sampling.sh  # generate pairwise samples
```

### EPO Training

```bash
bash bash/train_gec_epo.sh
```

**Note:** For Chinese GEC, you can find the corresponding scripts in the `bash` directory.

## Evaluation

```bash
bash bash/gec_eval.sh  # for English GEC model
```

```bash
bash bash/cgec_eval.sh  # for Chinese GEC model
```

## Acknowledgements

This project is built upon [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and utilizes the following tools for evaluation:

- [ERRANT](https://github.com/chrisjbryant/errant)
- [ChERRANT](https://github.com/HillZhang1999/MuCGEC/tree/main/scorers/ChERRANT)
- [M2Scorer](https://github.com/nusnlp/m2scorer)

We are grateful for their contributions.