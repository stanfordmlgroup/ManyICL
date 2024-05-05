# ManyICL: Scaling In-Context Learning for Multimodal Foundation Models


This repository contains implementation of [ManyICL](https://arxiv.org/abs/XXX). Prepare a dataframe, configure your API key, modify the prompt and just run it!

Please note that this code repo is intended for research purpose, and might not be suitable for large-scale production.


# Installation
Install packages using pip:
```bash
$ pip install -r requirements.txt
```

# Setup API keys
## For GPT-series models offered by OpenAI
1. Get your API key from [here](https://platform.openai.com/api-keys);
2. Replace the placeholder in LMM.py (Line 29);

## For Gemini-series models offered by Vertex AI
Note that you need a Google cloud project for this. 
1. In the Google Cloud console, go to the [Dashboard](https://console.cloud.google.com/home).
2. Click the project selection list at the top of the page. In the Select a resource window that appears, select a project. Note the project ID displayed in the Project info section.
3. Replace the placeholder in LMM.py (Line 113);
4. If you're developing locally or on Colab (not on GCP instances), you need to authenticate by following this [instruction](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/sdk-for-gemini/gemini-sdk-overview-reference#authenticate-vertex-python-sdk).

# Dataset preparation

# Configure the prompt

# Run the experiment

# Citation

If you find our work useful in your research please consider citing:

```
@inproceedings{
Luo2023closerlook,
title={A Closer Look at Few-shot Classification Again},
author={Luo, Xu and Wu, Hao and Zhang, Ji and Gao, Lianli and Xu, Jing and Song, Jingkuan},
booktitle={International Conference on Machine Learning},
year={2023},
}
```

## Acknowlegements
