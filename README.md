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
Prepare two pandas dataframe: one for demonstrating set and one for test set. You can find examples under dataset/ folder. Note that the index column should contain the filenames of the images. Here's a quick preview: 

| Index | Forest | Golf course | Freeway |
|:-------------|:--------------:|:--------------:|:--------------:|
|forest39.jpeg| 1 | 0 | 0 |
|golfcourse53.jpeg| 0 | 1 | 0 |
|freeway97.jpeg| 0 | 0 | 1 |

## Expected directory structure
```
ManyICL/
├── LMM.py
├── dataset
│   ├── UCMerced_21
│   │   ├── forest39.jpeg
│   │   ├── forest47.jpeg
│   │   ├── freeway09.jpeg
│   │   ├── freeway97.jpeg
│   │   ├── golfcourse53.jpeg
│   │   ├── golfcourse76.jpeg
│   │   ├── ...
│   ├── UCMerced_demo_21.pkl
│   └── UCMerced_test_21.pkl
├── exp_ucmerced.py
└── prompt.py
```

# Configure the prompt

Modify the prompt in prompt.py if needed.

# Run the experiment
Run the experiment script, and it'll save all the raw responses in UCMerced_21shot_Gemini1.5_1.pkl.
```bash
python3 exp_ucmerced.py --num_shot_per_class=1
```

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
