# **The First to Know: How Token Distributions Reveal Hidden Knowledge in Large Vision-Language Models?**

This repository contains the source code for the paper.

**The First to Know: How Token Distributions Reveal Hidden Knowledge in Large Vision-Language Models?** [[Paper](https://arxiv.org/abs/2403.09037)]

*Qinyu Zhao, Ming Xu, Kartik Gupta, Akshay Asthana, Liang Zheng, Stephen Gould*



## Abstract
Large vision-language models (LVLMs), designed to interpret and respond to human instructions, occasionally generate hallucinated or harmful content due to inappropriate instructions. This study uses linear probing to shed light on the hidden knowledge at the output layer of LVLMs. **We demonstrate that the logit distributions of the first tokens contain sufficient information to determine whether to respond to the instructions, including recognizing unanswerable visual questions, defending against multi-modal jailbreaking attack, and identifying deceptive questions.** Such hidden knowledge is gradually lost in logits of subsequent tokens during response generation. Then, we illustrate a simple decoding strategy at the generation of the first token, effectively improving the generated content. In experiments, we find a few interesting insights: First, the CLIP model already contains a strong signal for solving these tasks, indicating potential bias in the existing datasets. Second, we observe performance improvement by utilizing the first logit distributions on three additional tasks, including indicting uncertainty in math solving, mitigating hallucination, and image classification. Last, with the same training data, simply finetuning LVLMs improve models' performance but is still inferior to linear probing on these tasks.

## Getting Started

### Installation
####  Clone this repository to your local machine.

```
git clone https://github.com/Qinyu-Allen-Zhao/LVLM-LP.git
```

### Usage
We evaluate the linear probing method on six different tasks. For a specific task, you need to (i) prepare the dataset, (ii) prepare the model, (iii) run the model on the dataset to get logits of the first tokens, and last, (iv) evaluate the performance of linear probing. 

Don't worry. We are trying our best to make the process very simple for you.


#### 1. Download the dataset
After downloading and unzipping data, please modify dataset paths [here](https://github.com/Qinyu-Allen-Zhao/LVLM-LP/blob/main/dataset/__init__.py).

- Task 1. Identify unanswerable visual questions: Download images and annotations of the train and validation sets of [VizWiz](https://vizwiz.org/tasks-and-datasets/vqa/). 

- Task 2. Defense against jailbreak attack: We use [MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench/tree/main) and generated safe image-query pairs via the same data generation process of theirs. Our generated data is provided on [GoogleDrive](https://drive.google.com/file/d/16jULXndiNwFE8L6NzTz63StM9Njts3qS/view?usp=sharing).

- Task 3. Identify deceptive questions: When we conducted the experiment, the dataset [MAD-Bench](https://arxiv.org/abs/2402.13220) had not been released yet. We generated the data via the process described in the paper and augmented the dataset by adding normal questions. Download the val2017 set of [COCO](https://cocodataset.org/#home). The [annotations](https://github.com/Qinyu-Allen-Zhao/LVLM-LP/tree/main/data/MADBench) have been included in this repo.

- Task 4. Uncertainty in math solving: We use the testmini set of [MathVista](https://mathvista.github.io/). It will be automatically downloaded via HuggingFace. You don't need to do anything at this stage.

- Task 5. Mitigate hallucination: Download the train2014 and val2014 sets of [COCO](https://cocodataset.org/#home). The [annotations](https://github.com/Qinyu-Allen-Zhao/LVLM-LP/tree/main/data/pope) have been included in this repo.

- Task 6. Image classification: Download the train and val sets of [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index).


#### 2. Prepare the large models
Please prepare the models that you want to use, as illustrated in their original repos. We didn't modify the main structure of their models. 

- Clone their repos first. [LLaVA](https://github.com/haotian-liu/LLaVA), [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [InstructBLIP](https://github.com/salesforce/LAVIS), [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl/tree/main), [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter), [MMGPT](https://github.com/open-mmlab/Multimodal-GPT). *LLaVA might be an easy start.
- Download model checkpoints as required; 
- Set the model folder in the model files under [model](https://github.com/Qinyu-Allen-Zhao/LVLM-LP/tree/main/model).


#### 3. Run on different tasks

OK, life goes very easy at this stage. We provide you all scripts for running models on each task to get logits of the first tokens. Please find them [here](https://github.com/Qinyu-Allen-Zhao/LVLM-LP/tree/main/scripts).

For example, run this command (modify it based on your gpu numbers)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/VizWiz/run_LLaVA_7B.sh
```

#### 4. Evaluate linear probing

Step 3 is to generate and store logits of the first tokens. In this step, we will run linear probing (a simple classifer actually) and evaluate its performance.

Please refer to those eval.ipynb files.

### Use our finetuned model checkpoints

I'm learning how to upload models to HuggingFace. Will update soon!!!

## TODO

- [x] Upload data generated in our study

- [x] Write a step-by-step instruction in the README file

- [] Upload the checkpoints of finetuned LLaVA

- [x] Upload evaluation codes for all six tasks



