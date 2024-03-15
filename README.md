# **The First to Know: How Token Distributions Reveal Hidden Knowledge in Large Vision-Language Models?**

This repository contains the source code for the paper.

**The First to Know: How Token Distributions Reveal Hidden Knowledge in Large Vision-Language Models?** [[Paper](https://arxiv.org/abs/2403.09037)]

*Qinyu Zhao, Ming Xu, Kartik Gupta, Akshay Asthana, Liang Zheng, Stephen Gould*



## Abstract
Large vision-language models (LVLMs), designed to interpret and respond to human instructions, occasionally generate hallucinated or harmful content due to inappropriate instructions. This study uses linear probing to shed light on the hidden knowledge at the output layer of LVLMs. We demonstrate that the logit distributions of the first tokens contain sufficient information to determine whether to respond to the instructions, including recognizing unanswerable visual questions, defending against multi-modal jailbreaking attack, and identifying deceptive questions. Such hidden knowledge is gradually lost in logits of subsequent tokens during response generation. Then, we illustrate a simple decoding strategy at the generation of the first token, effectively improving the generated content. In experiments, we find a few interesting insights: First, the CLIP model already contains a strong signal for solving these tasks, indicating potential bias in the existing datasets. Second, we observe performance improvement by utilizing the first logit distributions on three additional tasks, including indicting uncertainty in math solving, mitigating hallucination, and image classification. Last, with the same training data, simply finetuning LVLMs improve models' performance but is still inferior to linear probing on these tasks.



## TODO

[] Upload data generated in our study

[] Upload the checkpoints of finetuned LLaVA

[] Upload evaluation codes for all six tasks

[] Write a step-by-step instruction in the README file

