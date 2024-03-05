# Large Language Models for Scientific Question Answering: an Extensive Analysis of the SciQA Benchmark

This repository contains all the code and data referenced in the paper titled "Large Language Models for Scientific Question Answering: an Extensive Analysis of the SciQA Benchmark" authored by 

Jens Lehmann<sup>1,2</sup>, Antonello Meloni<sup>3</sup>, Enrico Motta<sup>4</sup>, Francesco Osborne<sup>4,5</sup>, Diego Reforgiato Recupero<sup>3</sup>, Angelo Antonio Salatino<sup>4</sup>, and Sahar Vahdati<sup>1</sup>.

1. ScaDS.AI - TU Dresden, DE
2. Amazon, DE 
3. Department of Mathematics and Computer Science, University of Cagliari, IT 
4. Knowledge Media Institute, The Open University, UK 
5. University of Milan-Bicocca, IT

## Abstract
The SciQA benchmark for scientific question answering aims to represent a challenging task for next-generation question-answering systems on which vanilla large language models fail. In this article, we provide an analysis of the performance of language models on this benchmark including prompting and fine-tuning techniques to adapt them to the SciQA task. We show that both fine-tuning as well as prompting techniques with intelligent few-shot selection allow us to obtain excellent results on the SciQA benchmark. We discuss the valuable lessons and common error categories, and outline their implications on how to optimise large language models for question answering over knowledge graphs.

## More info

The paper delves into the evaluation of large language models on the SciQA benchmark, which focuses on scientific question answering over knowledge graphs. It discusses the dataset composition, including manually crafted and automatically generated questions, and explores the transformer architecture's role in semantic parsing and question answering tasks. A comparison between human-generated and automatically generated questions is presented, emphasizing the distribution across various question templates. 

## Structure of the repository

Folder ```code``` contains the codebase we used to perform our experiments.

Folder ```test_data``` contains the prompts for the large language models.

## How to cite

soon...
