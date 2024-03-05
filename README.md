# Large Language Models for Scientific Question Answering: an Extensive Analysis of the SciQA Benchmark

This repository contains all the code and data referenced in the paper titled "Large Language Models for Scientific Question Answering: an Extensive Analysis of the SciQA Benchmark" authored by 

Jens Lehmann<sup>1,2</sup>, Antonello Meloni<sup>3</sup>, Enrico Motta<sup>4</sup>, Francesco Osborne<sup>4,5</sup>, Diego Reforgiato Recupero<sup>3</sup>, Angelo Antonio Salatino<sup>4</sup>, and Sahar Vahdati<sup>1</sup>.

1. ScaDS.AI - TU Dresden, DE
2. Amazon, DE 
3. Department of Mathematics and Computer Science, University of Cagliari, IT 
4. Knowledge Media Institute, The Open University, UK 
5. University of Milan-Bicocca, IT

## Table of Content

- [Large Language Models for Scientific Question Answering: an Extensive Analysis of the SciQA Benchmark](#large-language-models-for-scientific-question-answering-an-extensive-analysis-of-the-sciqa-benchmark)
  - [Table of Content](#table-of-content)
  - [Abstract](#abstract)
  - [More info](#more-info)
    - [Models employed](#models-employed)
      - [T5-base](#t5-base)
      - [GPT-2-large](#gpt-2-large)
      - [Dolly-v2-3b](#dolly-v2-3b)
      - [GPT-3.5 Turbo](#gpt-35-turbo)
  - [Structure of the repository](#structure-of-the-repository)
  - [How to cite](#how-to-cite)


## Abstract
The SciQA benchmark for scientific question answering aims to represent a challenging task for next-generation question-answering systems on which vanilla large language models fail. In this article, we provide an analysis of the performance of language models on this benchmark including prompting and fine-tuning techniques to adapt them to the SciQA task. We show that both fine-tuning as well as prompting techniques with intelligent few-shot selection allow us to obtain excellent results on the SciQA benchmark. We discuss the valuable lessons and common error categories, and outline their implications on how to optimise large language models for question answering over knowledge graphs.

## More info

The paper delves into the evaluation of large language models on the SciQA benchmark, which focuses on scientific question answering over knowledge graphs. It discusses the dataset composition, including manually crafted and automatically generated questions, and explores the transformer architecture's role in semantic parsing and question answering tasks. A comparison between human-generated and automatically generated questions is presented, emphasizing the distribution across various question templates. 

### Models employed

#### T5-base
Description: T5-base is a transformer-based language model known for its versatility in natural language processing tasks. It is a relatively smaller model compared to other variants but has shown effectiveness in various applications.

#### GPT-2-large
Description: GPT-2-large is a variant of the Generative Pre-trained Transformer (GPT) model developed by OpenAI. It is known for its large number of parameters and ability to generate coherent text based on input prompts.

#### Dolly-v2-3b
Description: Dolly-v2-3b is a large language model designed for natural language processing tasks. It is known for its capacity to handle complex language understanding and generation tasks.

#### GPT-3.5 Turbo
Description: GPT-3.5 Turbo is an advanced version of the GPT-3 model, incorporating enhancements for improved performance in various language processing tasks. It is known for its large-scale capabilities and high-quality text generation.

These models were selected for their capabilities in handling complex language understanding tasks and were fine-tuned and evaluated on the SciQA benchmark to assess their performance in scientific question answering over knowledge graphs.



## Structure of the repository

Folder ```code``` contains the codebase we used to perform our experiments.

Folder ```test_data``` contains the prompts for the large language models.

Each folder has its own ```README.md``` file.

## How to cite

soon...
