# CasRel_Pytorch_EN

Pytorch reimplement of the paper "A Novel Cascade Binary Tagging Framework for Relational Triple Extraction" ACL2020. The [original code](https://github.com/weizhepei/CasRel) was written in keras.

# Introduction

I followed the previous work of [longlongman](https://github.com/longlongman/CasRel-pytorch-reimplement).

I made some changes in order to better apply to the English DataSet. The changed I have made are listed:

- I changed the tokenizer from HBTokenizer to AutoTokenizer；

- Establish the position mapping of the source text and word segmentation in order to fit the English datasets.

# Requirements

- transformers==4.11.0

- torch==1.10.0

# DataSet

The data are in form of json. Take one as an example:

> {
>     "text": "#####################",
>     "triple_list": [
>          [
> 
>             [
> 
>             "aaa","start_position","end_position"
> 
>             ],
> 
>             predicate,
> 
>             [
> 
>             "bbb","start_position","end_position"
> 
>             ]
> 
>           ]
>        ]
> }

# Usage

1. Get the pre-trained English BERT model

2. Train the model
   
   > python train.py

3. Test the model
   
   > python test.py
