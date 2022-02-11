
## Table of Content
> * [Multi Classification - Pytorch](#MultiClassification-Pytorch)
>   * [About the Project](#AbouttheProject)
>   * [About Database](#AboutDatabases)
>   * [Built with](#Builtwith)
>   * [Installation](#Installation)
>   * [Examples](#Example)

# Multi Classification - Pytorch
## About the Project
This project creates an algorithm detect 10 categories of objects in the STL-10 dataset. Teh model is one of the state-of-the-art and pre-trained on the ImageNet dataset. The model is fine-tuned on the STL-10 dataset.

![recipe](https://user-images.githubusercontent.com/75105778/153649787-46a34ba4-83b7-4a1f-9e9f-87babf9a3d95.jpg)


## About Database

Dataset is STL-10 from the PyTorch torchvision package.

For more information about STL-10  https:/ / cs. stanford.edu/ ~acoates/ stl10 .


## Built with
* Pytorch
* Model is ResNet18 (Both of randomly initialized weights or the pre-trained weights)
* Combination of  LogSoftmax  and NLLLoss
* Adam optimizer.

## Installation
    •	conda install pytorch torchvision cudatoolkit=coda version -c pytorch

## Examples

![mc](https://user-images.githubusercontent.com/75105778/153684919-cc1d4b9c-dd09-4e83-b86f-d83319e55ff9.png)



