# PatchTST

This is an offical implementation of PatchTST: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers."

# Getting Started

We seperate our codes for supervised learning and self-supervised learning into 2 folders. Please choose the one that you want to work with.

## Supervised Learning

1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download all the datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Create a seperate folder ```./dataset``` and put all the csv files in the directory.

3. Training. All the scripts are in the directory ```./scripts/EXP-LongForecasting/PatchTST```. The default model is PatchTST/42. For example, if you want to get the multivariate forecasting results for weather dataset, just run the following command, and you can open ```./result.txt``` to see the results once the training is done:
```
sh ./scripts/EXP-LongForecasting/PatchTST/weather
```

## Self-supervised Learning

1. Follow the first 2 steps above

2. Pre-training: The scirpt patchtst_pretrain.py is to train the PatchTST/64. To run the code with a single GPU on ettm1, just run the following command
```
python patchtst_pretrain.py --dset ettm1 --mask_ratio 0.4
```
The model will be saved to the saved_model folder for the downstream tasks. There are several other parameters can be set in the patchtst_pretrain.py script.
 
 3. Fine-tuning: The script patchtst_finetune.py is for fine-tuning step. Either linear_probing or fine-tune the entire network can be applied.
```
python patchtst_finetune.py --dset ettm1 --pretrained_model <model_name>
```
