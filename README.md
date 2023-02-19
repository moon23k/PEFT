## T5 Anchors
> The main purpose of this repo is to show basic usage of **Pretrained Language Model** in three NLG tasks and measure its performances. 
Each task is Neural Machine Translation, Dialogue Generation, Abstractive Text Summarization. The model architecture loaded from Hugging Face Library and WMT14, Daily-Dialogue, Daily-CNN datasets have used for each task.

<br>
<br>

## Model desc
> T5 Model is famous and powerful Pretrained Language Model. Unlike BERT, T5 is composed of Encoder-Decoder Architecture. This makes it more suitable for Natural Language Generation tasks. T5 model enhanced its performance by utilizing sequence based pretrain objectives.

<br>
<br>

## Configurations
| &emsp; **Vocab Config**                            | &emsp; **Model Config**                 | &emsp; **Training Config**               |
| :---                                               | :---                                    | :---                                     |
| **`Vocab Size:`** &hairsp; `30,000`                | **`Input Dimension:`** `30,000`         | **`Epochs:`** `10`                       |
| **`Vocab Type:`** &hairsp; `BPE`                   | **`Output Dimension:`** `30,000`        | **`Batch Size:`** `32`                   |
| **`PAD Idx, Token:`** &hairsp; `0`, `[PAD]` &emsp; | **`Embedding Dimension:`** `256` &emsp; | **`Learning Rate:`** `5e-4`              |
| **`UNK Idx, Token:`** &hairsp; `1`, `[UNK]`        | **`Hidden Dimension:`** `512`           | **`iters_to_accumulate:`** `4`           |
| **`BOS Idx, Token:`** &hairsp; `2`, `[BOS]`        | **`N Layers:`** `2`                     | **`Gradient Clip Max Norm:`** `1` &emsp; |
| **`EOS Idx, Token:`** &hairsp; `3`, `[EOS]`        | **`Drop-out Ratio:`** `0.5`             | **`Apply AMP:`** `True`                  |

<br>
<br>


## Results
> **Training Results**

<center>
  <img src="https://user-images.githubusercontent.com/71929682/201269096-2cc00b2f-4e8d-4071-945c-f5a3bfbca985.png" width="90%" height="70%">
</center>


</br>

> **Test Results**

</br>
</br>


## How to Use
**First clone git repo in your local env**
```
git clone https://github.com/moon23k/T5_Anchors
```

<br>

**Download and Process Dataset via setup.py**
```
bash setup.py -task [all, nmt, dialog, sum]
```

<br>

**Execute the run file on your purpose (search is optional)**
```
python3 run.py -task [nmt, dialog, sum] -mode [train, test, inference] -search [greedy, beam]
```


<br>
<br>

## Reference
* [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
