## PT Anchors
> The main purpose of this repo is to show basic usage of **Pre-Trained Language Generation Model** in three NLG tasks and measure its performances. 
Each task is Neural Machine Translation, Dialogue Generation, Abstractive Text Summarization. The model architecture loaded from Hugging Face Library and WMT14, Daily-Dialogue, Daily-CNN datasets have used for each task.

<br><br>

## Model desc
> T5 Model is famous and powerful Pretrained Language Model. Unlike BERT, T5 is composed of Encoder-Decoder Architecture. This makes it more suitable for Natural Language Generation tasks. T5 model enhanced its performance by utilizing sequence based pretrain objectives.

<br>

|                               | &emsp; **MarianMT**        | &emsp; **BlenderBot**                   | &emsp; **T5** |
| ---:                          | :---                       | :---                                    | :---   |
| **Architecture** &nbsp;       | MarianMTModel              | BlenderbotSmallForConditionalGeneration | T5ForConditionalGeneration |
| **Model Name** &nbsp;         | Helsinki-NLP/opus-mt-en-de | facebook/blenderbot_small-90M           | t5-small |
| **Vocab Size** &nbsp;         | 58,101                     | 54,944                                  | 32,128 |
| **Num Params** &nbsp;         |  |  |  |
| **Model Size** &nbsp;         |  |  |  |
| **Pretrain Dataset** &nbsp;   | Opus |  |  |
| &nbsp; **Pretrain Objective** &nbsp; |  |  |  |

<br><br>

## Results

</br></br>


## How to Use
**First clone git repo in your local env**
```
git clone https://github.com/moon23k/PT_Anchors.git
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

<br><br>

## Reference
* [**Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**](https://arxiv.org/abs/1910.10683)
* [**Recipes for building an open-domain chatbot**]()
* [**MarianMT**]()
