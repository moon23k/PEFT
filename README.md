## PLM Anchors
&nbsp; Pre-trained models have gained widespread adoption in various domains of artificial intelligence, showcasing impressive performance. In the realm of natural language processing, leveraging these models proves highly advantageous, particularly for challenging tasks like natural language generation. However, it is essential to note that the mere adoption of pre-trained models does not guarantee consistent excellence. The downstream performance can vary significantly based on the intricacies of the pre-training methodology employed. Moreover, selecting an appropriate model size that aligns with the available computing resources holds paramount importance.
To address these considerations, this repository compares the performance of three models across three distinct natural language generation tasks while taking into account model size, pre-training data, and the specific pre-training approaches. The evaluated models encompass **Marian**, **BlenderBot**, and **T5**, dedicated to **Neural Machine Translation**, **Dialogue Generation**, and **Summarization** tasks, respectively. All models are accessed and utilized through the HuggingFace library.
Through this comprehensive analysis and comparison, ultimate aim is to provide valuable guidelines for effectively harnessing pre-trained models in natural language generation. By understanding the impact of model size, pre-training data, and methodology, practitioners can make informed decisions when selecting and deploying these models, ensuring optimal performance and resource utilization.

<br><br>

## Model desc
**Marian** <br>
> Marian is a machine translation model developed by MicroSoft. It is specialized in the task of machine translation and has been trained on large-scale parallel corpora. Based on the Transformer architecture, Marian understands the context of language and generates translation results. It can be used for translation tasks across various languages.


**BlenderBot** <br>
> BlenderBot is a conversational chatbot model developed by Facebook AI Research. It has undergone extensive multitask training to improve the quality and consistency of conversations. BlenderBot utilizes predefined knowledge and data gathered from online sources to engage in conversations on various topics. Combining deep learning and generative models, BlenderBot simulates natural and seamless dialogues with humans.

**T5** <br>
> T5 (Text-to-Text Transfer Transformer) is a versatile natural language processing model developed by the Google AI research team. It is a pretrained language model trained on massive and diverse datasets. T5 can be applied to various NLP tasks such as machine translation, summarization, question answering, and more. Designed to handle mapping problems between text inputs and text outputs, T5 provides a solution for a wide range of text-based tasks.

<br><br>

## Model Configs

|                               |  **MarianMT**              | **BlenderBot**                          | **T5** |
| ---:                          | ---:                       | ---:                                    | ---:   |
| **Architecture** &nbsp;       | MarianMTModel              | BlenderbotSmallForConditionalGeneration | T5ForConditionalGeneration |
| **Model Name** &nbsp;         | Helsinki-NLP/opus-mt-en-de | facebook/blenderbot_small-90M           | t5-small |BERT
| **Vocab Size** &nbsp;         | 58,101                     | 54,944                                  | 32,128 |
| **Num Params** &nbsp;         | 73,886,208 | 87,508,992 | 60,506,624 |
| **Model Size** &nbsp;         | 284.075 MB | 334.030 MB | 230.814 MB |
| **Pretrain Dataset** &nbsp;   | Opus |  |  |
| &nbsp; **Pretrain Objective** &nbsp; |  |  |  |

<br><br>

## Results

| &emsp; **Model** &emsp; | &emsp; **Metric** &emsp; | &emsp; **Evaluation Score** &emsp; |
| ---:                    | :---:                    | :---:                          |
| Pretrained **MarianMT** | BLEU | - |
| Fine Tuned **MarianMT** | BLEU | - |
| Pretrained **BlenderBotSmall**| ROUGE | - |
| Fine Tuned **BlenderBotSmall** | ROUGE | - |
| Pretrained **T5** | ROUGE |17.72|
| Fine Tuned **T5** | ROUGE | - |

</br></br>


## How to Use
**Clone repository on your own env**
```
git clone https://github.com/moon23k/PT_Anchors.git
```

<br>

**Download and Process Dataset through setup.py**
```
bash setup.py -task [all, nmt, dialog, sum]
```

<br>

**Execute the run file on your purpose (search arg is optional)**
```
python3 run.py -task [nmt, dialog, sum] -mode [train, test, inference] -search [greedy, beam]
```

<br><br>

## Reference
* [**Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**](https://arxiv.org/abs/1910.10683)
* [**Recipes for building an open-domain chatbot**](https://arxiv.org/abs/2004.13637)
* [**Marian: Fast Neural Machine Translation in C++**](https://arxiv.org/abs/1804.00344)
