## Parameter Efficient Fine Tuning
&nbsp; Large-scale pretrained language models have shown remarkable performance across various natural language processing tasks. 
However, fine-tuning these large models and using them for inference requires substantial computational resources. 
To address this challenge, the concept of Parameter Efficient Fine-Tuning (PEFT) has been developed. 
PEFT involves tuning only a subset of a model's parameters, rather than the entire set.

PEFT offers several key advantages:

* **Improved Memory Efficiency**: <br> PEFT reduces the model's size, leading to decreased memory consumption. This alleviates the memory burden associated with handling large-scale models.

* **Enhanced Training Speed**: <br> Using smaller models results in faster training times. This allows for quicker model development and tuning.

* **Effectiveness on Small Datasets**: <br> PEFT is effective even when working with small datasets, making it a valuable approach when data is limited.

In this project, we apply PEFT using four representative techniques to three different Natural Language Generation (NLG) tasks, and conduct experiments to validate its effectiveness. 
Through these experiments, we aim to demonstrate the value of PEFT, enabling efficient utilization of resources and time while achieving outstanding natural language processing performance.


<br><br>

## PEFT Methodology
**LoRA** <br>
> Low-Rank Adaptation (LoRA) is a reparametrization method that aims to reduce the number of trainable parameters with low-rank representations. 
The weight matrix is broken down into low-rank matrices that are trained and updated. 
All the pretrained model parameters remain frozen. After training, the low-rank matrices are added back to the original weights. 
This makes it more efficient to store and train a LoRA model because there are significantly fewer parameters.

<br>

**Prefix Tuning** <br>
> Prefix tuning is an additive method where only a sequence of continuous task-specific vectors is attached to the beginning of the input, or prefix.
Only the prefix parameters are optimized and added to the hidden states in every layer of the model.
The tokens of the input sequence can still attend to the prefix as virtual tokens.
As a result, prefix tuning stores 1000x fewer parameters than a fully finetuned model, which means you can use one large language model for many tasks.

<br>

**P-Tuning** <br>
> It is challenging to finetune large language models for downstream tasks because they have so many parameters. 
To work around this, you can use prompts to steer the model toward a particular downstream task without fully finetuning a model. 
Typically, these prompts are handcrafted, which may be impractical because you need very large validation sets to find the best prompts. 
P-tuning is a method for automatically searching and optimizing for better prompts in a continuous space.

<br>

**Prompt Tuning** <br>
> Prompting helps guide language model behavior by adding some input text specific to a task. 
Prompt tuning is an additive method for only training and updating the newly added prompt tokens to a pretrained model. 
This way, you can use one pretrained model whose weights are frozen, and train and update a smaller set of prompt parameters for each downstream task instead of fully finetuning a separate model. 
As models grow larger and larger, prompt tuning can be more efficient, and results are even better as model parameters scale.

<br>

**IA3** <br>
> IA3 refers to "Infused Adapter by Inhibiting and Amplifying Inner Activations".

<br><br>

## Results

| &emsp; **PEFT** &emsp; | &emsp; **Translation** &emsp; | &emsp; **Dialogue** &emsp; | &emsp; **Summarization** &emsp; |
| :---:                  | :---:                         | :---:                      | :---:                           |
| None          | - | - | - |
| LoRA          | - | - | - |
| Prefix Tuning | - | - | - |
| P-Tuning      | - | - | - |
| Prompt Tuning | - | - | - |
| IA3           | - | - | - |

</br></br>


## How to Use
**Clone repository on your own env**
```
git clone https://github.com/moon23k/PEFT.git
```

<br>

**Download and Process Dataset through setup.py**
```
bash setup.py -task [all, translation, dialogue, summarization]
```

<br>

**Execute the run file on your purpose (search arg is optional)**
```
python3 run.py -task [translation, dialogue, summarization]
               -mode [train, test, inference]
               -peft [None, lora, p_tuning, prompt_tuning, ia3]
               -search [greedy, beam]
```

<br><br>

## Reference
* [**LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS**](https://arxiv.org/pdf/2106.09685.pdf)
* [**P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks**](https://arxiv.org/pdf/2110.07602.pdf)
* [**The Power of Scale for Parameter-Efficient Prompt Tuning**](https://arxiv.org/pdf/2104.08691.pdf)
* [**GPT Understands, Too**](https://arxiv.org/pdf/2103.10385.pdf)

<br>
