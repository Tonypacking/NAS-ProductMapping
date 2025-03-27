# ProMap Datasets and Language Models
This part of the repository contains part of the ProMapCzExt Test and ProMapEnExt Test datasets along with labels identifying matching and non=maching pairs predicted by several language models: Bard, Bing, ChatGPT, and LLaMA.
Additionally, it contains two scripts that were used during testing.

It contains the following folders:
* **bard**, which contains the data with labels after experiments using [Bard](https://bard.google.com/chat)
* **bing**, which contains the data with labels after experiments using [Bing AI](https://www.bing.com/)
* **chatgpt**,  which contains the data with labels after experiments using [ChatGPT](https://openai.com/blog/chatgpt)
* **llama**,  which contains the data with labels after experiments using [LLaMA](https://huggingface.co/meta-llama/Llama-2-13b)


## Chatgpt and Bard folders
We asked the ChatGPT 3.5 model using its API whether the pairs of products are describing the same product or not. We used only product names in the first round and then we used product names followed with short descriptions. We performed 3 rounds for each using both Czech and English datasets. We did the same with Google Bard but we used its unofficial API. The code is attached in *bard_run.py*.
In the end, we also averaged three models' predictions to eliminate the differences between individual runs.

## Bing and Llama folders
We asked the same questions LLaMA 2 7b model using the attached code in *run_llama.py* and the Bing model using its API. AS the results were very poor we used only product names for the experiments. We also performed 3 rounds for each using both Czech and English datasets and we averaged three models' predictions to eliminate the differences between individual runs.

