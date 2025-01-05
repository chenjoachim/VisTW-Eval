# evaluation

![TARO on yellow sticker](static/cover.jpg)

Evaluation code for benchmarking Taiwan vision benchmarks

Setup environment package

```
pip install -r requirements.txt
```


Make sure you have OpenAI key ready in your environment

```bash
export OAI_KEY="sk-xxxx"
```

Then try running a simple gpt-4o-mini to test if everything is working


```bash
python -m simplevals.eval gpt-4o-mini-2024-07-18 --datasets accounting --mode image
```

Parameters meaning:

```
datasets: support all 25 subjects
mode : default image, use "text" for not using the image
```


This should download the dataset and start evaluating on the first subject

For local models such as meta llama vision use the series 

```
python -m simplevals.eval meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo --series mllama --datasets accounting --mode image
```


