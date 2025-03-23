# Benchmarking Vision-Language Models for Traditional Chinese in Taiwan

<p align="center"> <img src="static/cover.jpg" style="width: 80%; max-width: 800px" id="title-icon">       </p>

VisTW consists of two subsets: (1) MCQ - a collection of multiple-choice questions from 21 academic subjects (answer choices omitted for space); and (2) Dialogue - real-life images with corresponding questions requiring understanding of Traditional Chinese and Taiwan-specific cultural context.

| Model | VisTW-Dialogue |  | VisTW-MCQ |  | Avg |
|  | **Score 0-10** | **Rank** | **Accuracy** | **Rank** | **Rank** |
| Gemini-2.0-pro-exp-02-05 | 6.72 | 1 | 0.6619 | 1 | 1.0 |
| Gemini-2.0-flash-001 | 6.15 | 3 | 0.6596 | 2 | 2.5 |
| gpt-4o-2024-11-20 | 6.12 | 4 | 0.5755 | 4 | 4.0 |
| Claude-3-5-sonnet-20241022 | 5.96 | 6 | 0.6019 | 3 | 4.5 |
| Gemini-2.0-flash-lite-preview-02-05 | 5.92 | 7 | 0.4992 | 6 | 6.5 |
| Qwen2.5-VL-72B-instruct | 4.87 | 9 | 0.5413 | 5 | 7.0 |
| Gemini-1.5-pro | 5.05 | 8 | 0.4417 | 9 | 8.5 |
| Gemini-2.0-flash-thinking-exp-1219 | 6.51 | 2 | 0.3764 | 15 | 8.5 |
| gpt-4o-2024-08-06 | 5.98 | 5 | 0.4000 | 13 | 9.0 |
| Mistral-Small-3.1-24B | 4.33 | 12 | 0.4590 | 8 | 10.0 |
| Qwen2-VL-72B-instruct | 4.21 | 14 | 0.4701 | 7 | 10.5 |
| gpt-4o-mini-2024-07-18 | 4.74 | 10 | 0.4091 | 12 | 11.0 |
| Gemma3-27b-it | 3.94 | 17 | 0.4375 | 10 | 13.5 |
| Qwen2.5-VL-7B-Instruct | 4.54 | 11 | 0.3592 | 16 | 13.5 |
| Gemini-1.5-flash | 4.26 | 13 | 0.3943 | 14 | 13.5 |
| Llama-3.2-90B-Vision-Instruct | 3.44 | 22 | 0.4119 | 11 | 16.5 |
| InternVL2.5-8B | 3.90 | 18 | 0.3447 | 17 | 17.5 |
| Gemini-1.5-flash-8B | 4.18 | 16 | 0.3280 | 22 | 19.0 |
| InternVL2-8B | 3.45 | 21 | 0.3431 | 18 | 19.5 |
| Claude-3-haiku-20240307 | 3.70 | 19 | 0.3291 | 20 | 19.5 |
| Qwen2-VL-7B-Instruct | 4.21 | 15 | 0.3004 | 26 | 20.5 |
| InternVL2.5-4B | 3.60 | 20 | 0.3291 | 21 | 20.5 |
| Nova-lite-v1 | 3.26 | 23 | 0.3376 | 19 | 21.0 |
| Llama-3.2-11B-Vision-Instruct | 2.58 | 27 | 0.3262 | 23 | 25.0 |
| Breeze2-8B-Instruct | 3.14 | 24 | 0.2915 | 28 | 26.0 |
| Breeze2-3B-Instruct | 2.90 | 26 | 0.2971 | 27 | 26.5 |
| InternVL2-4B | 2.31 | 28 | 0.3081 | 25 | 26.5 |
| Deepseek-vl2-small | 0.51 | 30 | 0.3181 | 24 | 27.0 |
| CogVLM2-llama3-chinese-chat | 2.96 | 25 | 0.2777 | 30 | 27.5 |
| InternVL2-2B | 2.22 | 29 | 0.2891 | 29 | 29.0 |


## Setup

1. Git clone the entire repo down first.

2/ Setup environment package

```
pip install -r requirements.txt
```


3. Make sure you have OpenAI key ready in your environment

```bash
export OAI_KEY="sk-xxxx"
```

4. Then try running a simple gpt-4o-mini to test if everything is working


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


5. For evaluating on dialogue dataset follow the following instructions

Its a 2 stage process, 1st stage requires inference on the dialogue prompts and image

```
python -m simplevals.freeform gpt-4o-mini --series openai --mode image
```

Once all samples are finished, start the scoring results

```
python -m simplevals.scoring --input_jsonl freeform_log/gpt-4o-mini.jsonl
```

You can find the final score in `/score_board`

```
scores = []
with open("freeform_log/score_board/gpt-4o-mini.jsonl", 'r') as f:
    for line in f:
        payload = json.loads(line)
        scores.append(payload['score'])
print(sum(scores)/len(scores))
```
