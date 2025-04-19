# Benchmarking Vision-Language Models for Traditional Chinese in Taiwan

<p align="center"> <img src="static/cover.jpg" style="width: 80%; max-width: 800px" id="title-icon">       </p>

VisTW consists of two subsets: (1) MCQ - a collection of multiple-choice questions from 21 academic subjects (answer choices omitted for space); and (2) Dialogue - real-life images with corresponding questions requiring understanding of Traditional Chinese and Taiwan-specific cultural context.

# VisTW Leaderboard (2025/04/19)


| Model | VisTW-MCQ |  | VisTW-Dialogue |  | Avg |
|-------|------------------------|----------|---------------------------------|----------|----------|
|  | **Accuracy** | **Rank** | **Score 0-10** | **Rank** | **Rank** |
| ~~gemini-2.0-pro-exp-02-05~~ | 0.6619 | 2 | 6.7237 | 2 | 2.0 |
| gemini-2.0-flash-001 | 0.6596 | 3 | 6.6451 | 4 | 3.5 |
| gemini-2.5-pro-preview-03-25 | 0.6072 | 7 | 7.9725 | 1 | 4.0 |
| quasar-alpha (early version of gpt-4.1) | 0.6673 | 1 | 6.2733 | 8 | 4.5 |
| optimus-alpha (early version of gpt-4.1) | 0.6434 | 6 | 6.6916 | 3 | 4.5 |
| gpt-4.1-2025-04-16 | 0.6504 | 5 | 6.5954 | 5 | 5.0 |
| gpt-4.1-mini-2025-04-16 | 0.5809 | 9 | 6.1344 | 9 | 9.0 |
| llama-4-maverick | 0.6529 | 4 | 4.884 | 15 | 9.5 |
| claude-3-5-sonnet-20241022 | 0.6019 | 8 | 5.9603 | 12 | 10.0 |
| gpt-4o-2024-11-20 | 0.5755 | 10 | 6.1176 | 10 | 10.0 |
| gemini-2.0-flash-lite-preview-02-05 | 0.4992 | 13 | 6.4159 | 7 | 10.0 |
| qwen2.5-vl-72b-instruct | 0.5504 | 11 | 4.8656 | 16 | 13.5 |
| qwen2.5-vl-32b-instruct | 0.4935 | 14 | 5.5027 | 13 | 13.5 |
| gemini-2.0-flash-thinking-exp-1219 | 0.3764 | 24 | 6.5053 | 6 | 15.0 |
| gemini-1.5-pro | 0.4417 | 17 | 5.0504 | 14 | 15.5 |
| gpt-4o-2024-08-06 | 0.4 | 21 | 5.9756 | 11 | 16.0 |
| mistral-small-3.1-24b-instruct-2503 | 0.459 | 16 | 4.3298 | 19 | 17.5 |
| llama-4-scout | 0.5292 | 12 | 4.0943 | 24 | 18.0 |
| gpt-4o-mini-2024-07-18 | 0.4091 | 20 | 4.7405 | 17 | 18.5 |
| gemma-3-12b-it | 0.4863 | 15 | 3.9403 | 25 | 20.0 |
| gemini-1.5-flash | 0.3943 | 23 | 4.2611 | 20 | 21.5 |
| Qwen-Qwen2.5-VL-7B-Instruct | 0.3592 | 25 | 4.542 | 18 | 21.5 |
| gpt-4.1-nano-2025-04-16 | 0.3974 | 22 | 4.1634 | 23 | 22.5 |
| qvq-72b-preview | 0.4094 | 19 | 3.6122 | 29 | 24.0 |
| meta-llama-Llama-3.2-90B-Vision-Instruct-Turbo | 0.4119 | 18 | 3.4443 | 32 | 25.0 |
| OpenGVLab-InternVL2_5-8B | 0.3447 | 27 | 3.9008 | 26 | 26.5 |
| OpenGVLab-InternVL2-8B-MPO | 0.3533 | 26 | 3.6778 | 28 | 27.0 |
| gemini-1.5-flash-8b | 0.328 | 32 | 4.1771 | 22 | 27.0 |
| claude-3-haiku-20240307 | 0.3291 | 30 | 3.6992 | 27 | 28.5 |
| Qwen-Qwen2-VL-7B-Instruct | 0.3004 | 37 | 4.2122 | 21 | 29.0 |
| OpenGVLab-InternVL2-8B | 0.3431 | 28 | 3.4504 | 31 | 29.5 |
| OpenGVLab-InternVL2_5-4B | 0.3291 | 31 | 3.6031 | 30 | 30.5 |
| nova-lite-v1 | 0.3377 | 29 | 3.2626 | 33 | 31.0 |
| llama3.2-ffm-11b-v-32k-chat | 0.3119 | 35 | 3.115 | 35 | 35.0 |
| meta-llama-Llama-3.2-11B-Vision-Instruct-Turbo | 0.3262 | 33 | 2.5786 | 38 | 35.5 |
| MediaTek-Research-Llama-Breeze2-8B-Instruct | 0.2915 | 39 | 3.1374 | 34 | 36.5 |
| OpenGVLab-InternVL2-4B | 0.3081 | 36 | 2.3069 | 39 | 37.5 |
| MediaTek-Research-Llama-Breeze2-3B-Instruct | 0.2971 | 38 | 2.8992 | 37 | 37.5 |
| deepseek-ai-deepseek-vl2-small | 0.3181 | 34 | 0.5084 | 44 | 39.0 |
| THUDM-cogvlm2-llama3-chinese-chat-19B | 0.2777 | 43 | 2.9618 | 36 | 39.5 |
| OpenGVLab-InternVL2-2B | 0.2891 | 40 | 2.2198 | 40 | 40.0 |
| phi-4-multimodal-instruct | 0.286 | 41 | 1.7863 | 43 | 42.0 |
| deepseek-ai-deepseek-vl2-tiny | 0.2781 | 42 | 2.0076 | 42 | 42.0 |
| OpenGVLab-InternVL2-1B | 0.2689 | 44 | 2.1298 | 41 | 42.5 |

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
