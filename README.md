# Benchmarking Vision-Language Models for Traditional Chinese in Taiwan

<p align="center"> <img src="static/cover.jpg" style="width: 80%; max-width: 800px" id="title-icon">       </p>

VisTW consists of two subsets: (1) MCQ - a collection of multiple-choice questions from 21 academic subjects (answer choices omitted for space); and (2) Dialogue - real-life images with corresponding questions requiring understanding of Traditional Chinese and Taiwan-specific cultural context.

# VisTW Leaderboard (2025/04/20)


| Model | VisTW-MCQ |  | VisTW-Dialogue |  | Avg |
|-------|------------------------|----------|---------------------------------|----------|----------|
|  | **Accuracy** | **Rank** | **Score 0-10** | **Rank** | **Rank** |
| o3-2025-04-16 | 0.7769 | 1 | 6.9878 | 2 | 1.5 |
| o4-mini-2025-04-16 | 0.7364 | 2 | 6.7802 | 3 | 2.5 |
| ~~gemini-2.0-pro-exp-02-05~~ | 0.6619 | 4 | 6.7237 | 4 | 4.0 |
| gemini-2.5-pro-preview-03-25 | 0.6072 | 9 | 7.9725 | 1 | 5.0 |
| gemini-2.0-flash-001 | 0.6596 | 5 | 6.6451 | 6 | 5.5 |
| quasar-alpha ( early version of gpt-4.1 ) | 0.6673 | 3 | 6.2733 | 10 | 6.5 |
| optimus-alpha ( early version of gpt-4.1 ) | 0.6434 | 8 | 6.6916 | 5 | 6.5 |
| gpt-4.1 | 0.6503 | 7 | 6.5954 | 7 | 7.0 |
| gpt-4.1-mini | 0.5809 | 11 | 6.1344 | 11 | 11.0 |
| llama-4-maverick | 0.6529 | 6 | 4.884 | 17 | 11.5 |
| claude-3-5-sonnet-20241022 | 0.6019 | 10 | 5.9603 | 14 | 12.0 |
| gpt-4o-2024-11-20 | 0.5755 | 12 | 6.1176 | 12 | 12.0 |
| gemini-2.0-flash-lite-preview-02-05 | 0.4992 | 15 | 6.4159 | 9 | 12.0 |
| qwen2.5-vl-72b-instruct | 0.5504 | 13 | 4.8656 | 18 | 15.5 |
| qwen2.5-vl-32b-instruct | 0.4935 | 16 | 5.5027 | 15 | 15.5 |
| gemini-2.0-flash-thinking-exp-1219 | 0.3764 | 26 | 6.5053 | 8 | 17.0 |
| gemini-1.5-pro | 0.4417 | 19 | 5.0504 | 16 | 17.5 |
| gpt-4o-2024-08-06 | 0.4 | 23 | 5.9756 | 13 | 18.0 |
| mistral-small-3.1-24b-instruct-2503 | 0.459 | 18 | 4.3298 | 21 | 19.5 |
| llama-4-scout | 0.5292 | 14 | 4.0943 | 26 | 20.0 |
| gpt-4o-mini-2024-07-18 | 0.4091 | 22 | 4.7405 | 19 | 20.5 |
| gemma-3-12b-it | 0.4863 | 17 | 3.9403 | 27 | 22.0 |
| gemini-1.5-flash | 0.3943 | 25 | 4.2611 | 22 | 23.5 |
| Qwen-Qwen2.5-VL-7B-Instruct | 0.3592 | 27 | 4.542 | 20 | 23.5 |
| gpt-4.1-nano | 0.3974 | 24 | 4.1634 | 25 | 24.5 |
| qvq-72b-preview | 0.4094 | 21 | 3.6122 | 31 | 26.0 |
| meta-llama-Llama-3.2-90B-Vision-Instruct-Turbo | 0.4119 | 20 | 3.4443 | 34 | 27.0 |
| OpenGVLab-InternVL2_5-8B | 0.3447 | 29 | 3.9008 | 28 | 28.5 |
| OpenGVLab-InternVL2-8B-MPO | 0.3533 | 28 | 3.6778 | 30 | 29.0 |
| gemini-1.5-flash-8b | 0.328 | 34 | 4.1771 | 24 | 29.0 |
| claude-3-haiku-20240307 | 0.3291 | 32 | 3.6992 | 29 | 30.5 |
| Qwen-Qwen2-VL-7B-Instruct | 0.3004 | 39 | 4.2122 | 23 | 31.0 |
| OpenGVLab-InternVL2-8B | 0.3431 | 30 | 3.4504 | 33 | 31.5 |
| OpenGVLab-InternVL2_5-4B | 0.3291 | 33 | 3.6031 | 32 | 32.5 |
| nova-lite-v1 | 0.3377 | 31 | 3.2626 | 35 | 33.0 |
| llama3.2-ffm-11b-v-32k-chat | 0.3119 | 37 | 3.115 | 37 | 37.0 |
| meta-llama-Llama-3.2-11B-Vision-Instruct-Turbo | 0.3262 | 35 | 2.5786 | 40 | 37.5 |
| MediaTek-Research-Llama-Breeze2-8B-Instruct | 0.2915 | 41 | 3.1374 | 36 | 38.5 |
| OpenGVLab-InternVL2-4B | 0.3081 | 38 | 2.3069 | 41 | 39.5 |
| MediaTek-Research-Llama-Breeze2-3B-Instruct | 0.2971 | 40 | 2.8992 | 39 | 39.5 |
| deepseek-ai-deepseek-vl2-small | 0.3181 | 36 | 0.5084 | 46 | 41.0 |
| THUDM-cogvlm2-llama3-chinese-chat-19B | 0.2777 | 45 | 2.9618 | 38 | 41.5 |
| OpenGVLab-InternVL2-2B | 0.2891 | 42 | 2.2198 | 42 | 42.0 |
| phi-4-multimodal-instruct | 0.286 | 43 | 1.7863 | 45 | 44.0 |
| deepseek-ai-deepseek-vl2-tiny | 0.2781 | 44 | 2.0076 | 44 | 44.0 |
| OpenGVLab-InternVL2-1B | 0.2689 | 46 | 2.1298 | 43 | 44.5 |

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
