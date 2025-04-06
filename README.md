# Benchmarking Vision-Language Models for Traditional Chinese in Taiwan

<p align="center"> <img src="static/cover.jpg" style="width: 80%; max-width: 800px" id="title-icon">       </p>

VisTW consists of two subsets: (1) MCQ - a collection of multiple-choice questions from 21 academic subjects (answer choices omitted for space); and (2) Dialogue - real-life images with corresponding questions requiring understanding of Traditional Chinese and Taiwan-specific cultural context.

# VisTW Leaderboard (2025/04/06)


| Model | VisTW-MCQ |  | VisTW-Dialogue |  | Avg |
|-------|------------------------|----------|---------------------------------|----------|----------|
|  | **Accuracy** | **Rank** | **Score 0-10** | **Rank** | **Rank** |
| ~~gemini-2.0-pro-exp-02-05~~ | 0.6619 | 2 | 6.7237 | 1 | 1.5 |
| gemini-2.0-flash-001 | 0.6596 | 3 | 6.6451 | 2 | 2.5 |
| quasar-alpha | 0.6673 | 1 | 6.2733 | 5 | 3.0 |
| gpt-4o-2024-11-20 | 0.5755 | 6 | 6.1176 | 6 | 6.0 |
| claude-3-5-sonnet-20241022 | 0.6019 | 5 | 5.9603 | 8 | 6.5 |
| gemini-2.0-flash-lite-preview-02-05 | 0.4992 | 9 | 6.4159 | 4 | 6.5 |
| llama-4-maverick | 0.6529 | 4 | 4.884 | 11 | 7.5 |
| qwen2.5-vl-72b-instruct | 0.5504 | 7 | 4.8656 | 12 | 9.5 |
| qwen2.5-vl-32b-instruct | 0.4935 | 10 | 5.5027 | 9 | 9.5 |
| gemini-2.0-flash-thinking-exp-1219 | 0.3764 | 19 | 6.5053 | 3 | 11.0 |
| gemini-1.5-pro | 0.4417 | 13 | 5.0504 | 10 | 11.5 |
| gpt-4o-2024-08-06 | 0.4 | 17 | 5.9756 | 7 | 12.0 |
| llama-4-scout | 0.5292 | 8 | 4.0943 | 19 | 13.5 |
| mistral-small-3.1-24b-instruct-2503 | 0.459 | 12 | 4.3298 | 15 | 13.5 |
| gpt-4o-mini-2024-07-18 | 0.4091 | 16 | 4.7405 | 13 | 14.5 |
| gemma-3-12b-it | 0.4863 | 11 | 3.9403 | 20 | 15.5 |
| gemini-1.5-flash | 0.3943 | 18 | 4.2611 | 16 | 17.0 |
| Qwen-Qwen2.5-VL-7B-Instruct | 0.3592 | 20 | 4.542 | 14 | 17.0 |
| qvq-72b-preview | 0.4094 | 15 | 3.6122 | 24 | 19.5 |
| meta-llama-Llama-3.2-90B-Vision-Instruct-Turbo | 0.4119 | 14 | 3.4443 | 27 | 20.5 |
| OpenGVLab-InternVL2_5-8B | 0.3447 | 22 | 3.9008 | 21 | 21.5 |
| OpenGVLab-InternVL2-8B-MPO | 0.3533 | 21 | 3.6778 | 23 | 22.0 |
| gemini-1.5-flash-8b | 0.328 | 27 | 4.1771 | 18 | 22.5 |
| claude-3-haiku-20240307 | 0.3291 | 25 | 3.6992 | 22 | 23.5 |
| OpenGVLab-InternVL2-8B | 0.3431 | 23 | 3.4504 | 26 | 24.5 |
| Qwen-Qwen2-VL-7B-Instruct | 0.3004 | 32 | 4.2122 | 17 | 24.5 |
| OpenGVLab-InternVL2_5-4B | 0.3291 | 26 | 3.6031 | 25 | 25.5 |
| nova-lite-v1 | 0.3377 | 24 | 3.2626 | 28 | 26.0 |
| llama3.2-ffm-11b-v-32k-chat | 0.3119 | 30 | 3.115 | 30 | 30.0 |
| meta-llama-Llama-3.2-11B-Vision-Instruct-Turbo | 0.3262 | 28 | 2.5786 | 33 | 30.5 |
| MediaTek-Research-Llama-Breeze2-8B-Instruct | 0.2915 | 34 | 3.1374 | 29 | 31.5 |
| OpenGVLab-InternVL2-4B | 0.3081 | 31 | 2.3069 | 34 | 32.5 |
| MediaTek-Research-Llama-Breeze2-3B-Instruct | 0.2971 | 33 | 2.8992 | 32 | 32.5 |
| deepseek-ai-deepseek-vl2-small | 0.3181 | 29 | 0.5084 | 39 | 34.0 |
| THUDM-cogvlm2-llama3-chinese-chat-19B | 0.2777 | 38 | 2.9618 | 31 | 34.5 |
| OpenGVLab-InternVL2-2B | 0.2891 | 35 | 2.2198 | 35 | 35.0 |
| phi-4-multimodal-instruct | 0.286 | 36 | 1.7863 | 38 | 37.0 |
| deepseek-ai-deepseek-vl2-tiny | 0.2781 | 37 | 2.0076 | 37 | 37.0 |
| OpenGVLab-InternVL2-1B | 0.2689 | 39 | 2.1298 | 36 | 37.5 |

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
