# To do

## 1. Prompt engineering: answer only about NVIDIA's 2026 Annual Report

The system prompt (`SYSTEM_PROMPT` in `multimodal_rag/generate.py`) tells the model to reply directly, without searching, to greetings and to questions that are not about the report. So it also answers unrelated requests (general knowledge, poems, code), working as a free general chatbot. Change the prompt so the agent declines anything unrelated to the report, while still handling greetings, "Who are you?" and follow-up questions such as "And the year before?".

## 2. Security layer with a System 1 decision model

A small open-source model that runs locally, in the spirit of TypeSafe AI's Jev (a "System One" model that returns probabilities over options declared in advance, in one pass, instead of generating text). It sees the new message, with the previous question so follow-ups are not judged alone, and decides between options such as:

- `report_question`: goes to the agent, as now;
- `greeting_or_about_you`: a fixed introduction, without calling DeepSeek;
- `off_topic` and `manipulation_attempt`: a fixed refusal, without calling DeepSeek.

Requirements and decisions so far:
- It must not add noticeable latency. Answers take 3 to 7 seconds; a classifier should add tens of milliseconds at most. A small generative model would add hundreds of milliseconds or more, so it is not the choice.
- Act on a fixed reply only above a confidence threshold (for example 0.9), and otherwise send the message to the agent: wrongly refusing a report question is worse than letting an off-topic one through.
- Log each decision and its probability to LangSmith.

Candidates to compare in a notebook (`notebooks/07_...`) on about 80 labeled messages (report questions, follow-ups, greetings, off-topic requests, manipulation attempts), measuring accuracy, wrongly refused report questions and latency on this machine (laptop Ryzen CPU with 12 threads, GTX 1650 with 4 GB, 7.3 GB RAM):
- GLiClass (`knowledgator/gliclass-edge-v3.0` or `-base-v3.0`, Apache-2.0): zero-shot, scores all declared options in one pass.
- `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` (MIT): zero-shot, one pass per option.
- A small open LLM read as a single-pass classifier (for example Qwen3-0.6B): the probabilities of the option words from one forward pass, no generation.
- For manipulation attempts, a dedicated classifier: `protectai/deberta-v3-base-prompt-injection-v2` (Apache-2.0, ONNX available) or `meta-llama/Llama-Prompt-Guard-2-22M` (gated, Llama license).

Keep their packages (transformers, torch, gliclass) out of `requirements.txt` until one is chosen.

## 3. Eval suite

25 to 30 questions about the report, each with its expected page and answer, kept as a LangSmith dataset: the 6 test questions in `generate.py`, more questions on tables, charts and text, follow-up questions, and questions the report cannot answer. Run the agent over the dataset as a LangSmith experiment and score:
- retrieval: whether the expected page is among the retrieved chunks;
- the answer: whether it is correct and grounded in the retrieved chunks.

Compare experiments after each change to the prompt, the retrieval settings or the models. The labeled messages from item 2 can join the dataset.
