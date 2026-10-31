# Retriever evaluation: the judge sees chart images (October 31, 2026)

The baseline run's judge saw a retrieved chart only as its written description, while the answering model gets the description and the image. This run gives the judge the image too, and changes nothing else.

- **Notebook:** `evals/notebooks/retriever_eval.ipynb`. Its outputs are this run, committed together with this file.
- **Compared with:** `2026-10-31-retriever-baseline.md`
- **Golden set, retriever settings, metrics and judge model:** unchanged (15 questions; top 3 chunks, cutoff 0.3; Contextual Precision and Contextual Recall; `deepseek-flash`)

## What changed

- A retrieved chart now reaches the judge as its description followed by its image: the notebook adds deepeval's `MLLMImage`, read from the chart's `image_path`, to that chunk's entry in the `retrieval_context`.
- deepeval marks a test case with an image as multimodal, and the metrics' prompts then tell the judge that a document "can be text or an image". Test cases without a chart are sent exactly as before.
- deepeval's `DeepSeekModel` sends text only, so the judge is the same model, `deepseek-flash`, called through deepeval's `LocalModel`, which talks to any OpenAI-compatible API and passes images on.

## Results

| Content | Questions | Precision, baseline | Precision, this run | Recall, baseline | Recall, this run |
|---|---|---|---|---|---|
| Chart | 3 | 1.00 | 1.00 | 0.89 | 0.89 |
| Text | 5 | 0.77 | 0.70 | 0.60 | 0.60 |
| Two pages | 2 | 0.79 | 0.79 | 0.25 | 0.25 |
| Table | 5 | 0.30 | 0.30 | 0.40 | 0.40 |
| **All** | **15** | **0.66** | **0.64** | **0.54** | **0.54** |

- **One score changed:** question 4's precision, from 0.33 to 0.00. All other 29 scores are the same as in the baseline.
- **Retrieval was the same:** the same pages in the same order for all 15 questions. Two relevance scores moved in the second decimal (question 4's page 124, 0.49 to 0.48; question 14's page 50, 0.68 to 0.67).

## Findings

### 1. The images changed no score

Only 2 of the 15 test cases retrieved a chart: questions 11 (page 7) and 12 (page 24). In both, the chart description ranked second, below the page's own text, and both scored 1.00 for precision and recall, as in the baseline. Question 13 retrieved no chart (see finding 3).

So on this golden set, giving the judge the images makes no difference to the retriever's scores. It does make the judge see what the answering model sees, which matters more for Faithfulness in stage 2, where an answer may state something read off a chart.

### 2. The one change is the judge varying between runs

Question 4 retrieved the same three chunks both times. The baseline judge counted the stockholder proposal on page 82, which says NVIDIA employs "more than 35,000 people", as relevant to the workforce question, and scored precision 0.33. This run's judge did not, and scored 0.00. Both readings are defensible, since the proposal gives an older, rounder headcount than the reference answer's "approximately 42,000".

So one question's score can move by a third between runs, while the means over 15 questions moved by 0.02 at most.

### 3. A correction to the baseline: question 13 was a retrieval shortfall, not a judge mistake

The baseline findings first said the judge wrongly missed the other NEOs' pay mix in question 13. That was wrong, and the baseline file is now corrected:

- The chunk question 13 retrieved is `page57-2`, the page's text, not the chart description. LlamaParse wrote the CEO half of the chart into the page's text as a table (with values like "44", without "%").
- The other NEOs' half is in the next chunk, `page57-3`, which ranks 6th.
- So no retrieved chunk holds that half, and the judge was right.

This also shows that, on page 57 at least, a chart's values reach the vector store twice: in the chart's description and in the page's text, where they can be split across chunks.

## Cost and time

- **Time:** 20 seconds of scoring, against 12 in the baseline; 38 seconds for the whole notebook.
- **Cost: not measured.** `LocalModel` reports no cost (deepeval shows "token cost: None"), and the DeepSeek account's balance read $1.85 before the run, after it, and again several minutes later. So either the balance updates with a delay, or the change was too small to show at cent precision. DeepSeek's usage page has the real figure.

## Limitations

- **Only 2 test cases had an image,** both already scoring 1.00, so this run says little about how well the judge reads charts.
- The baseline's limitations still apply: 15 questions, one run each, and precision is coarse with 3 chunks.
