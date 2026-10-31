# Is the answer correct? Judged by Liquid AI's d1 (October 31, 2026)

Stage 3 of the eval suite: the whole pipeline, judged on whether its answers are right. The earlier stages never checked that. Faithfulness, at its default, only catches claims that contradict the retrieved chunks, and Answer Relevancy only checks that the answer stays on the question.

- **Notebook:** `evals/notebooks/correctness_eval.ipynb`. Its outputs are this run, committed together with this file.
- **Answers:** the 15 that the agent (`deepseek-flash`) gave in stage 2, saved in `evals/notebooks/generator_answers.json`. No agent ran and no DeepSeek call was made.
- **References:** the reference answers in `evals/goldens/retriever_goldens.json`, written from the PDF
- **Judge:** Liquid AI's `d1.1`, through deepeval 4.3.2's `JevEval` and `TypeSafeModel`, pointed at `https://api.liquid.ai/decisions`

## What was tested

One "Correctness" metric compared each answer with its reference answer by asking d1 three questions:

| Question to d1 | Type | Weight |
|---|---|---|
| The answer contains no fact that contradicts the reference answer. | yes or no | 2 |
| The answer states every fact in the reference answer. | yes or no | 1 |
| How closely does the answer match the reference answer? | score: contradicts it, mostly wrong, mostly right, fully matches | 1 |

A yes-or-no question's value is d1's probability of "yes". The score question's value is the level d1 expects, from 0 for "contradicts it" to 1 for "fully matches". The metric is their weighted mean, so a contradiction counts twice as much as a missing fact, and facts an answer adds beyond the reference do not count against it.

## Results

| Content | Questions | Mean correctness |
|---|---|---|
| Text | 5 | 0.96 |
| Chart | 3 | 0.77 |
| Table | 5 | 0.76 |
| Two pages | 2 | 0.51 |
| **All** | **15** | **0.80** |

| # | Content | Question | No contradiction | Every fact | Closeness (most likely) | Correctness |
|---|---|---|---|---|---|---|
| 1 | text | The December 2025 Groq agreement | 0.96 | 0.96 | Fully matches (0.88) | 0.96 |
| 2 | text | Share buyback authorization left | 0.99 | 0.97 | Fully matches (0.87) | 0.98 |
| 3 | text | Why gross margin fell in fiscal 2026 | 0.99 | 0.98 | Fully matches (0.93) | 0.98 |
| 4 | text | Workforce size and research staff | 0.96 | 0.86 | Fully matches (0.79) | 0.93 |
| 5 | text | The 2026 annual meeting | 0.98 | 0.93 | Fully matches (0.84) | 0.96 |
| 6 | table | PwC's fees | 0.99 | 0.03 | Mostly right (0.73) | 0.68 |
| 7 | table | Data Center revenue, compute and networking | 0.93 | 0.07 | Mostly right (0.70) | 0.65 |
| 8 | table | Revenue from customers in China | 0.82 | 0.12 | Mostly right (0.76) | 0.61 |
| 9 | table | Non-employee director pay | 0.92 | 0.94 | Fully matches (0.77) | 0.92 |
| 10 | table | Largest shareholders | 0.95 | 0.93 | Fully matches (0.82) | 0.94 |
| 11 | chart | Open-source AI repositories | 0.85 | 0.83 | Fully matches (0.52) | 0.84 |
| 12 | chart | Cash returned over 1, 3 and 5 years | 0.97 | 0.94 | Fully matches (0.82) | 0.96 |
| 13 | chart | The CEO's target pay mix | 0.65 | 0.11 | Mostly right (0.70) | 0.52 |
| 14 | two pages | Committee chairs and meetings | 0.81 | 0.04 | Mostly right (0.82) | 0.58 |
| 15 | two pages | Cash equivalents and marketable securities, two years | 0.47 | 0.24 | Mostly right (0.57) | 0.44 |

Each answer, its reference and d1's full reason are in the notebook.

## Findings

### 1. No answer was confidently judged wrong

"No contradiction" was 0.81 or more for 13 of the 15 answers. The two below that (questions 13 and 15) are explained in finding 3. Nine answers were judged "fully matches" as the most likely level, and the other six "mostly right"; none was most likely "mostly wrong" or "contradicts it".

### 2. Three low scores come from the reference answers, not the answers

For questions 6, 7 and 8, the answer gives what the question asks, but "every fact" fails because the reference answer holds more:

- **Question 6** asks for PwC's fiscal 2026 fees and the audit part. The answer gives both ($13,361,322 and $10,166,400). The reference also gives fiscal 2025's total and the other three fee lines.
- **Question 7** asks how Data Center revenue splits between compute and networking. The answer gives fiscal 2026's split. The reference also gives fiscal 2025's.
- **Question 8** asks about China. The answer gives both years and the change. The reference also gives the United States, Taiwan and other countries.

The reference answers were written for the retriever stage, to say everything the expected page holds on the subject, so they are fuller than an ideal answer. d1 applied the question as written; the golden set is what makes these three scores look worse than the answers are.

### 3. Three low scores are real gaps, each traced to an earlier finding

- **Question 13 (0.52) lacks the other NEOs' pay mix.** That half of the page 57 chart is in a chunk the agent did not retrieve (the retriever baseline found the same). The answer also says "over 90%" of the CEO's target pay is performance-based, citing page 27, where the reference has 96% from page 57, and its table rounds cash to "~12%" while its text says 12.7%.
- **Question 14 (0.58) lacks the Audit Committee's meeting count.** The parse lost that sentence (retriever baseline, finding 3), and the answer said so instead of guessing.
- **Question 15 (0.44) answers a related total.** It reports cash, cash equivalents and marketable securities from the balance sheet ($62.6 billion and $43.2 billion), while the reference gives Note 7's cash equivalents and marketable securities ($65,076 million and $42,487 million). The answer mentions Note 7's $65.1 billion in a reconciliation note, but its main figures are a different measure, which d1 read as a likely contradiction (0.47).

### 4. Text answers were the most correct, and the mixed results match the earlier stages

Text questions scored 0.96. The two-page questions scored lowest (0.51), and both of their gaps come from the parse and the retrieval problems found in stage 1. Table questions (0.76) are pulled down by the fuller reference answers in finding 2. Without questions 6 to 8, the tables' mean would be 0.93.

## Cost and time

- **Time:** d1 judged the 15 answers in 9 seconds, one request each; the whole notebook took 15 seconds.
- **Cost:** no DeepSeek calls. deepeval reports no cost for `d1.1`, and Liquid's pricing for this key was not checked.

## Limitations

- **The reference answers are fuller than the questions,** so "every fact" penalizes concise answers (finding 2). Shorter references, holding only what each question asks, would make this metric fairer.
- **One judging of one set of answers.** d1's numbers move slightly between runs (the off-topic eval saw changes of up to 0.05 in probability), and the agent's answers change from run to run.
- **Chart values on page 7 are approximate** in the reference ("more than 400", "about 200"), while the answer gives the chart's estimated counts (420, 340 and so on). d1 still judged it "fully matches", but with its lowest probability for that level (0.52).
