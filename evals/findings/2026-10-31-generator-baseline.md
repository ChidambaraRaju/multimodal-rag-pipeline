# Generator evaluation: baseline (October 31, 2026)

The first evaluation of ReportLens's answers: the real agent answers each question, and a larger model judges each answer.

- **Notebook:** `evals/notebooks/generator_eval.ipynb`. Its outputs are this run, committed together with this file.
- **Answers:** `evals/notebooks/generator_answers.json`, the agent's 15 answers with their searches and every chunk those searches returned
- **Questions:** the 15 questions in `evals/goldens/retriever_goldens.json`. Their reference answers are not used.
- **Answering model:** `deepseek-flash`, through the agent as the chatbot runs it
- **Judge:** `deepseek-v4-pro` on DeepSeek's own API, through deepeval's `DeepSeekModel`. It reads text only, so a retrieved chart reaches it as its description.
- **Metrics:** deepeval 4.3.2's `FaithfulnessMetric` and `AnswerRelevancyMetric`, with their default settings

## What was tested

Each question went to `answer_question()`. The agent wrote its own searches, read the chunks (and the images of any charts) they returned, and wrote an answer with page citations. Each test case holds:

- `input`: the question
- `actual_output`: the agent's answer
- `retrieval_context`: every chunk from all of the agent's searches, each listed once, as text

What the two metrics do:

- **Faithfulness:** the judge lists the claims in the answer and the facts in the context, then gives each claim a verdict: "yes", "no" or "borderline". The score is the share of claims that pass. **With the default settings, only "no" fails, and the judge is told to use "no" only when the context directly contradicts the claim.** A claim the context does not back up is "borderline", which passes unless `penalize_ambiguous_claims=True`. So this run's Faithfulness measures "nothing contradicts the context", not "everything is supported by it".
- **Answer Relevancy:** the judge lists the statements in the answer and decides whether each one addresses the question. The score is the share that do.

## Results

| Content | Questions | Mean faithfulness | Mean answer relevancy |
|---|---|---|---|
| Text | 5 | 1.00 | 0.88 |
| Table | 5 | 1.00 | 0.95 |
| Chart | 3 | 1.00 | 0.85 |
| Two pages | 2 | 1.00 | 0.89 |
| **All** | **15** | **1.00** | **0.90** |

| # | Content | Question | Searches | Expected page found | Faithfulness | Answer Relevancy |
|---|---|---|---|---|---|---|
| 1 | text | The December 2025 Groq agreement | 2 | yes | 1.00 | 0.90 |
| 2 | text | Share buyback authorization left | 2 | yes | 1.00 | 1.00 |
| 3 | text | Why gross margin fell in fiscal 2026 | 2 | yes | 1.00 | 1.00 |
| 4 | text | Workforce size and research staff | 2 | yes | 1.00 | 0.50 |
| 5 | text | The 2026 annual meeting | 1 | yes | 1.00 | 1.00 |
| 6 | table | PwC's fees | 1 | yes | 1.00 | 1.00 |
| 7 | table | Data Center revenue, compute and networking | 1 | yes | 1.00 | 1.00 |
| 8 | table | Revenue from customers in China | 2 | yes | 1.00 | 0.75 |
| 9 | table | Non-employee director pay | 2 | yes | 1.00 | 1.00 |
| 10 | table | Largest shareholders | 2 | yes | 1.00 | 1.00 |
| 11 | chart | Open-source AI repositories | 2 | yes | 1.00 | 0.89 |
| 12 | chart | Cash returned over 1, 3 and 5 years | 1 | yes | 1.00 | 1.00 |
| 13 | chart | The CEO's target pay mix | 2 | yes | 1.00 | 0.67 |
| 14 | two pages | Committee chairs and meetings | 75 | yes | 1.00 | 1.00 |
| 15 | two pages | Cash equivalents and marketable securities, two years | 2 | partly | 1.00 | 0.79 |

"Expected page found" is not a metric. It says whether the agent's searches reached the golden set's pages. The full answers and the judge's reasons are in the notebook.

## Findings

### 1. Faithfulness scored 1.00 everywhere, but it let through figures the report never states

Five answers contain figures the model worked out itself, and all passed as "borderline":

| # | Figure in the answer | What the report states |
|---|---|---|
| 2 | "roughly $57.0 billion" of authorization left after February 20, 2026 (the answer says the report does not state it) | $58.5 billion as of January 25, 2026, and $1.5 billion repurchased since |
| 7 | compute "about 83.8%" and networking "about 16.2%" of Data Center revenue | the dollar amounts only |
| 8 | China revenue "down $5.4 billion (about 21%)" | $19,677 million and $25,048 million |
| 13 | cash 12.7% and stock awards 87.3% of the CEO's target pay, and "~12%" in the same answer's table | the dollar targets and the chart's 4%, 8%, 44% and 44% |
| 15 | a prior-year total of $43,210 million and growth of "roughly 45%" | the separate lines on the balance sheet, and the $62.6 billion for fiscal 2026 |

All of these are arithmetically correct (checked by hand), so no answer in this run is wrong. But the default metric would also pass a wrong figure, as long as nothing in the context contradicts it, so it cannot catch the "model adds its own arithmetic" issue recorded in CLAUDE.md. `penalize_ambiguous_claims=True` makes "borderline" fail and would count these.

### 2. The agent's own searches found the expected page for 14 of 15 questions

Stage 1 sent each raw question to the search tool and found the expected page for 9 of 15. Here the agent wrote its own queries, often two, and found the expected page for every question except half of question 15 (page 153, not 154). It rephrased in the report's own terms: "share repurchase authorization remaining amount" (question 2), "non-employee director compensation fiscal 2026" (question 9), "largest shareholders beneficial ownership" (question 10). So stage 1 understates the retrieval the chatbot actually gets; it measures the search tool alone.

### 3. A fact missing from the parse made the agent search 75 times

For question 14, the agent searched 75 times for the Audit Committee's meeting count, guessing phrasings such as "AC met four times", "AC met six times" and "on eight occasions". That sentence is in no chunk: LlamaParse lost it (retriever baseline findings, finding 3). In the end it answered correctly that the retrieved text does not state the count, and gave the other two committees' counts. The other 14 questions made 1 or 2 searches each.

- **Nothing limits the number of searches,** so a fact the store does not hold costs time and money. All 15 answers took 149 seconds, and question 14's 75 searches brought 28 chunks into its context.
- **A wrong citation:** the answer cites page 43 for A. Brooke Seawell chairing the Audit Committee, because the parse put page 44's Audit Committee box in page 43's chunk. A reader checking page 43 of the PDF would not find it.

### 4. Answer Relevancy marks down extra context, not wrong answers

Six answers lost relevancy, each for adding related facts beyond the question:

- Question 4 (0.50) added the share of technical roles, advanced degrees and the turnover rate.
- Question 13 (0.67) added actual payouts and prior-year comparisons.
- Question 8 (0.75) added total revenue and the share of revenue from outside the United States.
- Question 15 (0.79) added cash held abroad and lock-up details.
- Question 11 (0.89) added the chart's source and growth over time.
- Question 1 (0.90) added an execution risk from page 102.

Whether this extra context is a flaw is a product choice, but the metric counts it as one. Question 4's extra facts are the same ones the golden set's reference answer includes.

### 5. Question 15 answered a related total

It reported cash, cash equivalents and marketable securities from the balance sheet ($62.6 billion, the figure the MD&A also states), then reconciled it with Note 7's $65.1 billion of cash equivalents and marketable securities. That is a reasonable reading of the question, but not the figure the reference answer gives.

## Cost and time

- **Time:** 149 seconds answering, 107 seconds scoring, 266 seconds for the whole notebook.
- **Judge tokens: not reported.** deepeval reported no tokens for `deepseek-v4-pro` (the notebook prints 0). In stage 1 it reported them for `deepseek-flash`, which it had been given prices for.
- **Cost:** about $0.08 for answering and judging together. The DeepSeek balance read $1.81 before answering, $1.79 just after scoring, and $1.73 two minutes later, so it updates with a delay.

## Limitations

- **Faithfulness used its default settings,** which pass unsupported claims (finding 1).
- **The judge saw charts as descriptions.** Only questions 11 and 12 had a chart in their context.
- **One run.** The agent's answers and searches change from run to run; this run's are saved in `generator_answers.json`, so they can be scored again without the agent.
- **Answer Relevancy depends on how the judge splits an answer into statements,** so a long answer with extra context scores lower than a short one with the same core facts.
