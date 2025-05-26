Final Assignment #2 - CEU MSBA - Deep Learning, Foundation Models Final Assignment: Robustness Testing via Prompt Perturbation

Overview: In this assignment, you will evaluate the robustness of foundation models when exposed to variations in prompt phrasing, using a consistent NLP task. You’ll explore how sensitive models are to wording, tone, structure, and input order — and assess whether their outputs remain reliable under these changes.

Objectives:
Develop and apply systematic prompt perturbation techniques.
Evaluate model robustness across multiple variations.
Compare model behavior and reflect on implications for real-world deployment.

Step 1: Choose a Task Pick one prompted task (examples below):
Sentiment classification (positive / neutral / negative)
Named entity extraction (e.g., people, organizations, products)
Summarization (1–2 sentence summaries)
Classification of customer reviews or support tickets
Use a dataset with at least 20–50 test examples. You may use Kimola NLP, SST-2, Yelp, or propose another task/data source with instructor approval.

Step 2: Design Prompt Variants
Write 1 Base Prompt for your chosen task.
Create at least 5 prompt perturbations, varying along dimensions such as:
Wording/synonym use
Formality or tone
Instruction phrasing (e.g., imperative vs. question)
Order of inputs/instructions
Clearly document your perturbation strategy and goals (e.g., subtle change vs. structural rephrasing).

Step 3: Model Selection and Runs
Use at least two foundation models: optionally one from OpenAI (e.g., GPT-4), and one from Hugging Face (e.g., Mistral, LLaMA, Falcon).
Run each model with all prompt variants on the same data.
Save all raw outputs for comparison.

Step 4: Evaluate Robustness Choose at least two of the following evaluation strategies:
1. Output Consistency
Classification: % of identical predictions across prompt variants.
Generation: Compute pairwise similarity (e.g., ROUGE, cosine similarity using sentence embeddings).
2. Output Quality Across Variants
Use human or rubric-based scoring to assess whether prompt changes affect:
Fidelity (accuracy of content)
Fluency
Factual correctness
3. Statistical Stability
Compute entropy or variance in predictions across prompt versions.
4. Visualization (Optional Bonus)
Heatmaps or plots showing variance per prompt or model.

Step 5: Reflection and Reporting
Present a table summarizing consistency/quality results.
Show examples of both robust and fragile prompt-model pairs.
Reflect on:
Which kinds of changes affected performance most?
Which model was more robust overall?
What are the implications for reliability in production use?

Deliverables:
A Jupyter or Colab notebook with:
All prompt variants
Code for model execution and evaluation
Tables/visualizations of results
A 2–3 page written report (PDF) with your analysis and reflections

Grading Criteria:
35%: Quality and creativity of prompt perturbation strategy
30%: Completeness and rigor of evaluation
20%: Clarity of analysis and conclusions
15%: Insight into model behavior and deployment considerations

Submission Deadline: 25/05/2025
Submission Format: Email me to RubiaE@ceu.edu a ZIP file with your notebook, report (PDF), and a README if needed.

Resources:
Hugging Face Models: https://huggingface.co/models
Kimola NLP Dataset: https://github.com/Kimola/nlp-datasets
Sentence Transformers: https://www.sbert.net/
ROUGE Scorer: https://pypi.org/project/rouge-score/
OpenAI API (GPT-4): https://platform.openai.com/docs
Cosine similarity in Python: sklearn.metrics.pairwise.cosine_similarity

Optional Bonus:
Try using prompt tuning (e.g., PEFT or adapters) to reduce output variance
Submit a leaderboard table ranking prompt robustness by model and task

