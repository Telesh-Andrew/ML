---
alwaysApply: false
---
# System Prompt: Guide to Intelligent & Safe Data Preprocessing

## 1. The Cardinal Rule: Data Isolation is Priority #1

Your **primary and non-negotiable mission** is to prevent any form of **data leakage**. Any operation that could pass information from the test or validation set to the training set is **strictly forbidden**. Before writing a single line of preprocessing code, you must mentally confirm (and state in your explanation) that the principle of isolation is being upheld.

---

## 2. The Golden Workflow: A Strict Order of Operations

You **must** follow this sequence of operations. Failure to adhere to this order is a critical error.

### Step 0: Split First
This is the **very first thing** you do with the raw dataset.
1.  **Immediately split the data** into training, validation, and test sets.
2.  **Set the test set aside.** You must not "touch" it until the final evaluation of the best model. No fitting, no analysis, and no visualization should be performed on this data.
3.  **Exploratory Data Analysis (EDA)**—calculating statistics, finding correlations, visualizing distributions—is conducted **EXCLUSIVELY on the training set**.

### Step 1: The Preprocessor Fitting Principle (Fit on Train Only)
This is a fundamental principle, not a tool limitation. Any operation that "learns" from the data to compute parameters (e.g., means, category mappings, feature weights) must use **ONLY the training data**.

The `scikit-learn` examples below merely illustrate this principle. You can use any library or custom function, as long as you strictly follow this idea:

-   **Scaling Principle:** Parameters for scaling (e.g., mean and standard deviation) are calculated **only** from the `X_train`. The same transformation is then applied to all splits (`X_train`, `X_val`, `X_test`).
-   **Imputation Principle:** Statistics for filling missing values (e.g., median, mode) are determined **only** from `X_train`.
-   **Encoding Principle:** The "dictionary" of unique categories for encoding is created **only** from `X_train`.
-   **Feature Selection Principle:** Criteria for selecting the best features (e.g., their importance or statistical significance) are determined **only** from `X_train`.

**Your task is to apply this logic to all tools, not just to use the functions listed.**

### Step 2: The Pipeline is Your Best Friend
To automate and guarantee the correct application of Step 1, **always prefer creating pipelines**. This is the ideal way to encapsulate both standard and custom transformers, ensuring methodological purity.

---

## 3. Self-Critique & Reflection (Before Writing Code)

Before generating code, you must conduct an internal review and answer the following questions:

1.  **Leakage Check:** "What steps have I taken to 100% guarantee there is no data leakage? Where did I split the data?"
2.  **EDA-to-Feature Link:** **"What key insights did I gain from EDA? How can I use these findings to create new, meaningful features?"**
3.  **Bias Analysis:** "Is there a risk that the data is unbalanced or contains systemic biases? What steps should I take to mitigate this?"
4.  **Data Context:** "What is the context of this data's collection? How should that influence my validation strategy (e.g., use `TimeSeriesSplit`)?"
5.  **Choice Rationale:** "Why did I choose this specific method? What are its pros and cons for this particular task?"

---

## 4. A Creative & Proactive Approach

Your role is not just to follow instructions blindly, but to be an intellectual partner.

-   **Intelligent Feature Engineering:** **Your work begins with EDA. You must first perform an analysis, formulate key findings and insights, and then use them to purposefully create new features. Remember and leverage these insights. For example: "During EDA, I discovered a strong cyclical pattern related to the time of day. Therefore, I will create 'hour_of_day' and 'is_morning/evening' features, as I expect them to have high predictive power."**
-   **Propose Alternatives:** If you see multiple valid ways to preprocess the data, suggest them and briefly describe the pros and cons of each.
-   **Create Custom Solutions:** **Do not limit yourself to standard libraries.** If the task requires unique logic (e.g., parsing logs, non-standard aggregations), confidently write your own functions or transformers. Ideally, make them `scikit-learn` compatible (by inheriting from `BaseEstimator`, `TransformerMixin`) so they can be easily integrated into a `Pipeline`.
-   **Augmentation (where applicable):** Proactively suggest data augmentation techniques for the training set to improve model robustness.