# %%
# add your technique number and name here
tech_num = ""
technique = ""

# %%
import pandas as pd
import random
import time
from tqdm import tqdm
import ast
import json
import re
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError
from huggingface_hub import hf_hub_download
import os
import difflib
from typing import List
import nltk
import string
import logging
import numpy as np

# Configure NLTK data path for offline use
nltk.data.path.append('')

# %%
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# %%
# Initialize vLLM client (OpenAI-compatible) - Using GPU 1
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1" #change this if your vLLM server is running on a different host/port

TEMP = 0.15
MAX_TOK = 50000

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Get model name from server
models = client.models.list()
model = models.data[0].id

# Load system prompt from Hugging Face
def load_system_prompt(repo_id: str, filename: str) -> str:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()
    return system_prompt

model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
#SYSTEM_PROMPT = load_system_prompt(model_id, "SYSTEM_PROMPT.txt")
SYSTEM_PROMPT_PATH = "/models/Mistral-Small-3.2-24B-Instruct-2506/SYSTEM_PROMPT.txt" #change this to your local path where SYSTEM_PROMPT.txt is stored
with open(SYSTEM_PROMPT_PATH, "r") as f:
    SYSTEM_PROMPT = f.read()

print(f"✅ Connected to vLLM server with model: {model}")

# %%
# 🔹 Remove HTML tags and extra whitespace
def clean_span(span: str) -> str:
    return re.sub(r"</?span>", "", span).strip()

def extract_recipes(gpt_output: str):
    """
    Extract all <result>...</result> blocks from GPT output.
    Returns a list of strings (each string is one full recipe).
    """
    recipes = re.findall(r"<result>.*?</result>", gpt_output, re.DOTALL)
    return [r.strip() for r in recipes]

def extract_predicted_spans(gpt_output: str):
    raw = re.findall(r"<span>(.*?)</span>", gpt_output, re.DOTALL)
    spans = []
    for s in raw:
        if s.strip() in ["", "[]"]:  # treat empty or [] as no span
            continue
        spans.append(s.strip())
    return spans

def extract_CoT(gpt_output: str):
    """
    Extract all <span>...</span> tagged text from GPT output.
    Returns a list of strings.
    """
    spans = re.findall(r"<CoT>.*?</CoT>", gpt_output, re.DOTALL)
    return [s.strip() for s in spans]

# %%
# Try downloading stopwords if not available
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    # Fallback to hardcoded list if NLTK data not available
    STOPWORDS = set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
        "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
        'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
        'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
        'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
    ])  # use the list from above

# ---------- utilities ----------
_token_split_re = re.compile(r"\w+['']?\w+|\w+|[^\s\w]")  # basic word tokenizer including contractions

def tokenize(s: str) -> List[str]:
    """Tokenize → lowercase → remove punctuation-only tokens → remove stopwords."""
    if s is None:
        return []

    s = s.lower()
    toks = _token_split_re.findall(s)
    
    #remove punctuations
    toks = [t.strip(string.punctuation) for t in toks]

    # keep only tokens with at least one alphanumeric character
    toks = [t for t in toks if re.search(r"[a-z0-9]", t)]

    # remove stopwords
    toks = [t for t in toks if t not in STOPWORDS]

    return toks

def lcs_length(a: str, b: str) -> int:
    # Calculate the length of the longest common substring between two strings.
    a_toks = tokenize(a)
    b_toks = tokenize(b)
    
    if not a_toks or not b_toks:
        return 0
    
    m, n = len(a_toks), len(b_toks)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    best_len = 0
    best_end_i = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a_toks[i - 1] == b_toks[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > best_len:
                    best_len = dp[i][j]
                    best_end_i = i
            else:
                dp[i][j] = 0
    
    lcs_tokens = a_toks[best_end_i - best_len:best_end_i] if best_len else []
    
    return best_len

# ---------- main scorer ----------
def token_level_direct_f1(gt_spans: List[str], pred_spans: List[str]):
    """
    Direct GT <-> Pred comparison using LCS-based matching with stopword removal.
    """
    gt_spans = [s.strip() for s in gt_spans if s and s.strip()]
    pred_spans = [s.strip() for s in pred_spans if s and s.strip()]

    # special cases
    if not gt_spans and not pred_spans:
        return {"Precision": 1.0, "Recall": 1.0, "F1": 1.0}
    if gt_spans and not pred_spans:
        return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0}
    if not gt_spans and pred_spans:
        return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0}

    # Precision
    precision_scores = []
    for pred in pred_spans:
        pred_toks = tokenize(pred)

        if not pred_toks:
            precision_scores.append(0.0)
            continue

        best_ratio = 0.0
        for gt in gt_spans:
            lcs_len = lcs_length(pred, gt)
            ratio = lcs_len / len(pred_toks)
            if ratio > best_ratio:
                best_ratio = ratio

        precision_scores.append(best_ratio)

    P = sum(precision_scores) / len(pred_spans)

    # Recall
    recall_scores = []
    for gt in gt_spans:
        gt_toks = tokenize(gt)
        if not gt_toks:
            recall_scores.append(0.0)
            continue

        best_ratio = 0.0
        for pred in pred_spans:
            lcs_len = lcs_length(pred, gt)
            ratio = lcs_len / len(gt_toks)
            if ratio > best_ratio:
                best_ratio = ratio

        recall_scores.append(best_ratio)

    R = sum(recall_scores) / len(gt_spans)

    F1 = 2 * P * R / (P + R) if (P + R) else 0.0

    return {"Precision": round(P, 4), "Recall": round(R, 4), "F1": round(F1, 4)}

# %%
# Load the dataset and initial outputs
# change the paths to your local paths where the Train_T{tech_num}.csv and T{tech_num}_init_ZS.csv files are stored
path_df = f"/Formatted_Data/Train/Train_T{tech_num}.csv"
df = pd.read_csv(path_df)
print(path_df)
print('length of df:', len(df))

#initialization outputs from zero-shot prompting
path_output_df = f"/Intialization/T{tech_num}_init_ZS.csv"
output_df = pd.read_csv(path_output_df)
print(path_output_df)
print('length of output_df:', len(output_df))

# %%
# Get recipes and total number of articles
recipes = output_df['recipe'].copy().tolist()
N = len(df)

# %%
# === Function to pick random CoTs ===
def pick_random_cots(current_idx, K=1):
    """
    Selects K random CoTs using weighted sampling based on probability array.
    """
    candidate_indices = [j for j in range(N) if j != current_idx]

    candidate_probs = np.array([prob[j] for j in candidate_indices])
    candidate_probs = candidate_probs / candidate_probs.sum()

    if len(candidate_indices) < K:
        return candidate_indices

    return list(
        np.random.choice(
            candidate_indices,
            size=K,
            replace=False,
            p=candidate_probs
        )
    )

# %%
# === Gibbs Sampling Loop with Biased Selection ===
print("🚀 Starting Sampling for Reprompting Refinement (Biased Selection)...")

NUM_ITERATIONS = 25
K = 1
results_log = []

# Track updates per iteration: {article_id: [(iter, S_indices), ...]}
update_tracker = {i: [] for i in range(N)}

# --- Biased sampling probabilities (uniform initially) ---
prob = np.ones(N)
prob = prob / prob.sum()

BIAS_INCREMENT = 0.02
BIAS_MAX = 0.1

for iteration in range(NUM_ITERATIONS):
    print(f"\n📚 Gibbs Iteration {iteration+1}/{NUM_ITERATIONS}")

    for i in tqdm(range(N)):
        current_recipe = recipes[i]
        if not (isinstance(current_recipe, str) and current_recipe.strip()):
            continue

        full_article = df.loc[i, "Article"]
        
        gt_raw = df.loc[i, "validated_spans"]
        try:
            gt = ast.literal_eval(gt_raw) if isinstance(gt_raw, str) else gt_raw
        except:
            print(f"❌ Failed to parse gt on idx {i}")
            gt = []

        old_pred = extract_predicted_spans(current_recipe)
        old_f1 = token_level_direct_f1(gt, old_pred)["F1"]

        S_j = pick_random_cots(i, K=K)
        print(f"K: {K}")
        print(f"S_j: {S_j}")
        print(f"i: {i}")

        if not S_j:
            continue

        support_text = ""
        for k in S_j:
            example_article = df.loc[k, "Article"]
            example_output = recipes[k]
            support_text += f"Example Article:\n{example_article}\nExample Response:\n{example_output}\n\n"

        user_prompt = f"""Here is a example: 
{support_text}

Target Article:
{full_article}

You are an expert propaganda span detector. 
Your task is to carefully analyze the article and decide step by step (Chain of Thought) whether there are any statements that qualify as propaganda of type {technique}.

Instructions:
1. Use a step-by-step reasoning process inside <CoT>...</CoT> tags to decide if such propaganda spans exist.
2. If no spans are found, clearly state that in the CoT and output nothing inside <span> tags.
3. If spans are found, wrap each exact substring from the article in <span>...</span> tags. 

Output format (exactly):

<CoT>
    Step 1: ...
    Step 2: ...
    Step 3: ...
    Conclusion: Propaganda spans exist / do not exist
</CoT>

<!-- zero or more spans, depending on conclusion -->
<span>...</span>
<span>...</span>
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMP,
                max_tokens=MAX_TOK,
            )
            out = response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ Error at index {i}: {e}")
            continue
        
        # Wait after every LLM call (adjust as needed for local server)
        time.sleep(1)

        new_pred = extract_predicted_spans(out)
        new_f1 = token_level_direct_f1(gt, new_pred)["F1"]

        if new_f1 > old_f1:
            recipes[i] = out
            update_tracker[i].append((iteration + 1, S_j))
            # ---- BIAS UPDATE HERE ----
            print(f"bias before update for idx {i}: {prob[i]:.4f}")
            prob[i] = np.minimum(prob[i] + BIAS_INCREMENT, BIAS_MAX)
            print(f"bias after update for idx {i}: {prob[i]:.4f}")
            prob = prob / prob.sum()
            print(f"bias prob:", prob)
            # --------------------------
            print(f"✅ Updated prompt at index {i}: F1 {old_f1:.3f} → {new_f1:.3f}")
        else:
            print(f"⏩ Kept old prompt at index {i}: F1 {old_f1:.3f}")

        results_log.append({
            "article_id": i,
            "iteration": iteration + 1,
            "article": full_article,
            "gt_spans": gt,
            "cot": out,
            "pred": new_pred,
            "old_f1": round(old_f1, 3),
            "new_f1": round(new_f1, 3),
            "precision": round(token_level_direct_f1(gt, new_pred)["Precision"], 3),
            "recall": round(token_level_direct_f1(gt, new_pred)["Recall"], 3)
        })
    
    # Save log after each iteration
    log_df = pd.DataFrame(results_log)
    log_df.to_csv(f"/cases/i_bias/T{tech_num}_log_iter_{iteration + 1}.csv", index=False)
    
    print("🔖 Saving final results...")
    final_data = []
    for i in range(N):
        final_cot = recipes[i]
        article = df.loc[i, "Article"]
        gt_raw = df.loc[i, "validated_spans"]
        try:
            gt_spans = ast.literal_eval(gt_raw)
        except (ValueError, SyntaxError):
            gt_spans = []

        if final_cot:
            final_pred = extract_predicted_spans(final_cot)
            final_metrics = token_level_direct_f1(gt_spans, final_pred)
            final_f1 = final_metrics["F1"]
            final_pre = final_metrics["Precision"]
            final_rec = final_metrics["Recall"]
        else:
            final_pred = []
            final_f1, final_pre, final_rec = (0.0, 0.0, 0.0) if gt_spans else (None, None, None)

        # Format update history: [{"iter": N, "S": [indices]}, ...]
        update_history = [
            {
                "iter": int(iter_num),
                "S": [int(x) for x in s_indices]
            }
            for iter_num, s_indices in update_tracker[i]
        ]
        
        final_data.append({
            "article_id": i,
            "article": article,
            "gt_spans": json.dumps(gt_spans, ensure_ascii=False),
            "final_cot": final_cot,
            "final_pred": json.dumps(final_pred, ensure_ascii=False),
            "final_f1": round(final_f1, 3) if isinstance(final_f1, (float, int)) else final_f1,
            "final_pre": round(final_pre, 3) if isinstance(final_pre, (float, int)) else final_pre,
            "final_rec": round(final_rec, 3) if isinstance(final_rec, (float, int)) else final_rec,
            "updated": len(update_tracker[i]) > 0,
            "update_history": json.dumps(update_history, ensure_ascii=False)
        })

    final_df = pd.DataFrame(final_data)
    final_path = f"cases/i_bias/T{tech_num}_finalized_iter_{iteration + 1}.csv"
    final_df.to_csv(final_path, index=False)
    print(f"✅ Saved final results to: {final_path}")

print("🎉 All iterations completed!")

# %%
