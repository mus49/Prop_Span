# %%
import ast
import json
import os
import re
import string
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import nltk
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tqdm import tqdm

# Configure NLTK data path for offline use
nltk.data.path.append('') #add your local path where nltk data is stored

# %%
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
DEFAULT_API_BASE = os.environ.get("OPENAI_API_BASE", "http://localhost:8001/v1")

os.environ.setdefault("OPENAI_API_KEY", DEFAULT_API_KEY)
os.environ.setdefault("OPENAI_API_BASE", DEFAULT_API_BASE)

# %%
print("update")
# 🔹 Remove HTML tags and extra whitespace
def clean_span(span: str) -> str:
    # print(re.sub(r"</?span>", "", span).strip())
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
    # print([s.strip() for s in spans])
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
    ])

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
        # print("lcs tokens : []")
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
    # print("lcs tokens : ", end="")
    # print(lcs_tokens)
    
    return best_len


# ---------- main scorer ----------
def token_level_direct_f1(gt_spans: List[str], pred_spans: List[str]):
    """
    Direct GT <-> Pred comparison using LCS-based matching with stopword removal.
    """
    gt_spans = [s.strip() for s in gt_spans if s and s.strip()]
    # print('gt_spans :', gt_spans)
    pred_spans = [s.strip() for s in pred_spans if s and s.strip()]
    # print('pred_spans :', pred_spans)

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
        # print(' (precision) :')
        pred_toks = tokenize(pred)
        # print('pred_toks :', pred_toks)

        if not pred_toks:
            precision_scores.append(0.0)
            continue

        best_ratio = 0.0
        for gt in gt_spans:
            lcs_len = lcs_length(pred, gt)
            ratio = lcs_len / len(pred_toks)
            if ratio > best_ratio:
                best_ratio = ratio
        
        # print('best_ratio (precision) :', best_ratio)

        precision_scores.append(best_ratio)

    P = sum(precision_scores) / len(pred_spans)

    # Recall
    recall_scores = []
    for gt in gt_spans:
        # print(' (recall) :')
        gt_toks = tokenize(gt)
        # print('gt_toks :', gt_toks)
        if not gt_toks:
            recall_scores.append(0.0)
            continue

        best_ratio = 0.0
        for pred in pred_spans:
            lcs_len = lcs_length(pred, gt)
            ratio = lcs_len / len(gt_toks)
            if ratio > best_ratio:
                best_ratio = ratio
                
        # print('best_ratio (recall):', best_ratio)

        recall_scores.append(best_ratio)

    R = sum(recall_scores) / len(gt_spans)

    F1 = 2 * P * R / (P + R) if (P + R) else 0.0

    return {"Precision": round(P, 4), "Recall": round(R, 4), "F1": round(F1, 4)}

# %%
# Initialize LLM (local vLLM endpoint)
llm_init = ChatOpenAI(
    model_name="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base=os.environ["OPENAI_API_BASE"],
    temperature=0.15,
)

# %%
import time
from openai import RateLimitError, APIConnectionError, APITimeoutError

def safe_invoke(invocable, payload, max_retries=10, base_delay=5):
    """Retry wrapper around LangChain-compatible invocables."""
    for attempt in range(max_retries):
        try:
            return invocable.invoke(payload)

        except RateLimitError:
            wait = base_delay * (2 ** attempt)
            print(f"[429 Rate Limit] Retrying in {wait:.1f}s...")
            time.sleep(wait)

        except (APIConnectionError, APITimeoutError):
            wait = base_delay * (2 ** attempt)
            print(f"[Network Issue] Retrying in {wait:.1f}s...")
            time.sleep(wait)

        except Exception as e:
            print(f"Unexpected error: {e}")
            wait = base_delay * (2 ** attempt)
            print(f"[Unexpected] Retrying in {wait:.1f}s...")
            time.sleep(wait)

    print("❌ Max retries reached — returning None.")
    return None


# %%
# --- your 20-tech mapping (names you provided) ---
tech_name = {
    "T1": "Causal Oversimplification",
    "T2": "Black-and-white Fallacy",
    "T3": "Straw Man",
    "T4": "Whataboutism",
    "T5": "Reductio ad Hitlerum",
    "T6": "Red Herring",
    "T7": "Loaded Language",
    "T8": "Name Calling and Labeling",
    "T9": "Appeal to emotion/fear/prejudice",
    "T10": "Slogans",
    "T11": "Flag-Waving",
    "T12": "Exaggeration and Minimization",
    "T13": "Thought-terminating Cliches",
    "T14": "Bandwagon",
    "T15": "Smears",
    "T16": "Obfuscation, intentional vagueness, confusion",
    "T17": "Doubt",
    "T18": "Appeal to Authority",
    "T19": "Assertion",
    "T20": "Glittering Generalities (Virtue)"
}

TECH_IDS = [1, 2, 3, 4, 5, 6, 7, 9, 15, 19]
R_VALUES = [1,2,3,4,5]
MAX_PARALLEL_TECHNIQUES = int(os.environ.get("MAX_PARALLEL_TECHNIQUES", "6"))


def create_llm_client():
    return ChatOpenAI(
        model_name="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_base=os.environ["OPENAI_API_BASE"],
        temperature=0.15,
    )


def build_prompt(support_text: str, full_article: str, technique: str) -> str:
    return f""" Here is a example: 
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


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def run_inference_for_tech(tech_num: int):
    tech_id = f"T{tech_num}"
    technique = tech_name[tech_id]

    print(f"=== Starting T{tech_num}: {technique} ===")

    llm_client = create_llm_client()
    
    #change this to the path where your final iteration file from the refinement stage is stored.
    path_rf = f"/cases/i_bias/T{tech_num}_finalized_iter_25.csv" 
    improved_rf = pd.read_csv(path_rf)
    #change this to the path where your test file for T{tech_num} is stored.
    path_df = f"/Formatted_Data/Test/Test_T{tech_num}.csv"
    df = pd.read_csv(path_df)
    N = len(df)

    for R in R_VALUES:
        r_start_time = time.perf_counter()

        examples_pick_start = time.perf_counter()
        topk_df = improved_rf.sample(n=R, random_state=42)
        examples_pick_time_s = time.perf_counter() - examples_pick_start

        examples_build_start = time.perf_counter()
        support_text = ""
        for _, row in topk_df.iterrows():
            example_article = row["article"]
            example_output = row["final_cot"]
            support_text += f"Example Article:\n{example_article}\nExample Response:\n{example_output}\n\n"
        examples_build_time_s = time.perf_counter() - examples_build_start

        results_log = []
        for idx in tqdm(range(N), desc=f"T{tech_num} R={R}", leave=False):
            inference_start = time.perf_counter()

            article_pick_start = time.perf_counter()
            full_article = df.loc[idx, "Article"]
            gt_raw = df.loc[idx, "validated_spans"]
            try:
                gt = ast.literal_eval(gt_raw)
            except (ValueError, SyntaxError):
                gt = []
            article_pick_time_s = time.perf_counter() - article_pick_start

            prompt_build_start = time.perf_counter()
            prompt_input = build_prompt(support_text, full_article, technique)
            messages = [HumanMessage(content=prompt_input)]
            prompt_build_time_s = time.perf_counter() - prompt_build_start

            llm_call_start = time.perf_counter()
            out = safe_invoke(llm_client, messages)
            llm_call_time_s = time.perf_counter() - llm_call_start
            if out is None:
                continue

            final_output_start = time.perf_counter()
            response_text = getattr(out, "content", None)
            if response_text is None and isinstance(out, dict):
                response_text = out.get("text")
            if response_text is None:
                response_text = str(out)

            new_pred = extract_predicted_spans(response_text)
            metrics = token_level_direct_f1(gt, new_pred)

            final_output_time_s = time.perf_counter() - final_output_start
            total_inference_time_s = time.perf_counter() - inference_start

            results_log.append({
                "article_id": idx,
                "article": full_article,
                "gt_spans": gt,
                "cot": response_text,
                "pred": new_pred,
                "F1": round(metrics["F1"], 3),
                "precision": round(metrics["Precision"], 3),
                "recall": round(metrics["Recall"], 3),
                "examples_pick_time_s": round(examples_pick_time_s, 4),
                "examples_build_time_s": round(examples_build_time_s, 4),
                "article_pick_time_s": round(article_pick_time_s, 4),
                "prompt_build_time_s": round(prompt_build_time_s, 4),
                "llm_call_time_s": round(llm_call_time_s, 4),
                "final_output_time_s": round(final_output_time_s, 4),
                "total_inference_time_s": round(total_inference_time_s, 4)
            })

        res_df = pd.DataFrame(results_log)
        # change this to your output path
        output_path = f"/i_bias/R={R}/T{tech_num}_K1_inference_Random_whole.csv"
        ensure_dir(output_path)
        res_df.to_csv(output_path, index=False)
        r_total_time_s = time.perf_counter() - r_start_time
        print(f"T{tech_num} R={R} → {output_path}")
        print(
            f"Timing T{tech_num} R={R}: total={r_total_time_s:.2f}s, "
            f"examples_pick={examples_pick_time_s:.4f}s, examples_build={examples_build_time_s:.4f}s"
        )

    print(f"=== Completed T{tech_num} ===")


if __name__ == "__main__":
    print(f"Running techniques with up to {MAX_PARALLEL_TECHNIQUES} parallel workers")
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_TECHNIQUES) as executor:
        future_map = {executor.submit(run_inference_for_tech, tech_num): tech_num for tech_num in TECH_IDS}
        for future in as_completed(future_map):
            tech_num = future_map[future]
            try:
                future.result()
            except Exception as exc:
                print(f"❌ T{tech_num} failed: {exc}")
    

# %%


# %%


# %%

