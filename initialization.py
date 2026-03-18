#!/usr/bin/env python3
"""Initialize CoT "recipes" for a given propaganda technique using a local vLLM
(OpenAI-compatible) endpoint.

Outputs a CSV with per-article fields including: gt spans, predicted spans, CoT, and
LCS-token F1.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import string
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from tqdm import tqdm

import nltk


# Configure NLTK data path for offline use
nltk.data.path.append("")


logger = logging.getLogger(__name__)


# --------------------------- parsing helpers ---------------------------

def clean_span(span: str) -> str:
    return re.sub(r"</?span>", "", span).strip()


def extract_predicted_spans(model_output: str) -> List[str]:
    raw = re.findall(r"<span>(.*?)</span>", model_output, re.DOTALL)
    spans: List[str] = []
    for s in raw:
        if s.strip() in ["", "[]"]:
            continue
        spans.append(s.strip())
    return spans


def extract_cot(model_output: str) -> List[str]:
    blocks = re.findall(r"<CoT>.*?</CoT>", model_output, re.DOTALL)
    return [b.strip() for b in blocks]


# --------------------------- scoring (from notebook) ---------------------------

try:
    from nltk.corpus import stopwords

    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    STOPWORDS = set(
        [
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "you're",
            "you've",
            "you'll",
            "you'd",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "she's",
            "her",
            "hers",
            "herself",
            "it",
            "it's",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "that'll",
            "these",
            "those",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "a",
            "an",
            "the",
            "and",
            "but",
            "if",
            "or",
            "because",
            "as",
            "until",
            "while",
            "of",
            "at",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "to",
            "from",
            "up",
            "down",
            "in",
            "out",
            "on",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
        ]
    )


_token_split_re = re.compile(r"\w+['']?\w+|\w+|[^\s\w]")


def tokenize(s: str) -> List[str]:
    if s is None:
        return []

    s = s.lower()
    toks = _token_split_re.findall(s)
    toks = [t.strip(string.punctuation) for t in toks]
    toks = [t for t in toks if re.search(r"[a-z0-9]", t)]
    toks = [t for t in toks if t not in STOPWORDS]
    return toks


def lcs_length(a: str, b: str) -> int:
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

    _ = a_toks[best_end_i - best_len : best_end_i] if best_len else []
    return best_len


def token_level_direct_f1(gt_spans: List[str], pred_spans: List[str]) -> Dict[str, float]:
    gt_spans = [s.strip() for s in gt_spans if s and s.strip()]
    pred_spans = [s.strip() for s in pred_spans if s and s.strip()]

    if not gt_spans and not pred_spans:
        return {"Precision": 1.0, "Recall": 1.0, "F1": 1.0}
    if gt_spans and not pred_spans:
        return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0}
    if not gt_spans and pred_spans:
        return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0}

    precision_scores: List[float] = []
    for pred in pred_spans:
        pred_toks = tokenize(pred)
        if not pred_toks:
            precision_scores.append(0.0)
            continue

        best_ratio = 0.0
        for gt in gt_spans:
            lcs_len = lcs_length(pred, gt)
            ratio = lcs_len / len(pred_toks)
            best_ratio = max(best_ratio, ratio)
        precision_scores.append(best_ratio)

    P = float(sum(precision_scores) / len(pred_spans))

    recall_scores: List[float] = []
    for gt in gt_spans:
        gt_toks = tokenize(gt)
        if not gt_toks:
            recall_scores.append(0.0)
            continue

        best_ratio = 0.0
        for pred in pred_spans:
            lcs_len = lcs_length(pred, gt)
            ratio = lcs_len / len(gt_toks)
            best_ratio = max(best_ratio, ratio)
        recall_scores.append(best_ratio)

    R = float(sum(recall_scores) / len(gt_spans))
    F1 = float(2 * P * R / (P + R)) if (P + R) else 0.0

    return {"Precision": round(P, 4), "Recall": round(R, 4), "F1": round(F1, 4)}


# --------------------------- prompt + client ---------------------------

def build_user_prompt(article: str, technique: str) -> str:
    return f"""Article:
{article}

You are an expert propaganda span detector.
Your task is to carefully analyze the article and decide step by step (Chain of Thought) whether there are any statements that qualify as propaganda of type \"{technique}\".

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


def load_system_prompt(system_prompt_path: str) -> str:
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def call_vllm_chat(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    max_retries: int = 6,
) -> str:
    backoff_s = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_s,
            )
            return resp.choices[0].message.content or ""
        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            if attempt == max_retries:
                raise
            logger.warning("vLLM call failed (attempt %s/%s): %s; backing off %.1fs", attempt, max_retries, e, backoff_s)
            time.sleep(backoff_s)
            backoff_s = min(backoff_s * 2.0, 30.0)


# --------------------------- technique mapping ---------------------------

def normalize_tech_num(tech_num: str) -> str:
    s = str(tech_num).strip()
    # Accept forms like "2", "02", "T2", "T02".
    if s.lower().startswith("t"):
        s = s[1:]
    return s.zfill(2) if s.isdigit() else s


def _prettify_label(label: str) -> str:
    # Keep the technique label exactly as stored in the mapping CSV.
    return str(label).strip()


def lookup_technique(mapping_csv: str, tech_num_2d: str) -> Optional[str]:
    if not mapping_csv or not os.path.exists(mapping_csv):
        return None

    try:
        m = pd.read_csv(mapping_csv)
    except Exception as e:
        logger.warning("Failed reading mapping csv %s: %s", mapping_csv, e)
        return None

    # First, handle the repo's PTC mapping format: Tech_Number like "T02" and Label.
    if "Tech_Number" in m.columns and "Label" in m.columns:
        target = f"T{tech_num_2d}"
        hit = m.loc[m["Tech_Number"].astype(str).str.strip() == target]
        if not hit.empty:
            return _prettify_label(hit.iloc[0]["Label"])

    # Fallback: be tolerant to other column naming differences
    lower_cols = {c.lower(): c for c in m.columns}
    id_col = lower_cols.get("tech_num") or lower_cols.get("technique_num") or lower_cols.get("tech_number")
    name_col = lower_cols.get("technique") or lower_cols.get("technique_name") or lower_cols.get("label") or lower_cols.get("name")
    if not id_col or not name_col:
        return None

    def _norm(x: Any) -> str:
        xs = str(x).strip()
        if xs.lower().startswith("t"):
            xs = xs[1:]
        return xs.zfill(2) if xs.isdigit() else xs

    for _, row in m.iterrows():
        if _norm(row[id_col]) == tech_num_2d:
            return _prettify_label(row[name_col])
    return None


# --------------------------- main pipeline ---------------------------

def parse_gt_spans(gt_raw: Any) -> List[str]:
    if gt_raw is None or (isinstance(gt_raw, float) and np.isnan(gt_raw)):
        return []
    if isinstance(gt_raw, list):
        return [str(x) for x in gt_raw]
    if isinstance(gt_raw, str):
        s = gt_raw.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            return []
        except Exception:
            return []
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize CoT recipes using local vLLM (Mistral) for PTC dataset")
    parser.add_argument("--tech-num", required=True, help="Technique number, e.g. 2 or 02")
    parser.add_argument("--technique", default=None, help="Technique name (optional; otherwise read from mapping CSV)")
    parser.add_argument(
        "--mapping-csv",
        default="label_to_technique_mapping.csv",
        help="CSV mapping from technique id to name",
    )
    parser.add_argument(
        "--train-csv",
        default=None,
        help="Path to Train_Txx.csv (default: /Formatted_Data/Train/Train_T{tech_num}.csv)",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path (default: /Intialization/T{tech_num}_init_ZS.csv)",
    )
    parser.add_argument("--article-col", default="Article", help="Article text column name")
    parser.add_argument("--gt-col", default="validated_spans", help="Ground-truth spans column name")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep after each LLM call")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")

    parser.add_argument("--base-url", default="http://localhost:8001/v1", help="vLLM OpenAI-compatible base URL")
    parser.add_argument("--api-key", default="EMPTY", help="API key placeholder for vLLM")
    parser.add_argument("--temp", type=float, default=0.15, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=5000, help="Max tokens for completion")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds")
    parser.add_argument(
        "--system-prompt-path",
        default="/models/Mistral-Small-3.2-24B-Instruct-2506/SYSTEM_PROMPT.txt",
        help="Path to SYSTEM_PROMPT.txt for the model",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    tech_num = normalize_tech_num(args.tech_num)

    technique = args.technique
    if not technique:
        technique = lookup_technique(args.mapping_csv, tech_num)
    if not technique:
        raise SystemExit("Technique name not provided and could not be resolved from mapping CSV")

    train_csv = args.train_csv or f"/Formatted_Data/Train/Train_T{tech_num}.csv"
    out_csv = args.out_csv or f"/Intialization/T{tech_num}_init_ZS.csv"

    logger.info("Technique T%s = %s", tech_num, technique)
    logger.info("Reading train CSV: %s", train_csv)

    df = pd.read_csv(train_csv)
    if args.article_col not in df.columns:
        raise SystemExit(f"Missing article column '{args.article_col}'. Available columns: {list(df.columns)}")
    if args.gt_col not in df.columns:
        raise SystemExit(f"Missing GT column '{args.gt_col}'. Available columns: {list(df.columns)}")

    start = max(0, int(args.start))
    end = int(args.end) if args.end is not None else len(df)
    end = min(end, len(df))

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # Get model name from server
    models = client.models.list()
    if not models.data:
        raise SystemExit("No models returned by vLLM server")
    model = models.data[0].id
    logger.info("✅ Connected to vLLM server with model: %s", model)

    system_prompt = load_system_prompt(args.system_prompt_path)

    records: List[Dict[str, Any]] = []
    logger.info("🟢 Initializing CoT Recipes...")

    for idx in tqdm(range(start, end)):
        row = df.iloc[idx]

        gt = parse_gt_spans(row.get(args.gt_col))
        full_article = str(row.get(args.article_col, ""))

        user_prompt = build_user_prompt(article=full_article, technique=technique)
        try:
            text = call_vllm_chat(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=args.temp,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout,
            )
        except Exception as e:
            logger.exception("⚠️ Error at index %s: %s", idx, e)
            continue

        cot = extract_cot(text)
        pred = extract_predicted_spans(text)
        metrics = token_level_direct_f1(gt, pred)

        # Match the existing initialization CSV schema in Data/ptc/Intialization/*.csv
        rec: Dict[str, Any] = {
            "gt": json.dumps(gt, ensure_ascii=False),
            "pred": json.dumps(pred, ensure_ascii=False),
            "full_article": full_article,
            "CoT": json.dumps(cot, ensure_ascii=False),
            "precision": metrics["Precision"],
            "recall": metrics["Recall"],
            "f1": metrics["F1"],
            "recipe": text,
        }

        records.append(rec)

        if args.sleep and args.sleep > 0:
            time.sleep(float(args.sleep))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df = pd.DataFrame(records)
    out_df.to_csv(out_csv, index=False)

    logger.info("✅ Saved initialization CSV: %s", out_csv)
    if not out_df.empty:
        logger.info("Mean F1=%.4f | Precision=%.4f | Recall=%.4f", out_df["f1"].mean(), out_df["precision"].mean(), out_df["recall"].mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
