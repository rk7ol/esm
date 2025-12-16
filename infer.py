import os
from pathlib import Path
import argparse

import torch
import pandas as pd
from scipy.stats import spearmanr

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.pretrained import set_local_weights_dir

from huggingface_hub import login


############################################
#  路径约定（SageMaker Processing Job）
############################################

CODE_DIR = Path("/opt/ml/processing/input/code")
DATA_DIR = Path("/opt/ml/processing/input/data")
MODEL_DIR = Path("/opt/ml/processing/input/model")
OUTPUT_DIR = Path("/opt/ml/processing/output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "ADRB2_HUMAN_Jones_2020.csv"
OUTPUT_CSV = OUTPUT_DIR / "ADRB2_HUMAN_Jones_2020_esm3_zeroshot.csv"


############################################
#  加载 ESM-3 模型（本地权重）
############################################
parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_token",
    default=None,
    help="Hugging Face token (or set env HF_TOKEN / HUGGINGFACE_HUB_TOKEN).",
)
args = parser.parse_args()
hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token, add_to_git_credential=False)
else:
    print("Warning: no hf_token provided; skipping Hugging Face login.")

print(f"Loading ESM-3 model from {MODEL_DIR}")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

set_local_weights_dir(MODEL_DIR)
model = ESM3.from_pretrained("esm3-open-local").to(device).eval()
tok = model.tokenizers.sequence


############################################
#  工具函数
############################################

def parse_mutant(mut_str: str):
    """
    F1I -> wt=F, pos=1, mut=I
    """
    wt = mut_str[0]
    mut = mut_str[-1]
    pos = int(mut_str[1:-1])
    return wt, pos, mut


def recover_wt_sequence(mut_seq: str, wt_aa: str, pos1: int):
    """
    从 mutated_sequence 恢复 WT sequence
    """
    seq = list(mut_seq)
    seq[pos1 - 1] = wt_aa
    return "".join(seq)


@torch.no_grad()
def esm3_zero_shot_delta(model, wt_seq, pos1, mut_aa):
    """
    标准 ESM zero-shot ΔlogP
    """
    masked_seq = wt_seq[:pos1 - 1] + "_" + wt_seq[pos1:]

    pt = model.encode(ESMProtein(sequence=masked_seq))
    out = model.logits(pt, LogitsConfig(sequence=True))

    logits = out.logits.sequence[0, pos1, :]  # index 0 是 <cls>
    log_probs = torch.log_softmax(logits, dim=-1)

    wt_aa = wt_seq[pos1 - 1]
    wt_id = tok.encode(wt_aa, add_special_tokens=False)[0]
    mut_id = tok.encode(mut_aa, add_special_tokens=False)[0]

    return (log_probs[mut_id] - log_probs[wt_id]).item()


############################################
#  读取 CSV
############################################

print(f"Reading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

required_cols = {"mutant", "mutated_sequence", "DMS_score"}
assert required_cols.issubset(df.columns)

print(f"Loaded {len(df)} variants.")


############################################
#  批量 zero-shot
############################################

pred_scores = []
true_scores = []

print("Running zero-shot predictions...")

process_count = 0
for i, row in df.iterrows():
    mut_str = row["mutant"]
    mut_seq = row["mutated_sequence"]
    dms = row["DMS_score"]

    wt_aa, pos1, mut_aa = parse_mutant(mut_str)

    wt_seq = recover_wt_sequence(
        mut_seq=mut_seq,
        wt_aa=wt_aa,
        pos1=pos1,
    )

    if mut_seq[pos1 - 1] != mut_aa:
        raise ValueError(
            f"Mut AA mismatch in {mut_str}: "
            f"sequence has {mut_seq[pos1-1]}"
        )

    delta = esm3_zero_shot_delta(
        model=model,
        wt_seq=wt_seq,
        pos1=pos1,
        mut_aa=mut_aa,
    )

    pred_scores.append(delta)
    true_scores.append(dms)

    process_count += 1
    if process_count % 100 == 0:
        print(f"  processed {process_count}/{len(df)}")

print("Prediction done.")


############################################
#  评估 + 保存结果
############################################

rho, pval = spearmanr(pred_scores, true_scores)

df["esm3_delta_logp"] = pred_scores

df.to_csv(OUTPUT_CSV, index=False)

print("\n========== ProteinGym ESM-3 zero-shot ==========")
print(f"Variants:     {len(df)}")
print(f"Spearman ρ:   {rho:.4f}")
print(f"P-value:      {pval:.2e}")
print(f"Saved to:     {OUTPUT_CSV}")
print("================================================")
