# coding=utf-8
# Shared helpers for the Local Loss Refit scripts (calibration corpus
# streaming + git provenance). Mirrors scripts/oracle/01_calibrate.py's
# build_calib_tokens so calibration data is reproducible the same way.

import subprocess

import torch
from datasets import load_dataset


def git_commit(repo_dir):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir).decode().strip()
    except Exception:
        return "unknown"


def stream_corpus_ids(dataset_name, tokenizer, need, seed):
    """Yields token ids from a general corpus until at least `need` ids have
    been produced. dataset_name: 'c4' (streaming) or 'wikitext103' (shuffled
    by seed)."""
    buf = []
    if dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        for ex in ds:
            buf.extend(tokenizer(ex["text"]).input_ids)
            if len(buf) >= need:
                break
    elif dataset_name == "wikitext103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(ds), generator=g).tolist()
        for k in order:
            text = ds[k]["text"]
            if not text.strip():
                continue
            buf.extend(tokenizer(text).input_ids)
            if len(buf) >= need:
                break
    else:
        raise ValueError(dataset_name)
    return buf


def build_calib_tokens(dataset_name, tokenizer, nsamples, seqlen, seed):
    """Deterministic [nsamples, seqlen] token tensor from a general corpus."""
    from inference import refit_mlp
    torch.manual_seed(seed)
    buf = stream_corpus_ids(dataset_name, tokenizer, nsamples * seqlen, seed)
    return refit_mlp.reshape_calib_tokens(buf, nsamples, seqlen)
