import os, sys, json, torch
from pathlib import Path
from xpm_torch.huggingface import TorchHFHub
from llm2ner import utils
from llm2ner.data import load_all_splits

CKPT = sys.argv[1] if len(sys.argv) > 1 else "llm2ner/ToMMeR-Llama-3.2-1B_L6_R64"
OUT  = Path(sys.argv[2] if len(sys.argv) > 2 else "results_table1.json")
DATASETS = sys.argv[3].split(",") if len(sys.argv) > 3 else [
    "MultiNERD", "CoNLL 2003", "CrossNER_politics", "CrossNER_AI", "CrossNER_literature",
    "CrossNER_science", "CrossNER_music", "ncbi", "FabNER", "WikiNeural",
    "GENIA_NER", "ACE 2005", "Ontonotes"]
DATA = Path(os.environ["NER_DATA"])

tommer = TorchHFHub.from_pretrained(CKPT)
tommer.initialize()
llm = utils.load_llm(tommer.llm_name, cut_to_layer=tommer.layer).eval()
tommer.to(llm.device)
print("dtype backbone:", next(llm.parameters()).dtype, flush=True)
c_length = utils.get_model_max_length(tommer.llm_name)

res = json.load(open(OUT)) if OUT.exists() else {}
with torch.no_grad():
    for name in DATASETS:
        if all(f"{s}/{name}" in res for s in ["threshold", "greedy"]):
            continue
        ds = load_all_splits(dataset_name=name, data_folder=DATA, model=llm, mode="last",
                             max_ent_length=400, max_length=c_length, verbose=True)
        for strat in ["threshold", "greedy"]:
            m = tommer.evaluate(ds.get_loader(batch_size=30), decoding_strategy=strat,
                                threshold=0.5, verbose=True)
            m["n_samples"] = len(ds)
            res[f"{strat}/{name}"] = m
            json.dump(res, open(OUT, "w"), indent=2, default=float)
            print(strat, name, {k: m[k] for k in m if k in ("precision","recall","f1","total","n_samples")}, flush=True)
        del ds
