import json, sys, glob, os
DS = ["MultiNERD","CoNLL 2003","CrossNER_politics","CrossNER_AI","CrossNER_literature","CrossNER_science",
      "CrossNER_music","ncbi","FabNER","WikiNeural","GENIA_NER","ACE 2005","Ontonotes"]
REF = {"threshold": (92.6, 23.2, 37.1), "greedy": (84.0, 31.2, 45.5)}   # Llama-3.2-1B L6, Aggregated
files = sys.argv[1:] or sorted(glob.glob(os.path.expandvars("$WORK/results/*.json")))
for f in files:
    r = json.load(open(f)); name = os.path.basename(f)[:-5]
    print(f"\n=== {name} ===")
    for s in ["threshold", "greedy"]:
        TP = FP = TOT = 0; P = []; R = []; F = []; done = 0
        for d in DS:
            m = r.get(f"{s}/{d}")
            if not m: continue
            done += 1
            tp = m.get("true_pos", m["recall"] * m["total"])
            fp = m.get("false_pos", tp / m["precision"] - tp if m["precision"] > 0 else 0)
            TP += tp; FP += fp; TOT += m["total"]
            P.append(m["precision"]); R.append(m["recall"]); F.append(m["f1"])
            print(f"  {s:9s} {d:20s} R {100*m['recall']:5.1f}  P {100*m['precision']:5.1f}  F1 {100*m['f1']:5.1f}  n={m.get('n_samples')}")
        if not done: continue
        p, rc = TP / (TP + FP), TP / TOT; f1 = 2 * p * rc / (p + rc)
        print(f"  {s:9s} {'AGGREGATED':20s} R {100*rc:5.1f}  P {100*p:5.1f}  F1 {100*f1:5.1f}   ({done}/13 dataset)")
        print(f"  {s:9s} {'AVERAGED':20s} R {100*sum(R)/done:5.1f}  P {100*sum(P)/done:5.1f}  F1 {100*sum(F)/done:5.1f}")
        if "Llama-3.2-1B_L6" in name:
            ref = REF[s]; diff = [abs(a - b) for a, b in zip((100*rc, 100*p, 100*f1), ref)]
            print(f"  {s:9s} vs paper {ref} -> selisih maks {max(diff):.1f} poin -> {'LOLOS' if max(diff) <= 5 else 'TIDAK LOLOS'}")
