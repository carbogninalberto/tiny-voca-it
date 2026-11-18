"""Copy emotion/intensity from the synthetic source manifests into the annotation manifests (idempotent)."""
import json
for split in ("train", "validation", "test"):
    labels = {}
    for line in open(f"data/emotion-synth/{split}.jsonl"):
        row = json.loads(line); labels[row["id"]] = (row["emotion"], row["intensity"])
    path = f"data/emotion-annotations/{split}.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    for row in rows:
        row["emotion"], row["intensity"] = labels[row["id"]]
    with open(path, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(split, len(rows), "labelled")
