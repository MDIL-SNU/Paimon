import sys
import json
from pathlib import Path
import os
import os.path as osp

print("python ... {path to wd}")

root = Path(sys.argv[1])

keys = [
    "total_input_tokens",
    "total_reasoning_tokens",
    "total_output_tokens",
    "total_cost",
]

with open(osp.join(root, ".token.json")) as f:
    planner_tok = json.load(f)

agent_tok = {}
for ls in os.listdir(root):
    if osp.exists(root / ls / ".agent_tokens.json"):
        with open(root / ls / ".agent_tokens.json") as f:
            agent_tok[ls] = json.load(f)

for at in (root / ".trash_bin").rglob(".agent_tokens.json"):
    with open(at) as f:
        agent_tok[str(at.parent)[len(".trash_bin"):]] = json.load(f)

print("Planner LLM:")
print(planner_tok["items"][0]["llm_model"])
print("Executor LLM:")
print(list(agent_tok.values())[0]["items"][0]["llm_model"])

dct = {f"planner_{k}": planner_tok[k] for k in keys}

dct_a = {
    f"subagent_{k}": sum([at[k] for at in list(agent_tok.values())]) for k in keys
}

dct.update(dct_a)
dct["total_cost"] = dct["planner_total_cost"] + dct["subagent_total_cost"]

print(f"{root / "total.json"} is written.")

with open(root / "total.json", "w") as f:
    json.dump(dct, f, indent=4)
