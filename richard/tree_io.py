"""
Save / load a GP tree as plain JSON (no pickle, so it survives code changes and
can be read by anything). A tree is stored as nested dicts:

    function node : {"f": "if_lte", "c": [child, child, child, child]}
    feature leaf  : {"x": 17}          (column index into the feature vector)
    constant leaf : {"k": 0.4213...}   (JSON round-trips Python floats exactly)

The file also carries the feature names, so a tree trained on the derived
terminal set (264 indicators + 8 derived = 272 columns) is self-describing.
"""
import json

from richard.node import Node
from richard.terminals import Terminal


def tree_to_dict(node: Node) -> dict:
    if node.is_terminal:
        t = node.terminal
        return {"x": t.feature_index} if t.feature_index is not None else {"k": t.const_value}
    return {"f": node.func_name, "c": [tree_to_dict(c) for c in node.children]}


def tree_from_dict(d: dict) -> Node:
    if "f" in d:
        return Node(func_name=d["f"], children=[tree_from_dict(c) for c in d["c"]])
    if "x" in d:
        return Node(terminal=Terminal(feature_index=int(d["x"])))
    return Node(terminal=Terminal(const_value=float(d["k"])))


def save_tree(path: str, tree: Node, feature_names: list[str], meta: dict | None = None) -> None:
    payload = {"meta": meta or {}, "n_features": len(feature_names), "feature_names": feature_names,
               "size": tree.size(), "depth": tree.depth(), "expression": repr(tree),
               "tree": tree_to_dict(tree)}
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)


def load_tree(path: str) -> tuple[Node, dict]:
    with open(path) as f:
        payload = json.load(f)
    return tree_from_dict(payload["tree"]), payload