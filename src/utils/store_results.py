import json
from typing import Any

def write_results(
    run_id: str,
    cls_loss_list: list[int],
    topk_acc: list[dict[str, float]],
    topk_acc_barch: dict[str, float],
    res_folder: str = "results"
    ):

    result_data: dict[str, Any] = {
        "run_id": run_id,
        "cls_loss_list": cls_loss_list,
        "topk_acc": topk_acc,
        "topk_acc_barchart": topk_acc_barch
    }

    with open(f"{res_folder}/results_{run_id}.json", "w") as f:
        json.dump(result_data, f, indent=4)
