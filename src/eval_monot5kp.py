from pyterrier_t5 import MonoT5ReRanker
import faiss
import pyterrier as pt
import json

from pyterrier.measures import *
from pyterrier.transformer import TransformerBase
from pyterrier_pisa import PisaIndex
from typing import List, Optional

pt.init()

# fmt: off
from pyterrier_colbert.ranking import ColBERTFactory
# fmt: on


def get_topics(dataset, keyphrases_json_path: Optional[str]):
    trec_topics = dataset.get_topics()
    query_type = "original_queries"

    if keyphrases_json_path:
        query_type = "auto_keyphrases"
        with open(keyphrases_json_path) as f:
            kps_json = json.load(f)
            for i, row in trec_topics.iterrows():
                trec_topics.at[i, "query"] = kps_json[row["qid"]]

    return trec_topics, query_type


def run_experiment(name: str, models: List[TransformerBase], models_names: List[str],  dataset_name: str, keyphrases_json_path: Optional[str] = None):
    print(f"Running experiments for: {name}")
    print()

    dataset = pt.get_dataset(dataset_name)
    trec_topics, query_type = get_topics(dataset, keyphrases_json_path)

    if "trec-dl-2019" in dataset_name:
        metrics = [AP(rel=2)@1000, nDCG@10, RR(rel=2)@10]
        ds_results_dir = "results-2019"
    elif "trec-dl-2020" in dataset_name:
        metrics = [AP(rel=2)@1000, nDCG@10, RR(rel=2)@10]
        ds_results_dir = "results-2020"
    elif "dev" in dataset_name:
        metrics = [RR@10, R@50, R@200, R@1000]
        ds_results_dir = "results-msmarco"
    else:
        raise Exception(
            f"Define the metrics and results directory for the {dataset_name} dataset")

    pipelines = []
    for model in models:
        pipelines.append(bm25 >> pt.text.get_text(
            dataset, "text") >> model)

    res = pt.Experiment(
        pipelines,
        trec_topics,
        dataset.get_qrels(),
        eval_metrics=metrics,
        batch_size=1024,
        drop_unused=True,
        names=models_names,
        round=4,
        save_dir=f"../resources/results/{query_type}/results-reranking/{ds_results_dir}"
    )

    print(res)
    print("="*100)
    print()


index = PisaIndex.from_dataset('msmarco_passage')
bm25 = index.bm25(num_results=1000)

# loads castorini/monot5-base-msmarco by default
monoT5 = MonoT5ReRanker(batch_size=1024)

monot5kp_path = "../resources/models/monot5kp"
monoT5KP = MonoT5ReRanker(model=monot5kp_path, batch_size=1024)

models_list = [monoT5, monoT5KP]


# TREC DL 2019 experiments
run_experiment("TREC2019 original", models_list, [
               "MonoT5", "MonoT5KP"], "irds:msmarco-passage/trec-dl-2019/judged")
run_experiment("TREC2019 mistral", models_list, ["MonoT5", "MonoT5KP"],
               "irds:msmarco-passage/trec-dl-2019/judged", "../resources/data/trec_2019_test_mistral_kps.json")


# TREC DL 2020 experiments
run_experiment("TREC2020 original", models_list, [
               "MonoT5", "MonoT5KP"], "irds:msmarco-passage/trec-dl-2020/judged")
run_experiment("TREC2020 mistral", models_list, ["MonoT5", "MonoT5KP"],
               "irds:msmarco-passage/trec-dl-2020/judged", "../resources/data/trec_2020_test_mistral_kps.json")
