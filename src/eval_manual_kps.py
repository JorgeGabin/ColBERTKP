import warnings
import faiss
import pyterrier as pt
import pandas as pd
import pickle as pkl
import json

from pyterrier.measures import *
from pyterrier.transformer import TransformerBase
from pyterrier_pisa import PisaIndex
from typing import List, Optional

pt.init()

# fmt: off
from pyterrier_colbert.ranking import ColBERTFactory
# fmt: on


warnings.simplefilter(action='ignore', category=FutureWarning)


def transform_topics(trec_topics, keyphrases_json_paths: Optional[List[str]]):
    if keyphrases_json_paths:
        for i, row in trec_topics.iterrows():
            for path in keyphrases_json_paths:
                with open(path) as f:
                    kps_json = json.load(f)
                    if row["qid"] in kps_json:
                        trec_topics.at[i, "query"] = kps_json[row["qid"]]

    return trec_topics


dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
topics_2019 = dataset.get_topics()
qrels_2019 = dataset.get_qrels()


index_dir = "/nfs/indices/"
colbert_index_name = "msmarco"
colbert_kp_index_name = "msmarco_passage_index_colbertkp_from_cp_llm_kps"
setups = [
    ("ColBERT", "../resources/models/colbert-cosine-200k.dnn",
     index_dir, colbert_index_name),
    ("ColBERTKP", "../resources/models/colbertkp-cosine-25k.dnn",
     index_dir, colbert_kp_index_name),
    ("KPEncoder", "../resources/models/colbertkp_enc-cosine-25k.dnn",
     index_dir, colbert_index_name)
]

index = PisaIndex.from_dataset('msmarco_passage')
bm25 = index.bm25(num_results=1000)

models_list_rerank = []
models_list_e2e = [bm25]
for model_name, checkpoint, index_path, index_name in setups:
    print("Experiment Setup")
    print("-"*50)
    print(f"Name: {model_name}")
    print(f"Model: {checkpoint}")
    print(f"Index: {index_path}{index_name}")
    print("="*100)

    factory = ColBERTFactory(
        checkpoint,
        index_path,
        index_name,
        faiss_partitions=1,
        memtype='mem'
    )

    rerank = bm25 >> pt.text.get_text(dataset, "text") >> factory.text_scorer()
    dense_e2e = factory.end_to_end()

    models_list_rerank.append(rerank)
    models_list_e2e.append(dense_e2e)


trec_topics = pd.concat([topics_2019, topics_2019, topics_2019])
kps_jsons = ["../resources/data/trec_2019_test_assessor1.json",
             "../resources/data/trec_2019_test_assessor2.json",
             "../resources/data/trec_2019_test_assessor3.json"]
trec_topics = transform_topics(trec_topics, kps_jsons)
trec_qrels = qrels_2019


################### RERANKING ###################
res = pt.Experiment(
    models_list_rerank,
    trec_topics,
    trec_qrels,
    eval_metrics=[AP(rel=2)@1000, nDCG@10, RR(rel=2)@10],
    batch_size=1024,
    drop_unused=True,
    names=["ColBERT", "ColBERTKP", "KPEncoder"],
    round=4,
    save_dir="../resources/results/manual_keyphrases/results-reranking"
)

print(res.to_latex())
print("="*100)
print()


################### E2E ###################
res = pt.Experiment(
    models_list_e2e,
    trec_topics,
    trec_qrels,
    eval_metrics=[AP(rel=2)@1000, nDCG@10, RR(rel=2)@10],
    batch_size=1024,
    drop_unused=True,
    names=["BM25", "ColBERT", "ColBERTKP", "KPEncoder"],
    round=4,
    save_dir="../resources/results/manual_keyphrases/results-dense"
)

print(res.to_latex())
print("="*100)
print()
