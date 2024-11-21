import pyterrier as pt
import json

from collections import defaultdict
from pyterrier.measures import *
from pyterrier_pisa import PisaIndex
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

from pyterrier_colbert.ranking import ColBERTFactory


def transform_topics(trec_topics, keyphrases_json_paths):
    if keyphrases_json_paths:
        for path in keyphrases_json_paths:
            with open(path) as f:
                kps_json = json.load(f)
                for i, row in trec_topics.iterrows():
                    if row["qid"] in kps_json:
                        trec_topics.at[i, "query"] = kps_json[row["qid"]].split(",")[
                            0].strip()

    return trec_topics


index = PisaIndex.from_dataset('msmarco_passage')
bm25 = index.bm25(num_results=1000)

dataset_2019 = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
topics_2019 = dataset_2019.get_topics()

checkpoint = "resources/models/colbert-cosine-200k.dnn"
checkpoint_kp = "resources/models/colbertkp-cosine-25k.dnn"
checkpoint_kp_enc = "resources/models/colbertkp_enc-cosine-25k.dnn"

index_path = "/nfs/indices/"
index_name = "msmarco"
index_name_kp = "msmarco_passage_index_colbertkp_from_cp_llm_kps"

factory = ColBERTFactory(
    checkpoint,
    index_path,
    index_name, faiss_partitions=100, memtype='mem',
)

factory_kp = ColBERTFactory(
    checkpoint_kp,
    index_path,
    index_name_kp, faiss_partitions=100, memtype='mem',
)

factory_kp_enc = ColBERTFactory(
    checkpoint_kp_enc,
    index_path,
    index_name, faiss_partitions=100, memtype='mem',
)

reranking_colbert = bm25 >> pt.text.get_text(
    dataset_2019, "text") >> factory.text_scorer()
reranking_colbertkp = bm25 >> pt.text.get_text(
    dataset_2019, "text") >> factory_kp.text_scorer()
reranking_colbertkp_enc = bm25 >> pt.text.get_text(
    dataset_2019, "text") >> factory_kp_enc.text_scorer()

res_dict = defaultdict(lambda: defaultdict(list))

for num in range(1, 4):
    topics_2019 = dataset_2019.get_topics()
    topics_transf = transform_topics(topics_2019, [
                                     f"resources/data/trec_2019_test_assessor{num}.json"])
    res = pt.Experiment(
        [bm25, bm25, bm25],
        topics_transf,
        dataset_2019.get_qrels(),
        batch_size=1024,
        eval_metrics=[AP(rel=2)@1000, nDCG@10, RR(rel=2)@10],  # type: ignore
        drop_unused=True,
        names=["BM25_ColBERT", "BM25_ColBERTKP", "BM25_ColBERTKPEnc"],
        round=4,
        perquery=True,
        save_dir=f"resources/results/manual_keyphrases/assessor{num}",
        save_mode="reuse"
    )

    for i, row in res.iterrows():
        res_dict[row["name"]][row["measure"]].append(row["value"])

for method, _ in res_dict.items():
    print(method)
    for metric, v in res_dict[method].items():
        print(metric, sum(v) / len(v))
    print()

for k in res_dict["BM25_ColBERT"].keys():
    print(k)
    pv1 = ttest_rel(res_dict["BM25_ColBERT"][k], res_dict["BM25_ColBERTKP"][k])
    pv2 = ttest_rel(res_dict["BM25_ColBERT"][k],
                    res_dict["BM25_ColBERTKPEnc"][k])

    print(multipletests([pv1, pv2], method="holm"))
