import json
import pyterrier as pt

from collections import defaultdict
from pyterrier.measures import *
from pyterrier_pisa import PisaIndex
from random import shuffle, randint


def filter_topics(trec_topics, qids, keyphrases_json_path=None):
    if keyphrases_json_path:
        with open(keyphrases_json_path) as f:
            kps_json = json.load(f)
            for i, row in trec_topics.iterrows():
                trec_topics.at[i, "query"] = kps_json[row["qid"]]

    if qids:
        trec_topics = trec_topics[trec_topics["qid"].isin(qids)]

    return trec_topics


index = PisaIndex.from_dataset('msmarco_passage')
bm25 = index.bm25(num_results=1000)

models = ["ColBERT", "ColBERTKP", "KPEncoder"]
models_reranking = ["BM25_ColBERT", "BM25_ColBERTKP", "BM25_KPEncoder"]

qids = ['87452', '1129237', '1113437', '182539', '1112341', '962179', '573724', '148538', '1117099', '1121402', '451602', '1114646', '915593', '47923', '1063750', '833860', '490595', '130510', '489204', '1037798', '87181',
        '1121709', '1133167', '156493', '359349', '1106007', '1114819', '146187', '131843', '405717', '1110199', '264014', '1124210', '183378', '1115776', '855410', '168216', '527433', '207786', '443396', '104861', '19335', '1103812']

final_dict = defaultdict(lambda: defaultdict(list))
final_dict_dense = defaultdict(lambda: defaultdict(list))
for i in range(100):
    shuffle(qids)

    res_dict = defaultdict(lambda: defaultdict(list))
    strategy = "reranking"
    # Original queries
    query_type = "original_queries"

    dataset_2019 = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')

    topics_2019 = dataset_2019.get_topics()

    qids_question_2019 = qids[:21]

    filt_topics_2019 = filter_topics(topics_2019, qids_question_2019)

    res = pt.Experiment(
        [bm25, bm25, bm25],
        filt_topics_2019,
        dataset_2019.get_qrels(),
        batch_size=1024,
        eval_metrics=[AP(rel=2)@1000, nDCG@10, RR(rel=2)
                      @ 10, "mrt"],  # type: ignore
        drop_unused=True,
        names=models,
        round=4,
        save_dir=f"resources/results/{
            query_type}/results-{strategy}/results-2019",
        perquery=True,
        save_mode="reuse"
    )

    for i, row in res.iterrows():
        res_dict[row["name"]][row["measure"]].append(row["value"])

    # Manual queries
    query_type = "manual_keyphrases"
    num_assessor = randint(1, 3)

    qids_kps_2019 = qids[21:]  # We exclude a random one so that they are 50/50

    topics_2019 = dataset_2019.get_topics()
    filt_topics_2019 = filter_topics(topics_2019, qids_kps_2019)

    res = pt.Experiment(
        [bm25, bm25, bm25],
        filt_topics_2019,
        dataset_2019.get_qrels(),
        batch_size=1024,
        eval_metrics=[AP(rel=2)@1000, nDCG@10, RR(rel=2)
                      @ 10, "mrt"],  # type: ignore
        drop_unused=True,
        names=models_reranking,
        round=4,
        save_dir=f"resources/results/{query_type}/assessor{num_assessor}/",
        perquery=True,
        save_mode="reuse"
    )

    for i, row in res.iterrows():
        res_dict[row["name"]][row["measure"]].append(row["value"])

    # Compute metrics
    for k, v in res_dict["ColBERT"].items():
        final_dict["ColBERT"][k].append(
            sum(res_dict["ColBERT"][k]) / len(res_dict["ColBERT"][k]))

    for k, v in res_dict["ColBERTKP"].items():
        final_dict["ColBERTKP"][k].append(
            sum(res_dict["ColBERTKP"][k]) / len(res_dict["ColBERTKP"][k]))

    for k, v in res_dict["KPEncoder"].items():
        final_dict["KPEncoder"][k].append(
            sum(res_dict["KPEncoder"][k]) / len(res_dict["KPEncoder"][k]))

    res_dict = defaultdict(lambda: defaultdict(list))
    strategy = "dense"
    # Original queries
    query_type = "original_queries"

    dataset_2019 = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')

    topics_2019 = dataset_2019.get_topics()

    qids_question_2019 = qids[:21]

    filt_topics_2019 = filter_topics(topics_2019, qids_question_2019)

    res = pt.Experiment(
        [bm25, bm25, bm25],
        filt_topics_2019,
        dataset_2019.get_qrels(),
        batch_size=1024,
        eval_metrics=[AP(rel=2)@1000, nDCG@10, RR(rel=2)
                      @ 10, "mrt"],  # type: ignore
        drop_unused=True,
        names=models,
        round=4,
        save_dir=f"resources/results/{
            query_type}/results-{strategy}/results-2019",
        perquery=True,
        save_mode="reuse"
    )

    for i, row in res.iterrows():
        res_dict[row["name"]][row["measure"]].append(row["value"])

    # Manual queries
    query_type = "manual_keyphrases"
    num_assessor = randint(1, 3)

    qids_kps_2019 = qids[21:]  # We exclude a random one so that they are 50/50

    topics_2019 = dataset_2019.get_topics()
    filt_topics_2019 = filter_topics(topics_2019, qids_kps_2019)

    res = pt.Experiment(
        [bm25, bm25, bm25],
        filt_topics_2019,
        dataset_2019.get_qrels(),
        batch_size=1024,
        eval_metrics=[AP(rel=2)@1000, nDCG@10, RR(rel=2)
                      @ 10, "mrt"],  # type: ignore
        drop_unused=True,
        names=models,
        round=4,
        save_dir=f"resources/results/{query_type}/assessor{num_assessor}/",
        perquery=True,
        save_mode="reuse"
    )

    for i, row in res.iterrows():
        res_dict[row["name"]][row["measure"]].append(row["value"])

    # Compute metrics
    for k, v in res_dict["ColBERT"].items():
        final_dict_dense["ColBERT"][k].append(
            sum(res_dict["ColBERT"][k]) / len(res_dict["ColBERT"][k]))

    for k, v in res_dict["ColBERTKP"].items():
        final_dict_dense["ColBERTKP"][k].append(
            sum(res_dict["ColBERTKP"][k]) / len(res_dict["ColBERTKP"][k]))

    for k, v in res_dict["KPEncoder"].items():
        final_dict_dense["KPEncoder"][k].append(
            sum(res_dict["KPEncoder"][k]) / len(res_dict["KPEncoder"][k]))


means_dict = defaultdict(lambda: defaultdict(float))
for k, v in final_dict["ColBERT"].items():
    means_dict["ColBERT"][k] = round(
        sum(final_dict["ColBERT"][k]) / len(final_dict["ColBERT"][k]), 4)

for k, v in final_dict["ColBERTKP"].items():
    means_dict["ColBERTKP"][k] = round(
        sum(final_dict["ColBERTKP"][k]) / len(final_dict["ColBERTKP"][k]), 4)

for k, v in final_dict["KPEncoder"].items():
    means_dict["KPEncoder"][k] = round(
        sum(final_dict["KPEncoder"][k]) / len(final_dict["KPEncoder"][k]), 4)

means_dict_dense = defaultdict(lambda: defaultdict(float))
for k, v in final_dict_dense["ColBERT"].items():
    means_dict_dense["ColBERT"][k] = round(
        sum(final_dict_dense["ColBERT"][k]) / len(final_dict_dense["ColBERT"][k]), 4)

for k, v in final_dict_dense["ColBERTKP"].items():
    means_dict_dense["ColBERTKP"][k] = round(
        sum(final_dict_dense["ColBERTKP"][k]) / len(final_dict_dense["ColBERTKP"][k]), 4)

for k, v in final_dict_dense["KPEncoder"].items():
    means_dict_dense["KPEncoder"][k] = round(
        sum(final_dict_dense["KPEncoder"][k]) / len(final_dict_dense["KPEncoder"][k]), 4)


print(means_dict)
print(means_dict_dense)
