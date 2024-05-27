import argparse
import csv
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

TEMPLATE = """[INST]
You are an expert in transforming a query into keyphrase format. 

You must follow these rules:
1. When the query is a keyphrase you must just return the query and nothing else (examples of keyphrases are: information retrieval, solar panels, quantum computing, heart rate or blockchain).
2. Try not to include variations of the same keyphrase.
3. Acronyms are also considered keyphrases.
4. One-word keyphrases are allowed.
5. Do not provide several sets of keyphrases stick to the output format (do not add "Or: {{<Keyphrases>{{keyphrase 1}}, {{keyphrase 2}}</Keyphrases>}}")

The input format is:
Generate keyphrases for:
<Query>{{input query}}</Query>

The output format is:
<Keyphrases>{{keyphrase 1}}, {{keyphrase 2}}</Keyphrases>

Generate keyphrases for:
<Query>how many calories in jiffy natural peanut butter</Query>
[/INST]
<Keyphrases>jiffy natural peanut butter calories</Keyphrases>
</s>

<s>[INST]
Generate keyphrases for:
<Query>where do belgian draft horses originate</Query>
[/INST]
<Keyphrases>belgian draft horses origin</Keyphrases>
</s>

<s>[INST]
Generate keyphrases for:
<Query>who are the royal guards</Query>
[/INST]
<Keyphrases>royal guard</Keyphrases>
</s>

<s>[INST]
Generate keyphrases for:
<Query>{query}</Query>
[/INST]
<Keyphrases>"""

LLM = "mistralai/Mistral-7B-Instruct-v0.2"
HF_TOKEN = ""  # use your HF token here
DEVICE = "cuda"


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM, token=HF_TOKEN, quantization_config=quantization_config)

llm_tokenizer = AutoTokenizer.from_pretrained(LLM, token=HF_TOKEN)
llm_tokenizer.pad_token = llm_tokenizer.eos_token


def _generate_keyphrases(queries_batch):
    model_inputs = llm_tokenizer(
        queries_batch, return_tensors="pt", padding=True).to(DEVICE)

    output = llm_model.generate(**model_inputs, max_new_tokens=64,
                                do_sample=False, pad_token_id=llm_tokenizer.eos_token_id)

    return llm_tokenizer.batch_decode(output)


def _write_batch(tsv_writer, keyphrases, rel_docs, non_rel_docs):
    for r, pos, neg in zip(keyphrases, rel_docs, non_rel_docs):
        kps = r.split(
            "[/INST]")[-1].split("</Keyphrases>")[0].replace("<Keyphrases>", "").strip()

        if not "\n" in kps:
            tsv_writer.writerow([kps, pos, neg])


def generate_triples(input_path, output_path, batch_size):
    with open(input_path) as input_f, open(output_path, "w") as output_f:
        tsv_reader = csv.reader(input_f, delimiter="\t")
        tsv_writer = csv.writer(output_f, delimiter="\t")

        pos_batch = []
        neg_batch = []
        queries_batch = []
        for i, line in enumerate(tsv_reader):
            query = line[0].strip()
            rel_doc = line[1].strip()
            non_rel_doc = line[2].strip()

            prompt = TEMPLATE.format(
                query=query
            )

            if i != 0 and i % batch_size == 0:
                llm_output = _generate_keyphrases(queries_batch)
                _write_batch(tsv_writer, llm_output, pos_batch, neg_batch)

                # Clean batches
                queries_batch = []
                rels_batch = []
                non_rels_batch = []

            queries_batch.append(prompt)

            rels_batch.append(rel_doc)
            non_rels_batch.append(non_rel_doc)

        # Process remaining triplets
        if queries_batch:
            llm_output = _generate_keyphrases(queries_batch)
            _write_batch(tsv_writer, llm_output, pos_batch, neg_batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate MSMarco train tiples.")

    # Create arguments configuration
    parser.add_argument("--input", required=True, type=str,
                        help="Path to MSMarco train triples tsv")
    parser.add_argument("--output", type=str,
                        help="Path to save the new train tiples tsv")
    parser.add_argument("--batch_size", type=str,
                        help="LLM inference batch size")
    args = parser.parse_args()

    generate_triples(args.input, args.output, args.batch_size)
