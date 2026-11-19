import argparse

import torch
from transformers import AutoModel, AutoTokenizer

from vector_search import Index

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
PRODUCTS = [
    "High-top sneakers with ankle support and traction for indoor courts.",
    "Lightweight running shoes for long road races.",
    "Waterproof hiking boots for rocky trails.",
    "A burr grinder for freshly ground coffee.",
    "A gooseneck kettle for brewing pour-over coffee.",
    "Noise-cancelling wireless headphones for travel.",
    "A compact mechanical keyboard with quiet switches.",
    "A portable Bluetooth speaker for outdoor music.",
    "A two-person backpacking tent for rainy nights.",
    "A down sleeping bag for cold-weather camping.",
    "An official-size basketball for outdoor pickup games.",
    "Cushioned basketball footwear for quick cuts on the court.",
]


@torch.inference_mode()
def embed(texts, tokenizer, model):
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    hidden = model(**tokens).last_hidden_state
    mask = tokens.attention_mask.unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Search a small product catalog by meaning")
    parser.add_argument("queries", nargs="*", default=["basketball shoes"])
    parser.add_argument("--model-dir", help="Use a local copy of the MiniLM model")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()
    source = args.model_dir or MODEL
    options = {"local_files_only": True} if args.model_dir else {"revision": REVISION}
    tokenizer = AutoTokenizer.from_pretrained(source, **options)
    model = AutoModel.from_pretrained(source, **options).eval()
    index = Index(embed(PRODUCTS, tokenizer, model))
    ids, distances = index.search(embed(args.queries, tokenizer, model), k=args.k)
    for query, matches, scores in zip(args.queries, ids, distances):
        print(query)
        for row, squared_l2 in zip(matches, scores):
            print(f"  {row:2d}  {squared_l2:.4f}  {PRODUCTS[row]}")


if __name__ == "__main__":
    main()
