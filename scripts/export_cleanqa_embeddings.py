import argparse

import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_INPUT = "reference_qa_clean.csv"
DEFAULT_OUTPUT = "qa_embeddings.pkl"


def norm_text(x):
    return "" if pd.isna(x) else str(x).strip()


def build_input_text(row, question_col, answer_col, mode):
    question = norm_text(row.get(question_col, ""))
    answer = norm_text(row.get(answer_col, ""))

    if mode == "question":
        return question

    parts = [question]
    if answer:
        parts.append(answer)
    return "\n".join([p for p in parts if p])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--question-col", default="主旨")
    parser.add_argument("--answer-col", default="答覆內容")
    parser.add_argument("--mode", choices=["question", "qa"], default="question")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.question_col not in df.columns:
        raise ValueError(f"missing question column: {args.question_col}")

    df = df.copy()
    df["embedding_text"] = df.apply(
        lambda row: "passage: " + build_input_text(
            row,
            question_col=args.question_col,
            answer_col=args.answer_col,
            mode=args.mode,
        ),
        axis=1,
    )

    model = SentenceTransformer(args.model_name)
    vectors = model.encode(
        df["embedding_text"].tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=args.normalize,
    )

    out_df = df.copy()
    out_df["embedding"] = [vec.tolist() for vec in vectors]
    out_df.to_pickle(args.output_csv)
    print(f"saved embeddings PKL: {args.output_csv}, rows: {len(out_df)}")


if __name__ == "__main__":
    main()


