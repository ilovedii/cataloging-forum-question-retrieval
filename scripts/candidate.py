import argparse
import json
import os
import pickle
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
from rank_bm25 import BM25Okapi

MODEL_NAME = "intfloat/multilingual-e5-large"
QUESTION_COL = "主旨"
ANSWER_COL = "答覆內容"
INDEX_PATH = "qa_embeddings.pkl"
ALLOWED_POS = {"Na", "Nb", "Nc", "Ncd", "FW", "Nv", "VC"}
STOPWORDS = {
    # 禮貌稱呼
    "老師", "您好", "老師您好", "老師好",

    # 提問框架
    "請問", "想請教", "請教", "想了解", "想了解一下",
    "想問", "想要",
    "想", "了解", "問",

    # 禮貌結尾
    "謝謝", "感謝", "感謝您", "麻煩", "再麻煩",
    "敬請", "協助", "指導", "解答", "回答",

    # 泛用語
    "請", "一下", "另外", "以上",
}

PUNCT = "，。！？；：、,.!?;:()（）[]【】\"'“”"


def norm_text(x: Any) -> str:
    """把空值轉成空字串，其餘轉成去頭尾空白的字串。"""
    return "" if pd.isna(x) else str(x).strip()

def clean_token(word: str) -> str:
    """
    清理 BM25 token：
    1. 去掉前後空白
    2. 去掉前後標點
    3. 英文字母轉小寫，避免 MARCEDIT / marcedit 被視為不同詞
    """
    word = word.strip()
    word = word.strip(PUNCT)
    word = word.lower()
    return word


def to_native(obj: Any) -> Any:
    """把 numpy / pandas 型別轉成 JSON 可序列化的 Python 原生型別。"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is pd.NA:
        return None
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    return obj


def load_reference(reference_csv: str, question_col: str, answer_col: str) -> pd.DataFrame:
    """讀 reference QA CSV，保留有主旨的列。"""
    df = pd.read_csv(reference_csv)
    if question_col not in df.columns:
        raise ValueError(f"missing question column in reference csv: {question_col}")

    df = df.dropna(subset=[question_col]).copy()
    df[question_col] = df[question_col].map(norm_text)
    if answer_col in df.columns:
        df[answer_col] = df[answer_col].map(norm_text)

    return df[df[question_col] != ""].reset_index(drop=True)


def build_embedding_index(
    reference_csv: str,
    index_path: str,
    question_col: str,
    answer_col: str,
    model_name: str,
    batch_size: int,
) -> None:
    """建立 embedding index，格式為 {'df': df, 'embeddings': embeddings}。"""
    df = load_reference(reference_csv, question_col, answer_col)
    model = SentenceTransformer(model_name)

    passage_texts = ["passage: " + q for q in df[question_col].tolist()]
    embeddings = model.encode(
        passage_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    with open(index_path, "wb") as f:
        pickle.dump({"df": df, "embeddings": embeddings}, f)

    print(f"embedding index saved: {index_path}, rows: {len(df)}")


def load_embedding_index(index_path: str) -> Dict[str, Any]:
    """
    讀 embedding index。

    支援兩種格式：
    1. brute.py 產生的 dict：{'df': df, 'embeddings': embeddings}
    2. export_cleanqa_embeddings.py 產生的 DataFrame PKL，且含 embedding 欄位
    """
    with open(index_path, "rb") as f:
        index_obj = pickle.load(f)

    if isinstance(index_obj, dict) and "df" in index_obj and "embeddings" in index_obj:
        return {
            "df": index_obj["df"].reset_index(drop=True),
            "embeddings": np.asarray(index_obj["embeddings"], dtype="float32"),
        }

    if isinstance(index_obj, pd.DataFrame):
        if "embedding" not in index_obj.columns:
            raise ValueError("precomputed DataFrame PKL missing required column: embedding")

        def parse_vector(x: Any) -> np.ndarray:
            if isinstance(x, list):
                return np.asarray(x, dtype="float32")
            if isinstance(x, str):
                return np.asarray(json.loads(x), dtype="float32")
            return np.asarray(x, dtype="float32")

        embeddings = np.vstack(index_obj["embedding"].map(parse_vector).to_list()).astype("float32")
        return {"df": index_obj.reset_index(drop=True), "embeddings": embeddings}

    raise ValueError("unsupported index format; expected dict or DataFrame PKL")


def row_to_record(
    row: pd.Series,
    rank: int,
    score_name: str,
    score: float,
    question_col: str,
    answer_col: str,
) -> Dict[str, Any]:
    """把檢索到的 reference row 包成統一輸出格式。"""
    return {
        "rank": rank,
        score_name: float(score),
        "序號": row.get("序號", ""),
        "serial": row.get("serial", ""),
        "主旨": row.get(question_col, ""),
        "答覆內容": row.get(answer_col, ""),
    }


def embedding_search(
    query: str,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int,
    question_col: str,
    answer_col: str,
) -> List[Dict[str, Any]]:
    """用 E5 embedding 做 brute-force cosine similarity top-k。"""
    q_emb = model.encode(
        ["query: " + norm_text(query)],
        normalize_embeddings=True,
    ).astype("float32")[0]

    # embeddings 與 q_emb 都已 normalize，所以 dot product 等價於 cosine similarity。
    similarities = embeddings @ q_emb
    top_ids = np.argsort(similarities)[::-1][:top_k]

    return [
        row_to_record(df.iloc[row_id], rank, "similarity", similarities[row_id], question_col, answer_col)
        for rank, row_id in enumerate(top_ids, start=1)
    ]


def ckip_tokenize(
    texts: List[str],
    ws_driver: CkipWordSegmenter,
    pos_driver: Optional[CkipPosTagger],
    batch_size: int,
    use_pos_filter: bool = True,
) -> List[List[str]]:
    """CKIP 斷詞；預設保留指定詞性，對 BM25 較乾淨。"""
    ws_results = ws_driver(texts, batch_size=batch_size)

    if not use_pos_filter:
        tokenized = []

        for sentence in ws_results:
            tokens = []

            for word in sentence:
                word = clean_token(word)

                if not word:
                    continue

                if word in STOPWORDS:
                    continue

                tokens.append(word)

            tokenized.append(tokens)

        return tokenized

    if pos_driver is None:
        raise ValueError("pos_driver is required when use_pos_filter=True")
    
    pos_results = pos_driver(ws_results, batch_size=batch_size)
    tokenized = []

    for ws_sentence, pos_sentence in zip(ws_results, pos_results):
        tokens = [
            word
            for word, pos in zip(ws_sentence, pos_sentence)
            if pos in ALLOWED_POS and len(word) > 1
        ]
        tokenized.append(tokens)

    return tokenized


def bm25_search(
    query_tokens: List[str],
    df: pd.DataFrame,
    bm25: BM25Okapi,
    top_k: int,
    question_col: str,
    answer_col: str,
) -> List[Dict[str, Any]]:
    """用 rank-bm25 套件計算 BM25 top-k。"""
    scores = bm25.get_scores(query_tokens)
    top_ids = np.argsort(scores)[::-1][:top_k]

    return [
        row_to_record(df.iloc[row_id], rank, "bm25_score", scores[row_id], question_col, answer_col)
        for rank, row_id in enumerate(top_ids, start=1)
    ]

def build_dist_for_queries(
    queries: List[Tuple[str, str, Dict[str, Any]]],
    df: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    bm25: BM25Okapi,
    ws_driver: CkipWordSegmenter,
    pos_driver: CkipPosTagger,
    top_k: int,
    question_col: str,
    answer_col: str,
    ckip_batch_size: int,
    use_pos_filter: bool = True,
) -> List[Dict[str, Any]]:
   
    query_texts = [query for _, query, _ in queries]

    query_tokens_list = ckip_tokenize(
        query_texts,
        ws_driver=ws_driver,
        pos_driver=pos_driver,
        batch_size=ckip_batch_size,
        use_pos_filter=use_pos_filter,
    )

    outputs = []

    for (input_serial, query, input_meta), query_tokens in zip(queries, query_tokens_list):
        dist = {
            "embedding_top50": embedding_search(
                query=query,
                df=df,
                embeddings=embeddings,
                model=embedding_model,
                top_k=top_k,
                question_col=question_col,
                answer_col=answer_col,
            ),
            "bm25_top50": bm25_search(
                query_tokens=query_tokens,
                df=df,
                bm25=bm25,
                top_k=top_k,
                question_col=question_col,
                answer_col=answer_col,
            ),
        }

        output_row = {
            "input_serial": input_serial,
            "input_question": query,
            "query_tokens": query_tokens,
            "dist": dist,
        }

        output_row.update(input_meta)
        outputs.append(output_row)

    return outputs

def collect_queries(args: argparse.Namespace) -> List[Tuple[str, str, Dict[str, Any]]]:
    if args.query:
        return [("", norm_text(args.query), {})]

    if args.input_csv:
        input_df = pd.read_csv(args.input_csv)

        if args.input_question_col not in input_df.columns:
            raise ValueError(f"missing input question column: {args.input_question_col}")

        queries = []

        for _, row in input_df.iterrows():
            query = norm_text(row[args.input_question_col])

            if not query:
                continue

            input_meta = {}
            for col in args.keep_input_cols:
                if col in input_df.columns:
                    input_meta[col] = norm_text(row.get(col, ""))

            queries.append((norm_text(row.get("serial", "")), query, input_meta))

        return queries

    raise ValueError("please provide --query or --input-csv")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-csv", default="reference_qa_clean.csv")
    parser.add_argument("--index-path", default=INDEX_PATH)
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--question-col", default=QUESTION_COL)
    parser.add_argument("--answer-col", default=ANSWER_COL)
    parser.add_argument("--input-csv", default="")
    parser.add_argument("--input-question-col", default="分散問題")
    parser.add_argument("--keep-input-cols", nargs="*", default=["答案", "對應序號"])
    parser.add_argument("--query", default="")
    parser.add_argument("--output-json", default="dist_result.json")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--ckip-batch-size", type=int, default=64)
    parser.add_argument("--ckip-model", default="albert-base")
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0 for first GPU")
    parser.add_argument(
        "--no-pos-filter",
        action="store_true",
        help="不要使用 CKIP POS 篩選，BM25 直接使用所有斷詞結果",
    )
    args = parser.parse_args()

    if args.build_index or not os.path.exists(args.index_path):
        build_embedding_index(
            reference_csv=args.reference_csv,
            index_path=args.index_path,
            question_col=args.question_col,
            answer_col=args.answer_col,
            model_name=args.model_name,
            batch_size=args.embedding_batch_size,
        )

    index = load_embedding_index(args.index_path)
    df = index["df"]
    embeddings = index["embeddings"]

    if args.question_col not in df.columns:
        raise ValueError(f"missing question column in index df: {args.question_col}")

    print("loading embedding model...")
    embedding_model = SentenceTransformer(args.model_name)

    print("loading CKIP word segmenter...")
    ws_driver = CkipWordSegmenter(model=args.ckip_model, device=args.device)
    if args.no_pos_filter:
        pos_driver = None
    else:
        print("loading CKIP POS tagger...")
        pos_driver = CkipPosTagger(model=args.ckip_model, device=args.device)

    print("building rank-bm25 corpus from answer column...")
    if args.answer_col not in df.columns:
        raise ValueError(f"missing answer column in index df: {args.answer_col}")

    # corpus_texts = df[args.answer_col].map(norm_text).tolist()
    corpus_texts = df[args.question_col].map(norm_text).tolist()
    corpus_tokens = ckip_tokenize(
        corpus_texts,
        ws_driver=ws_driver,
        pos_driver=pos_driver,
        batch_size=args.ckip_batch_size,
        use_pos_filter=not args.no_pos_filter,
    )
    bm25 = BM25Okapi(corpus_tokens)

    queries = collect_queries(args)
    outputs = build_dist_for_queries(
        queries=queries,
        df=df,
        embeddings=embeddings,
        embedding_model=embedding_model,
        bm25=bm25,
        ws_driver=ws_driver,
        pos_driver=pos_driver,
        top_k=args.top_k,
        question_col=args.question_col,
        answer_col=args.answer_col,
        ckip_batch_size=args.ckip_batch_size,
        use_pos_filter=not args.no_pos_filter,
    )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(to_native(outputs), f, ensure_ascii=False, indent=2)

    print(f"result saved: {args.output_json}")


if __name__ == "__main__":
    main()
