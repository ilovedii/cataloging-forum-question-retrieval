import json
import csv

def convert_json_to_csv(json_filename, csv_filename):
    # 讀取 JSON 檔案
    try:
        with open(json_filename, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print(f"找不到檔案：{json_filename}，請確認檔案名稱與路徑。")
        return
    except json.JSONDecodeError:
        print("JSON 格式錯誤，請確認檔案內容是否為有效的 JSON。")
        return

    # 確認資料不為空
    if not data:
        print("資料為空，無法轉換。")
        return

    # 定義 CSV 的欄位名稱
    headers = [
        "input_serial",
        "input_question",
        "answer_ids_from_ground_truth",
        "answer_text",
        "embedding_rank_by_answer_id",
        "bm25_rank_by_answer_id",
        "best_overall_rank"
    ]

    # 寫入 CSV 檔案
    # 使用 'utf-8-sig' 可以確保 Excel 開啟時正確顯示中文，不會亂碼
    with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        
        # 寫入標題列
        writer.writeheader()
        
        # 遍歷 JSON 資料
        for item in data:
            input_serial = item.get("input_serial", "")
            input_question = item.get("input_question", "")
            answer_text = item.get("答案", "")
            answer_ids = item.get("對應序號", "")
            
            # 解析對應序號（可能是 "id1;id2" 或 "id1;id2;id3" 的格式）
            answer_ids_list = [str(id.strip()) for id in answer_ids.split(";") if id.strip()]
            
            # 從 dist 中提取 embedding_top50 和 bm25_top50
            dist = item.get("dist", {})
            embedding_top50 = dist.get("embedding_top50", [])
            bm25_top50 = dist.get("bm25_top50", [])
            
            # 建立序號到排名的映射
            embedding_rank_map = {}
            for rank_item in embedding_top50:
                seq_no = str(rank_item.get("序號", ""))
                rank = rank_item.get("rank", "")
                if seq_no:
                    embedding_rank_map[seq_no] = rank
            
            bm25_rank_map = {}
            for rank_item in bm25_top50:
                seq_no = str(rank_item.get("序號", ""))
                rank = rank_item.get("rank", "")
                if seq_no:
                    bm25_rank_map[seq_no] = rank
            
            # 建立 embedding_rank 和 bm25_rank 的配對列表
            embedding_pairs = []
            bm25_pairs = []
            best_ranks = []
            
            for answer_id in answer_ids_list:
                embedding_rank = embedding_rank_map.get(answer_id)
                bm25_rank = bm25_rank_map.get(answer_id)
                
                if embedding_rank:
                    embedding_pairs.append((embedding_rank, f"{answer_id}:{embedding_rank}"))
                    best_ranks.append(embedding_rank)
                if bm25_rank:
                    bm25_pairs.append((bm25_rank, f"{answer_id}:{bm25_rank}"))
                    best_ranks.append(bm25_rank)
            
            # 按照排名值從小到大排序，然後提取字符串部分
            embedding_pairs.sort(key=lambda x: x[0])
            bm25_pairs.sort(key=lambda x: x[0])
            
            embedding_rank_str_parts = [pair[1] for pair in embedding_pairs]
            bm25_rank_str_parts = [pair[1] for pair in bm25_pairs]
            
            # 確定最佳整體排名（取所有排名中的最小值）
            best_overall_rank = min(best_ranks) if best_ranks else ""
            
            # 每個問題只寫一行
            row_data = {
                "input_serial": input_serial,
                "input_question": input_question,
                "answer_ids_from_ground_truth": "; ".join(answer_ids_list),
                "answer_text": answer_text,
                "embedding_rank_by_answer_id": "; ".join(embedding_rank_str_parts),
                "bm25_rank_by_answer_id": "; ".join(bm25_rank_str_parts),
                "best_overall_rank": best_overall_rank
            }
            writer.writerow(row_data)
            
    print(f"轉換成功！檔案已儲存為：{csv_filename}")

# 執行轉換函數 (請根據您實際的檔案路徑與名稱進行修改)
convert_json_to_csv('./hybrid/分散問題qa_nopos.json', './hybrid/分散問題qa_nopos.csv')