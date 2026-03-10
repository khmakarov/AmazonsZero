import orjson
import os
import math
import random
from pathlib import Path
from tqdm import tqdm

PROCESSED_ROOT = r"/home/khmakarov/AmazonsZero/data/pretrain_global_deduplicated"
MERGE_ROOT = r"/home/khmakarov/AmazonsZero/data/merge"
CHUNK_SIZE = 256  # 每个合并文件包含的JSON对象数
GROUP_SIZE = 80  # 每个子文件夹包含的合并文件数


def merge_jsonl_files():
    all_objects = []

    # 收集所有JSONL文件路径
    jsonl_files = list(Path(PROCESSED_ROOT).rglob("*.jsonl"))
    print(f"发现 {len(jsonl_files)} 个.jsonl文件")

    # 读取并验证所有JSON对象
    for file_path in tqdm(jsonl_files, desc="读取和验证文件"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    # 验证JSON格式并重新序列化保证一致性
                    obj = orjson.loads(line)
                    all_objects.append(orjson.dumps(obj).decode('utf-8'))
                except orjson.JSONDecodeError:
                    continue

    # 随机打乱所有对象
    random.shuffle(all_objects)

    total_objects = len(all_objects)
    print(f"有效JSON对象总数：{total_objects}")

    # 计算分块信息
    total_chunks = math.ceil(total_objects / CHUNK_SIZE)
    chunks = [all_objects[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE] for i in range(total_chunks)]

    # 写入分块文件
    for chunk_idx, chunk in enumerate(tqdm(chunks, desc="写入合并文件")):
        global_file_num = chunk_idx + 1
        subfolder_num = (global_file_num - 1) // GROUP_SIZE + 1
        subfolder_path = Path(MERGE_ROOT) / f"Matches{subfolder_num}"
        subfolder_path.mkdir(parents=True, exist_ok=True)

        cnt = len(chunk)
        filename = f"matches{global_file_num}-{cnt}.jsonl"
        file_path = subfolder_path / filename

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(chunk))

    print(f"合并完成！结果保存在：{MERGE_ROOT}")


if __name__ == "__main__":
    merge_jsonl_files()
