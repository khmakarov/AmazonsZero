import orjson
from pathlib import Path
import hashlib
import logging
from tqdm import tqdm  # 进度条支持

# 配置日志和路径
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
PROCESS_DATA = Path("/home/khmakarov/AmazonsZero/data/pretrain")
OUTPUT_DIR = PROCESS_DATA.parent / "pretrain_global_deduplicated"


def normalize_json(data: dict) -> bytes:
    """标准化JSON并生成紧凑字节流"""
    return orjson.dumps(data, option=orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS)


def process_global_deduplication():
    seen = set()  # 全局去重集合
    total_files = 0
    total_duplicates = 0
    total_valid = 0

    # 创建输出目录结构
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 获取所有JSONL文件路径（带进度条）
    all_files = list(PROCESS_DATA.rglob("*.jsonl"))
    file_progress = tqdm(all_files, desc="Processing files", unit="file")

    for input_path in file_progress:
        # 构建输出路径
        relative_path = input_path.relative_to(PROCESS_DATA)
        output_path = OUTPUT_DIR / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
                for line in f_in:
                    try:
                        # 解析并标准化
                        json_data = orjson.loads(line)
                        normalized = normalize_json(json_data)

                        # 计算哈希指纹
                        fingerprint = hashlib.sha256(normalized).digest()

                        if fingerprint not in seen:
                            seen.add(fingerprint)
                            f_out.write(line)  # 保留原始行
                            total_valid += 1
                        else:
                            total_duplicates += 1

                    except orjson.JSONDecodeError:
                        logging.warning(f"Invalid JSON in {input_path}: {line[:100]}")

            total_files += 1
            file_progress.set_postfix({"unique": total_valid, "duplicates": total_duplicates})

        except Exception as e:
            logging.error(f"Failed to process {input_path}: {str(e)}")

    # 最终统计报告
    logging.info(f"\nProcessed {total_files} files")
    logging.info(f"Total unique records: {total_valid}")
    logging.info(f"Total duplicates found: {total_duplicates}")
    logging.info(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_global_deduplication()
