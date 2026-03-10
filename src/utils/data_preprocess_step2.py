import orjson
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

PROCESSED_ROOT = r"/home/khmakarov/AmazonsZero/data/processed"


def remove_fields_from_file(file_path):
    """从单个文件中移除指定字段"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        modified_lines = []
        for line in lines:
            try:
                obj = orjson.loads(line)
                obj.pop('_id', None)
                obj.pop('players', None)

                if 'log' in obj:
                    for log_entry in obj['log']:
                        for player_key in ['0', '1']:
                            if player_key in log_entry and 'response' in log_entry[player_key]:
                                log_entry[player_key]['response'].pop('globalData', None)

                modified_lines.append(orjson.dumps(obj).decode('utf-8'))
            except orjson.JSONDecodeError:
                continue

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(modified_lines))

        return file_path

    except Exception as e:
        print(f"处理文件失败 {file_path}: {str(e)}")
        return None


def process_existing_files(max_workers=8):
    """处理现有结果文件"""
    file_paths = []
    for root, _, files in os.walk(PROCESSED_ROOT):
        for file in files:
            if file.endswith('.jsonl'):
                file_paths.append(os.path.join(root, file))

    print(f"发现 {len(file_paths)} 个待处理文件")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(remove_fields_from_file, f) for f in file_paths]
        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    print(f"成功处理 {len([r for r in results if r is not None])} 个文件")


if __name__ == "__main__":
    process_existing_files(max_workers=8)
