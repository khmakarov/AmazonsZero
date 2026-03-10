import orjson
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm

RAW_ROOT = r"/home/khmakarov/AmazonsZero/data/raw"
PROCESSED_ROOT = r"/home/khmakarov/AmazonsZero/data/processed"
BOTS = [
    "6210cc5dbbc22c318db9c486", "634d180d832a161fc4290335", "66fe9f70bae30059c86086d9", "66fef606bae30059c860b53c", "62219f66beec6e2acb80a51e", "657a8d5682ee46246bc7bc4e",
    "63a8f4321688b565e81bac1e", "65e83036e7439a290b65c934", "63a006c21688b565e8fca5ac", "6327cd9d66467507caed7e4b", "63a8f46c1688b565e81bad65", "63b77cefecf34307aedaf6e3",
    "6320776066467507cae65af1", "638aab3f8d38157ab2ef9bb7", "6327cb7366467507caed7c9c", "61dbc40f99f541427735727d", "63a29ada1688b565e804ebc2", "6327cd2366467507caed7ddc",
    "6327cb3a66467507caed7c76", "63a0ff1d1688b565e8fd9021", "634937f74172273132e033b0", "63bd0358ee1bce5e6c720ab6", "638deb238d38157ab2f268a6", "642eaabf6b5e994f52e98d8b",
    "6384a9998d38157ab2e0a13b", "5e024c8ab63f956037036e76", "6399af601688b565e8f5ed25", "63b77b06ecf34307aedaf56e", "6327ce8266467507caed7eba", "6327ce3466467507caed7e9d",
    "6327ceea66467507caed7eee", "6393ddf61688b565e8ee7ae8", "638b481b8d38157ab2f007d0", "632076fe66467507cae65a93", "64030e9b227405118d721bc0", "6327caf366467507caed7c32",
    "5fd2211297190c3037b99367", "631ed97d66467507cae47ec8", "6347f3234172273132de82dc", "63b4e753ecf34307aed6f87a"
]


def convert_file(file_path):
    """转换原始文件为JSON Lines格式"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        valid_objects = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                orjson.loads(line)  # 验证JSON合法性
                valid_objects.append(line)  # 保留原始行
            except orjson.JSONDecodeError:
                continue

        if not valid_objects:
            return None

        # 构建新路径（保持目录结构）
        relative_path = Path(file_path).relative_to(RAW_ROOT)
        new_path = Path(PROCESSED_ROOT) / relative_path.with_suffix('.jsonl')
        new_path.parent.mkdir(parents=True, exist_ok=True)

        # 写入JSON Lines格式
        with open(new_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(valid_objects))

        return str(new_path)

    except Exception as e:
        print(f"转换失败 {file_path}: {str(e)}")
        return None


def filter_errors(file_path):
    """过滤包含错误信息的JSON对象"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 过滤包含错误的行
        valid_lines = []
        for line in lines:
            try:
                obj = orjson.loads(line)
                if not any("err" in log.get("output", {}).get("display", {}) for log in obj.get("log", [])):
                    valid_lines.append(line.strip())
            except orjson.JSONDecodeError:
                continue

        if not valid_lines:
            os.remove(file_path)
            return None

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(valid_lines))

        return file_path

    except Exception as e:
        print(f"错误过滤失败 {file_path}: {str(e)}")
        return None


def clean_log_entry(entry):
    """清洗单个日志条目"""
    cleaned = entry.copy()

    for field in ['memory', 'time', 'verdict', 'debug', 'keep_running']:
        cleaned.pop(field, None)

    if 'output' in cleaned:
        output_data = cleaned['output']
        new_output = {}

        if 'display' in output_data:
            display = output_data['display']
            if 'blackCount' in display:
                new_output['blackCount'] = display['blackCount']
            if 'whiteCount' in display:
                new_output['whiteCount'] = display['whiteCount']

        if new_output:
            cleaned['output'] = new_output
        else:
            del cleaned['output']

    # 处理0/1字段（仅保留response）
    for player_key in ['0', '1']:
        if player_key in cleaned:
            player_data = cleaned[player_key]
            if 'response' in player_data:
                cleaned[player_key] = {'response': player_data['response']}
            else:
                del cleaned[player_key]

    return cleaned


def clean_file(file_path):
    """清洗整个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        processed_lines = []
        for line in lines:
            try:
                obj = orjson.loads(line)
                # 过滤非bot对局
                players = obj.get('players', [])
                if not all(p.get('type') == 'bot' for p in players):
                    continue
                bot_ids = [p.get('bot') for p in players]
                if not any(bot_id in BOTS for bot_id in bot_ids):
                    continue
                cleaned_log = []
                for entry in obj.get('log', []):
                    try:
                        cleaned_log.append(clean_log_entry(entry))
                    except Exception as e:
                        print(f"日志条目清洗失败: {e}")
                        continue

                new_obj = {"_id": obj.get("_id"), "scores": obj.get("scores"), "players": players, "log": cleaned_log}
                processed_lines.append(orjson.dumps(new_obj).decode('utf-8'))
            except orjson.JSONDecodeError:
                continue

        if not processed_lines:
            os.remove(file_path)
            return None

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(processed_lines))

        return file_path

    except Exception as e:
        print(f"文件清洗失败 {file_path}: {str(e)}")
        return None


def process_pipeline(max_workers=16):
    """完整的处理流水线"""
    # 阶段1：格式转换
    print("正在转换原始数据...")
    raw_files = [str(p) for p in Path(RAW_ROOT).rglob('*') if p.is_file() and p.suffix != '.jsonl']
    with ThreadPoolExecutor(max_workers) as executor:
        futures = [executor.submit(convert_file, f) for f in raw_files]
        converted_files = [f.result() for f in tqdm(as_completed(futures), total=len(futures))]

    # 阶段2：错误过滤
    print("\n正在过滤错误数据...")
    converted_files = [f for f in converted_files if f is not None]
    with ProcessPoolExecutor(max_workers) as executor:
        futures = [executor.submit(filter_errors, f) for f in converted_files]
        valid_files = [f.result() for f in tqdm(as_completed(futures), total=len(futures))]

    # 阶段3：日志清洗
    print("\n正在清洗日志数据...")
    valid_files = [f for f in valid_files if f is not None]
    with ProcessPoolExecutor(max_workers) as executor:
        futures = [executor.submit(clean_file, f) for f in valid_files]
        _ = [f.result() for f in tqdm(as_completed(futures), total=len(futures))]

    print("\n处理完成！结果保存在：", PROCESSED_ROOT)


if __name__ == "__main__":
    process_pipeline(max_workers=16)
