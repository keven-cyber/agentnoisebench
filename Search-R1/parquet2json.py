# 将.parquet转化成.json格式

import pandas as pd
import json
import sys
import os
import numpy as np

def parquet_to_json(parquet_file, json_file):
    # 读取parquet文件
    df = pd.read_parquet(parquet_file)

    # 将DataFrame转换为字典列表
    data = df.to_dict(orient='records')

    # 将 numpy 类型（ndarray, np.generic 等）转换为原生 Python 类型，便于 json 序列化
    def _convert(obj):
        # numpy ndarray -> list
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # numpy scalar -> python scalar
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.generic):
            try:
                return obj.item()
            except Exception:
                return str(obj)
        # dict -> convert values
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        # list/tuple -> convert elements
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    # 写入json文件（先转换不可序列化的 numpy 类型）
    data_for_dump = _convert(data)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data_for_dump, f, ensure_ascii=False, indent=4)
        
        
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parquet2json.py <input_parquet_file> <output_json_file>")
        sys.exit(1)

    input_parquet_file = sys.argv[1]
    output_json_file = sys.argv[2]

    if not os.path.exists(input_parquet_file):
        print(f"Error: The file {input_parquet_file} does not exist.")
        sys.exit(1)

    parquet_to_json(input_parquet_file, output_json_file)
    print(f"Converted {input_parquet_file} to {output_json_file}")
    
    # python parquet2json.py data/nq_hotpotqa_train/test_2wikimultihopqa.parquet data/nq_hotpotqa_train/test_2wikimultihopqa.json
    # python parquet2json.py data/nq_hotpotqa_train/test_hotpotqa.parquet data/nq_hotpotqa_train/test_hotpotqa.json