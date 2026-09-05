# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the QA dataset to json format
"""

import json
import os
import datasets
import argparse

"""
示例运行
python ./data_process/prepare_test_data.py --local_dir ./data/nq_hotpotqa_train --data_sources all
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_hotpotqa_train')
    parser.add_argument('--data_sources', default='2wikimultihopqa',choices=['2wikimultihopqa','hotpotqa', 'all']) 
    
    # hotpotqa有两个split：train、dev（7.41k）；2wikimultihopqa有两个split：train、dev（12.6k）

    args = parser.parse_args()

    data_sources = args.data_sources
    
    if data_sources == 'all':
        data_sources = ['2wikimultihopqa','hotpotqa']
    else:
        data_sources = [args.data_sources]

    for data_source in data_sources:

        if data_source != 'strategyqa':
            dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)
        else:
            dataset = datasets.load_dataset('json', data_files="/home/peterjin/mnt/data/strategyqa/test_correct.jsonl")

        if 'test' in dataset:
            print(f'Using the {data_source} test dataset...')
            test_dataset = dataset['test']
        elif 'dev' in dataset:
            print(f'Using the {data_source} dev dataset...')
            test_dataset = dataset['dev']
        else:
            print(f'Using the {data_source} train dataset...')
            test_dataset = dataset['train']
        
        # 只保留id，question，golden_answers列
        test_dataset = test_dataset.remove_columns(
            [col for col in test_dataset.column_names if col not in ['id', 'question', 'golden_answers']]
        )
        
        """
        存为.json
        [
            {"_id": ..., "question": ..., "golden_answers": ...},
            ...
        ]
        """
        os.makedirs(args.local_dir, exist_ok=True)
        out_file = os.path.join(args.local_dir, f'test_{data_source}.json')
        print(f'Saving to {out_file}...')

        records = []
        for ex in test_dataset:
            records.append({
                "id": ex["id"],
                "question": ex["question"],
                "golden_answers": ex["golden_answers"]
            })

        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f'Saved {len(records)} examples to {out_file}')
        
        
        
        

        

    