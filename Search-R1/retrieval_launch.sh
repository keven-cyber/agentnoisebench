export CUDA_VISIBLE_DEVICES=5

file_path=./corpus
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

# python search_r1/search/retrieval_server.py --index_path $index_file \
#                                             --corpus_path $corpus_file \
#                                             --topk 3 \
#                                             --retriever_name $retriever_name \
#                                             --retriever_model $retriever_path \
#                                             --faiss_gpu


python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path 


# 检查启动情况
curl -s -X POST http://127.0.0.1:8008/retrieve \
  -H "Content-Type: application/json" \
  -d '{"queries":["Who wrote Sapiens?"],"topk":5,"return_scores":true}'