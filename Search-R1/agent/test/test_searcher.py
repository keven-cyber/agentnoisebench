from test_api import Searcher

searcher = Searcher()

res = searcher.search("Who wrote Sapiens?", topk=5)

print(res)
