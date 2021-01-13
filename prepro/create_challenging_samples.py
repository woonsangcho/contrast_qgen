from tqdm import tqdm
import csv
import pickle
from pathlib import Path

# load gen encoding of MSMARCO QA dataset
genenc_path = '../data/genenc_marco.pkl'
with open(genenc_path, "rb") as read_file:
    genenc = pickle.load(read_file)

from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# users may modify to acquire different levels of effective training samples for contrastive learning
low_thresh = 0.7
high_thresh = 0.85

def _computer_cluster(queries, genenc):
    subpairs = []

    # remove quora queries
    new_queries = []
    for q in queries:
        if q in genenc:
            new_queries.append(q)
    queries = new_queries

    # remove duplicates
    queries = list(set(queries))
    if len(queries) < 2:
        return subpairs

    genenc_q_list = []
    for q in queries:
        if q not in genenc:
            return subpairs
        genenc_q_list.append(genenc[q])

    genenc_df = DataFrame.from_records(genenc_q_list)
    db = cosine_similarity(genenc_df)

    for combo in combinations(list(range(len(queries))), 2):
        if db[combo[0], combo[1]] <= high_thresh and db[combo[0], combo[1]] >= low_thresh:
            subpairs.append((queries[combo[0]], queries[combo[1]]))

    return subpairs

# specify the path to the conversational search dataset here
convsession_data_path = None

pairs = []
total_cnt = 0
marco_only_cnt = 0
with open(convsession_data_path, encoding='utf_8') as f:
    f = csv.reader(f, delimiter='\t')
    for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
        total_cnt += 1

        subpairs = _computer_cluster(line[1:], genenc)
        if len(subpairs) == 0:
            continue

        for pair in subpairs:
            pairs.append(pair)
            # also append inverted pair
            pairs.append(pair[::-1])

        marco_only_cnt += 1

print(marco_only_cnt)
print(total_cnt)

# remove duplicates
pairs = list(set(pairs))




