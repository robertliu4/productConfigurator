import itertools
import pinecone

api_key = "05307ad5-70e6-4b94-b686-9a282429cd7b"

pinecone.init(api_key=api_key, environment='us-west4-gcp-free')

index_name = "projects"

index = pinecone.Index(index_name=index_name);

print(pinecone.describe_index(index_name))
print(index.describe_index_stats())

import csv
project_number = []
project_list = []
with open('data.csv', 'r', encoding='utf-8-sig') as file:
    csv_reader = csv.reader(file)
    count = 0
    for row in csv_reader:
        count += 1
        project_number.append(str(1000 + count))
        project_list.extend(row)


print(project_number)
print(project_list)


import pandas as pd
# data = {
#     'ticketno': [1001,1002,1003,1004,1005,1006,1007,1008,1009,1010],
#     'projects':[
#         'Adjustable 190T rotor lifting beam for precise lifting',
#         'ball pit screw - long term mobile storage for rotors',
#         'folding platforms for nuclear refurbishment case study: safety first and alara - scaffolding',
#         'Generator jacking beam system - generator repair to keep power plants operating',
#         'in station transfer skid - through reactor',
#         'nuclear lead lined flasks for decommissioning, containment and waste disposal',
#         'radioactive waste removal system - small items',
#         'Rotor storage container - long term mobile storage for rotors',
#         'rotor tipping upending fixture - heavy material handling of rotor needs to be precise',
#         'simulation cave for nuclear waste storage - needs to be done safely and efficiently'
#     ]
# }
data = {
    'ticketno':project_number,
    'projects':project_list
}
df = pd.DataFrame(data)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("average_word_embeddings_glove.6B.300d")

df["question_vector"] = df.projects.apply(lambda x: model.encode(str(x)).tolist())

def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk=tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk=tuple(itertools.islice(it,batch_size))
for batch in chunks([(str(t), v) for t, v in zip(df.ticketno, df.question_vector)]):
    index.upsert(vectors=batch)

# print(index.describe_index_stats())
# print(df)

query_questions = ["container"]

query_vectors = [model.encode(str(question)).tolist() for question in query_questions]
query_results = index.query(queries=query_vectors,top_k=10,include_values=False)

print(query_results)

matches = []
scores = []
for match in query_results['results'][0]['matches']:
    matches.append(match['id'])
    scores.append(match['score'])

matches_df = pd.DataFrame({'id': matches, 'score':scores})

df["ticketno"] = df["ticketno"].astype(str)
result_df = matches_df.merge(df,left_on="id",right_on="ticketno")
print(result_df)

