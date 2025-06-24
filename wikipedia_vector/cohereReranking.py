import os
import getpass
import json
import pandas as pd
import cohere

cohere_api_key = "H8aT1c5eupS8ipwRFMRjtKqLcULpHLmbI8oCcs3q"
MONGO_CONN_STR = "mongodb+srv://admin:admin@sandbox.taxfp.mongodb.net/"
# json_file = "/Users/nick.gogan/Desktop/Tech/SA/Workshops/WikipediaVector/sample_mflix_movies.jsonl"
json_file = "/Users/nick.gogan/Desktop/Tech/SA/Workshops/WikipediaVector/movies_sample_dataset.jsonl"

# Create pandas dataframe from json file
# df = pd.read_json(json_file, orient="records", lines=True)
# df[:3]
# print(df[:3])

# Create Cohere client & dataset for it
co_client = cohere.Client(cohere_api_key)
dataset = co_client.datasets.create(name='movies',
                                   data=open(json_file,'rb'),
                                   keep_fields=["fullplot","title","year"],
                                   type="embed-input")
print(co_client.wait(dataset))
print(dataset.id)

embed_job = co_client.embed_jobs.create(dataset_id=dataset.id, 
    input_type='search_document',
    model='embed-english-v3.0', 
    truncate='END')
# embed_job.wait()
# output_dataset = co_client.get_dataset(embed_job.output.id)
# results = list(output_dataset)
# len(results)