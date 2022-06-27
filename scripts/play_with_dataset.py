import json
import pandas as pd

with open('../dataset/abcd/data/abcd_v1.1.json') as json_file:
    data = json.load(json_file)
    df_train = pd.DataFrame.from_records(data['train'])
    df_dev = pd.DataFrame.from_records(data['dev'])
    df_test = pd.DataFrame.from_records(data['test'])

# print(df_train.columns)
# print(df_train.head())

print(df_train.delexed[0])
for t in df_train.delexed[0]:
    print(t['text'])

'''
So, df_train.original contains all the conversations between the agent and the customer.

We have 8.034 conversations of varying lengths.

df_train.delexed contains the conversations after pre-processing (delexelised) for training and testing

df_train.delexed[0] contains all the information require for the first conversation

for t in df_train.delexed[0]:
    print(t['text'])
    
This returns all the turns in the first conversation.



________________________________________________________________________________________________
example of how to read from file
out_path = os.path.join(sys.path[0], 'sst_{}.txt')
dataset = pytreebank.load_sst('./raw_data')

# Store train, dev and test in separate files
for category in ['train', 'test', 'dev']:
    with open(out_path.format(category), 'w') as outfile:
        for item in dataset[category]:
            outfile.write("__label__{}\t{}\n".format(
                item.to_labeled_lines()[0][0] + 1,
                item.to_labeled_lines()[0][1]
            ))
# Print the length of the training set
print(len(dataset['train']))
'''