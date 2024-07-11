'''
This script is used to run the single model attack on the LLM model.
'''
import os
from datetime import datetime
import shlex
import pandas as pd


df = pd.DataFrame(columns=['goal', 'is_success', 'completion', 'trigger'])
now = datetime.now()
filename = now.strftime("%Y-%m-%d_%H-%M-%S")
filename = f'../result/SINGLE_{filename}.csv'
df.to_csv(filename, index=False)

test = pd.read_csv('../data/harmful_behaviors.csv')
OFFSET = 0  # offset means the starting index of the test data

for i in range(OFFSET, OFFSET+100):
    os.system(f'python ../SingleModelSingleBehavior/single.py \
    --user_prompt {shlex.quote(test["goal"][i])} \
    --target {shlex.quote(test["target"][i])} \
    --save_result {filename}')
