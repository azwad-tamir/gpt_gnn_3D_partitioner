import glob
import re
import pandas as pd
import numpy as np

file_path = "./aes_tier1.cell"
lib_pattern1 = r'(.+) saed32_hvt|saed32_hvt_std\n'
lib_pattern2 = r'(.+) saed32_lvt|saed32_lvt_std\n'
lib_pattern3 = r'(.+) saed32_rvt|saed32_rvt_std\n'

cell_name =[]
cell_type = []
num = 0

# splitting cells into name and types
with open(file_path) as f:
    lines = f.readlines()
    for i in range(len(lines)):
        if (re.findall(lib_pattern1, lines[i]) != []) or (re.findall(lib_pattern2, lines[i]) != []) or (re.findall(lib_pattern3, lines[i]) != []):
            cell_name.append(lines[i].split()[0])
            cell_type.append(lines[i].split()[1])
            num+=1
        #temp = re.findall(, lines[i])

# creating table
unique_cell_types = []
unique_cell_count = []
for i in cell_type:
    if i in unique_cell_types:
        unique_cell_count[unique_cell_types.index(i)]+=1
    else:
        unique_cell_types.append(i)
        unique_cell_count.append(1)

d = {'cell_name': unique_cell_types, 'cell_count': unique_cell_count}
df = pd.DataFrame(data=d)
df = df.sort_values(by=['cell_count'])

df.to_csv('aes_tier1_cp.csv', index=False)