import glob
import re
import pandas as pd

folder_list = glob.glob('./test_files/*CLK_PERIOD*')
CLK_period_pattern = r'__CLK_PERIOD-(.+)__'
CLK_period = []
MIV_pattern = r'Number of MIVs: (.+)\n'
MIV_list = []

for num in range(0,len(folder_list)):
    CLK_period.append(float(re.findall(CLK_period_pattern, folder_list[num])[0]))
    file_path = folder_list[num] + '/init_design_cascade2d.log'
    with open(file_path) as f:
        lines = f.readlines()
    i=0
    while 1:
        temp = re.findall(MIV_pattern, lines[i])
        if(temp):
            MIV_list.append(int(temp[0]))
            break
        i+=1

d = {'CLK_period': CLK_period, 'MIVs': MIV_list}
df = pd.DataFrame(data=d)
df = df.sort_values(by=['CLK_period'])

df.to_csv('MIV_list.csv', index=False)