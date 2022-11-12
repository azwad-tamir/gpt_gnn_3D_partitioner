import glob
import re
import pandas as pd

util_list = ['0.5', '0.6', '0.7']
clk_list = ['0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1']
MIV_tag = []
MIV_tag_util = []
MIV_tag_clk = []
MIV_tag_netlist = []
MIV_list = []
tag_type_pattern = r'placement_(.+)_2tier_cut_nets.v'

for util in util_list:
    for clk in clk_list:
        path = './dhm_mini_mapped/' + util + '/' + clk + '/*_cut_nets.v'
        folder_list = glob.glob(path)
        for file_path in folder_list:
            with open(file_path, 'r') as fp:
                x = len(fp.readlines())
                tag = file_path.split('/')
                MIV_tag_util.append(tag[2])
                MIV_tag_clk.append(tag[3])
                tag_trimmed = 'placement_' + re.findall(tag_type_pattern, tag[4])[0]
                MIV_tag_netlist.append(tag_trimmed)
                MIV_tag.append(file_path)
                MIV_list.append(x)
                print(file_path, ': ', x)  # 8



d = {'Util': MIV_tag_util, 'CLK': MIV_tag_clk, 'tag': MIV_tag_netlist, 'MIVs': MIV_list}
df = pd.DataFrame(data=d)
df.to_csv('MIV_list.csv', index=False)

