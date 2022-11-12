import networkx as nx
import re
import os
import io
#from halp.undirected_hypergraph import UndirectedHypergraph
import copy
class vparser(object):

    def __init__(self, vcode_name):
        self.vcode_name = vcode_name
        with io.open(self.vcode_name, "rt") as f:  # rt read as text mode
            code = f.read()
        code = code.split(';')
        self.vcode = [w.replace('\n', '') for w in code]

    def gethygraph(self):

        '''generate the hypergraph dict from verilog code'''

        # regex patterns
        module_pattern = r'module\s(.+)'  # like input abc123
        wire_pattern = r'wire\s(.+)'  # like input abc123
        #net_pattern = r'\.\w*\(([^\)]+?)\)'  # like .abc123(abc123)
        net_pattern =r'\(.+'  # like .abc123(abc123)
        device_pattern = r'\.\w*\((.+)'  # include .abc123(
        input_pattern = r'input\s(.+)'  # like input abc123
        output_pattern = r'output\s(.+)'  # like output abc123
        index_pattern = r'\[[0-9]*:0\]\s'  # like [123:0]
        netindex_pattern = r'\[[0-9]*\]'  # like [123]
        device_pattern_re = re.compile(device_pattern)
        hygraphdict = {}

        self.inputnets = []
        self.outputnets = []
        self.getNets = []
        self.node_names = []
        self.node_node = []
        self.allOutputs = []
        self.allIntputs = []
        self.wires = []

        name = self.vcode[0][self.vcode[0].find("module") + 6:].split(' ')
        self.id = name[1]
        for i in range(len(self.vcode)):
            #print(self.vcode)
            if self.vcode[i].startswith("//"):
                continue
            # get wires
            iswire = re.findall(wire_pattern, self.vcode[i])
            if iswire:
                self.wires.append(iswire)

            # get inout nets
            isinput = re.findall(input_pattern, self.vcode[i])
            isoutput = re.findall(output_pattern, self.vcode[i])

            # identity input nets
            if isinput:
                inputs = isinput[0].split(", ")  # split each input by '' , ''
                self.allIntputs += inputs
                inputs = [re.sub(index_pattern, '', w) for w in inputs]
                # if 'clk' in inputs:
                #     inputs.remove('clk')
                self.inputnets += inputs  # Get the inputs

            if isoutput:
                outputs = isoutput[0].split(", ")
                self.allOutputs += outputs
                # print("outputs are: ", outputs)
                outputs = [re.sub(index_pattern, '', w) for w in outputs]
                self.outputnets += outputs  # Get the outputs

            isdevice = device_pattern_re.search(self.vcode[i])  # check if the string contains device description
            if isdevice:
                #print(self.vcode[i])
                nets = re.findall(net_pattern, self.vcode[i])  # extract all the nets connecting to the device (Wires)
                # print("nets is {}".format(nets))
                #nets = [w.replace(' ', '') for w in nets]
                self.getNets.append(nets[0])
                instances = self.vcode[i].split()  # get the device name and the type as the array [type, name]
                self.node_names.append(instances[0])
                self.node_node.append(instances[1])
                #print("instances is {}".format(instances))
                #print('instances are: ', instances[1])
                # construct dict for the hypergraph
                for count, key in enumerate(nets):
                    isio = re.sub(netindex_pattern, '', key)  # remove [0-9] in netnames
                    if key not in hygraphdict:
                        for i in range(len(instances)):
                            if instances[i] != '':  # get rid of the '' in the instance names
                                if isio in self.inputnets:
                                    hygraphdict[key] = [[instances[i], instances[i + 1], 'input']]
                                elif isio in self.outputnets:
                                    hygraphdict[key] = [[instances[i], instances[i + 1], 'output']]
                                else:
                                    hygraphdict[key] = [[instances[i], instances[i + 1]]]
                                break
                    else:
                        for i in range(len(instances)):
                            if instances[i] != '':  # get rid of the '' in the instance names
                                if isio in self.inputnets:
                                    hygraphdict[key].append([instances[i], instances[i + 1], 'input'])
                                elif isio in self.outputnets:
                                    hygraphdict[key].append([instances[i], instances[i + 1], 'output'])
                                else:
                                    hygraphdict[key].append([instances[i], instances[i + 1]])
                                break

        print(len(self.node_names))
        return hygraphdict

obj = vparser('aes_cipher.v')
x = obj.gethygraph()
input_net = obj.inputnets
out_net = obj.outputnets
get_net = obj.getNets
get_net2 = copy.deepcopy(get_net)
node_names = obj.node_names
allout = obj.allOutputs
allin = obj.allIntputs
node_node = obj.node_node
wires = obj.wires
wires_new = [item for sublist in wires for item in sublist]
node_node1 = []
node_node2 = []

# Reading tier files to identify cell classification
file1 = open('2tier_0_aes_cipher.txt', 'r')
Lines_tier0 = file1.readlines()
file1.close()
file1 = open('2tier_1_aes_cipher.txt', 'r')
Lines_tier1 = file1.readlines()
file1.close()
file1 = open('2tier_cut_nets_aes_cipher.txt', 'r')
Lines_cut_nets = file1.readlines()
file1.close()

cells_tier0 = []
cells_tier1 = []
cut_nets = []
for line in Lines_tier0:
    #line.replace('\n', '')
    line = line.replace('\n', '').replace('\r', '')
    cells_tier0.append(line)
for line in Lines_tier1:
    #line.replace('\n', '')
    line = line.replace('\n', '').replace('\r', '')
    cells_tier1.append(line)
for line in Lines_cut_nets:
    line = line.replace('\n', '').replace('\r', '')
    cut_nets.append(line)
# get_net_all = list(set(get_net_all))
# out_out = input_net

# Fixing wires formatting:
wires1 = []
for i in wires_new:
    temp = str(i).replace(' ','')
    wires1.append(temp.replace(',', ' , '))

wires2 = []
for i in wires1:
    if i[0] == "[":
        wires2.append(i.replace(']', '] '))
    else:
        wires2.append(i)

cells1_tier0 = []
cells1_tier1 = []
# Assigning tiers to cells:
for net in node_node:
   if net in cells_tier0:
       temp = net + 'tier0'
       node_node1.append(temp)
       cells1_tier0.append(temp)
   elif net in cells_tier1:
       temp = net + 'tier1'
       node_node1.append(temp)
       cells1_tier1.append(temp)
   else:
       print("Feltal error: All instances not assigned!!")

# Extracting individual nets from a netlist line::
get_net1 = []
pattern = "\.(.*?)\)"
pattern1 = r'\((.+)'
for nets1 in get_net:
    substring1 = []
    substring = re.findall(pattern, nets1)
    for nets in substring:
        temp_str = re.findall(pattern1, nets)
        if temp_str.__len__() != 0:
            flat_list = temp_str[0]
            flat_list = flat_list.replace(' ','')
            substring1.append(flat_list)
    get_net1.append(substring1)

# Cutting nets
cut_wires = []
cut_nets1 = []
pattern2 = "1'b.+"
for nets in cut_nets:
    if not re.findall(pattern2, nets):
        cut_nets1.append(nets)
    else:
        print(nets)

i=0
for nets in get_net1:
    j=0
    repeat_nets = []
    for node in nets:
        if node in cut_nets1:
            temp = node.replace('[', 'O')
            temp = temp.replace(']', 'C')
            if re.findall(r'.+tier1', node_node1[i]):
                if not(node in repeat_nets):
                    get_net2[i] = get_net2[i].replace(node,temp+'tier1')
                get_net1[i][j] = get_net1[i][j].replace(node,temp+'tier1')
                get_net1[i][j] = get_net1[i][j].replace(' ','')
                cut_wires.append(temp+'tier1')
                repeat_nets.append(node)
        j+=1
        get_net2[i] = get_net2[i].replace('  ','')
    i+=1
cut_wires = list(set(cut_wires))
# cut_wires1 = []
# for nets in cut_wires:
#     temp = nets.replace(']', 'C')
#     cut_wires1.append(temp.replace('[', 'O'))

#print(substring1)
miv_wires = []
i=0
for net in cut_nets1:
    miv_wires.append('mivwire' + str(i))
    i+=1

# Separating input and output nets with and without brackets:
input_net_braket = []
input_net_normal = []
output_net_braket = []
output_net_normal = []
for net in allin:
    if net[0] == '[':
        input_net_braket.append(net)
    else:
        input_net_normal.append(net)

for net in allout:
    if net[0] == '[':
        output_net_braket.append(net)
    else:
        output_net_normal.append(net)

with open("output_circuit.v", "w") as out_circuit:
    out_circuit.write("module aes_cipher ( {},{} );\n".format(', '.join(input_net), ', '.join(out_net)))
    out_circuit.write("\n")

    for net in input_net_braket:
        out_circuit.write("  input {};\n".format(net))
    out_circuit.write("  input {};\n".format(', '.join(input_net_normal)))
    out_circuit.write("\n")

    for net in output_net_braket:
        out_circuit.write("  output {};\n".format(net))
    out_circuit.write("  output {};\n".format(', '.join(output_net_normal)))
    out_circuit.write("\n")

    for i in wires2:
        out_circuit.write("  wire {} ;\n".format(i))
    # out_circuit.write("wire {};\n".format(','.join(wire_out)))
    # out_circuit.write("\n")
    #out_circuit.write("  wire  ")
    out_circuit.write("  wire  {} ;\n".format(' , '.join(cut_wires)))
    #out_circuit.write(";\n")
    #out_circuit.write("  wire  ")
    out_circuit.write("  wire  {} ;\n".format(' , '.join(miv_wires)))
    #out_circuit.write(";\n")
    #out_circuit.write("  wire {};\n".format(', '.join(miv_wires)))
    i=0
    for nodes in node_names:
        out_circuit.write("  {} {} {};\n".format(nodes, node_node1[i],get_net2[i]))
        i+=1

    miv_num=0
    for nodes in cut_nets1:
        out_circuit.write("  {} {} ( .I( {} ), .Z( {} ) );\n".format('MIV_SOURCE', 'Msource'+str(miv_num)+'tier0', nodes, 'mivwire'+str(miv_num)))
        out_circuit.write("  {} {} ( .I( {} ), .Z( {} ) );\n".format('MIV_SINK', 'Msink' + str(miv_num)+'tier1', 'mivwire' + str(miv_num), (nodes.replace('[', 'O')).replace(']', 'C')+'tier1'))
        miv_num+=1
    out_circuit.write("endmodule\n")
