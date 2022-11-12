import os
import sys
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from torch_geometric.datasets import Reddit
from GPT_GNN.data import *
import math
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import random
from scipy.linalg import solve
import numpy as np
import argparse

# Parsing command line arguments
# data_dir_root = sys.argv[1]
# subject_name = sys.argv[2]
# data_dir_root = 'data'
# subject_name = 'aes_cipher'


# AQ 2D placement algorithm:
def aq_placement(data_, raw_lines):
    # data_ = data
    drop_threshold = 10000
    gates_num = len(data_)

    all_nets_dirty = []
    for gate_counter in range(gates_num):
        all_nets_dirty.extend(data_[gate_counter]['nets'])

    print(len(np.unique(all_nets_dirty)))
    nets_name_inverted = {}
    counter = 0
    for net in np.unique(all_nets_dirty):
        nets_name_inverted[net] = counter
        counter += 1
    del all_nets_dirty

    # nets_num = len(nets_name_inverted)
    # nets_name = dict([[v, k] for k, v in nets_name_inverted.items()])
    # find connections of gates to nets
    gates = {}
    gates_names = {}
    gates_names_original = {}
    for gate_counter in range(0, gates_num):
        gate_name = gate_counter
        dummy = []
        for net in data_[gate_counter]['nets']:
            dummy.extend([nets_name_inverted[net]])
        gates[gate_counter] = dummy
        gates_names[gate_counter] = gate_name
        gates_names_original[gate_counter] = data_[gate_counter]['name']

    # read the input and output pins from the netlist
    input_pins = []
    output_pins = []
    input_found = 0
    output_found = 0
    for line in raw_lines:
        current_line = line.lstrip()
        if current_line.startswith('input'):
            input_pins.extend(current_line.replace(" ", "").lstrip('input').rstrip(';\n').split(','))
            if ';' not in current_line:
                input_found = 1
        else:
            if input_found == 1:
                input_pins.extend(current_line.replace(" ", "").lstrip('input').rstrip(';\n').split(','))
                if ';' in current_line:
                    input_found = 0
        if current_line.startswith('output'):
            output_pins.extend(current_line.replace(" ", "").lstrip('output').rstrip(';\n').split(','))
            if ';' not in current_line:
                output_found = 1
        else:
            if output_found == 1:
                output_pins.extend(current_line.replace(" ", "").lstrip('output').rstrip(';\n').split(','))
                if ';' in current_line:
                    output_found = 0
    emptys = []
    for p in range(len(input_pins)):
        if input_pins[p] == '':
            emptys.append(p)
    emptys.reverse()
    for e in emptys:
        input_pins.pop(e)
    emptys = []
    for p in range(len(output_pins)):
        if output_pins[p] == '':
            emptys.append(p)
    emptys.reverse()
    for e in emptys:
        output_pins.pop(e)
    # check if any pin name has been changed in the netlist with "assign"
    assignments = {}
    for line in raw_lines:
        if 'assign' in line:
            dummy = line.replace(" ", "").lstrip('assign').rstrip(';\n').split('=')
            assignments[dummy[1]] = dummy[0]

    for i in range(len(input_pins)):
        if input_pins[i] in assignments:
            input_pins[i] = assignments[input_pins[i]]
    for i in range(len(output_pins)):
        if output_pins[i] in assignments:
            output_pins[i] = assignments[output_pins[i]]

    # read pins from netlist and add it to the connections
    # pins_num_in = len(input_pins)
    # pins_num_out = len(output_pins)
    pins_in = input_pins
    pins_out = output_pins
    pins_dirty = pins_in + pins_out
    pins_cleaned = []
    cords = []
    pins_num = 0
    type_prev = 0
    type = 0
    for pin in pins_dirty:
        dummy = pin.split(':')
        while type == type_prev:
            type = random.randint(0, 3)
        type_prev = type
        rand = random.randint(0, 99)
        if type == 0:
            x = 0
            y = rand
        if type == 1:
            x = 99
            y = rand
        if type == 2:
            x = rand
            y = 0
        if type == 3:
            x = rand
            y = 99
        if len(dummy) > 1:
            num = int(dummy[0][1:])
            name = dummy[1][2:]
            # print(num, name)
            pins_dummy = []
            for i in range(num):
                if name + '[' + str(i) + ']' not in nets_name_inverted:
                    continue
                pins_dummy.append(name + '[' + str(i) + ']')
                cords.append([x, y])
                pins_num += 1
            pins_cleaned.extend(pins_dummy)
        else:
            if pin not in nets_name_inverted:
                continue
            pins_cleaned.append(pin)
            cords.append([x, y])
            pins_num += 1

    pins_cord = {}
    pins = {}
    for pin_counter in range(gates_num, gates_num + len(pins_cleaned)):
        pin = pins_cleaned[pin_counter - gates_num]
        if pin not in nets_name_inverted:
            continue
        con_pin = nets_name_inverted[pin]
        pins_cord[pin_counter] = cords[pin_counter - gates_num]
        pins[pin_counter] = [con_pin]
    gates_wpin = z = {**gates, **pins}

    nets_wpin = {}
    all_gates = sorted(gates_wpin.keys())
    for gate in all_gates:
        nets = gates_wpin[gate]
        for net in nets:
            if net not in nets_wpin:
                dummy = []
            else:
                dummy = nets_wpin[net]
            if gate not in dummy:
                dummy.extend([gate])
            nets_wpin[net] = dummy

    # make k-nets into 2-nets
    k_lcm = 1

    # extra_nets = 0
    # remove_nets = 0
    # index = []
    # duo = []
    # old_net_counter = 0

    all_nets = nets_wpin.keys()
    nets_wpin_ext = {}
    weights = {}
    unassigned_net = max(all_nets) + 1
    missed_nets = []
    for net in all_nets:
        old_con = nets_wpin[net]
        con_num = len(old_con)
        if con_num > drop_threshold:
            print('Net has too many connections, it is dropped:', net, con_num)
            missed_nets.append(net)
            continue

        if con_num <= 2:
            weights[net] = k_lcm
            nets_wpin_ext[net] = old_con
            continue
        for i in range(len(old_con) - 1):
            for j in range(i + 1, len(old_con)):
                one = old_con[i]
                two = old_con[j]
                nets_wpin_ext[unassigned_net] = [one, two]
                weights[unassigned_net] = k_lcm / (con_num - 1)
                unassigned_net += 1

    gates_wpin_ext = {}
    all_nets = nets_wpin_ext.keys()
    for net in all_nets:
        if net in missed_nets:
            continue
        connected_gates = nets_wpin_ext[net]
        for gate in connected_gates:
            if gate not in gates_wpin_ext:
                dummy = []
            else:
                dummy = gates_wpin_ext[gate]
            if net not in dummy:
                dummy.extend([net])
            gates_wpin_ext[gate] = dummy

    # find connection matrix between gates
    c = np.zeros((gates_num + pins_num, gates_num + pins_num))
    b_x = np.zeros(gates_num)
    b_y = np.zeros(gates_num)
    all_pins = sorted(list(pins.keys()))
    gate_gate = {}
    # for gate in range(len(con_wpin_ext)):
    for gate in all_gates:
        connected_nets = gates_wpin_ext[gate]
        connected_gates = []
        connected_weight = []
        for net in connected_nets:
            connected_gates_to_net = nets_wpin_ext[net]
            for connected_gate in connected_gates_to_net:
                if connected_gate == gate:
                    continue
                connected_gates.append(connected_gate)
                connected_weight.append(-weights[net])
                if connected_gate in all_pins:
                    b_x[gate] += weights[net] * pins_cord[connected_gate][0]
                    b_y[gate] += weights[net] * pins_cord[connected_gate][1]
        gate_gate[gate] = connected_gates
        for counter in range(len(connected_gates)):
            c[gate][connected_gates[counter]] = connected_weight[counter]

    del weights

    print('Calculating placement.')
    a_wopin = c[:-pins_num, :-pins_num]

    c_sum = -np.sum(c, axis=0)
    for gate in range(len(a_wopin)):
        a_wopin[gate, gate] = c_sum[gate]

    x = solve(a_wopin, b_x)
    y = solve(a_wopin, b_y)

    print('Generating output.', subject)
    output = np.zeros((gates_num, 3))
    for i in range(len(output)):
        output[i][0] = i
        output[i][1] = x[i]
        output[i][2] = y[i]

    # del b_x, b_y, output, x, y, connected_pins
    output_wpin = np.zeros((gates_num + pins_num, 3))
    output_wpin[:gates_num] = output

    for pin in all_pins:
        dummy = [pin, pins_cord[pin][0], pins_cord[pin][1]]
        output_wpin[pin] = dummy

    x_gate = []
    y_gate = []
    x_pin = []

    y_pin = []
    all_gates_wopin = sorted(gates.keys())
    for i in range(len(output_wpin)):
        if output_wpin[i, 0] in all_gates_wopin:
            x_gate.append(output_wpin[i, 1])
            y_gate.append(output_wpin[i, 2])
        elif output_wpin[i, 0] in all_pins:
            x_pin.append(output_wpin[i, 1])
            y_pin.append(output_wpin[i, 2])

    for i in range(len(data_)):
        data_[i]['x'] = x_gate[i]
        data_[i]['y'] = y_gate[i]
    return data_


timings = [0.7]
utils = [0.7]
# placement_types = ['none', 'aq', 'icc2', 'both']
placement_types = ['icc2']
timing_types = ['none', 'included']
# hop_types = ['none', 'included']
hop_types = ['included']
for util in utils:
    for timing in timings:
        # timing = 0.8
        # data_dir = 'data/aes_cipher.v'
        # lef_dir = 'data/saed32_rvt.lef'
        # timings = [0.5,0.6,0.7,0.8,0.9,1.0,1.1]
        parser = argparse.ArgumentParser(description='Pre-processing Data')

        parser.add_argument('--data_dir_root', type=str, default='data', help='The address of the initial data')
        parser.add_argument('--subject_name', type=str, default='aes_cipher', help='The name of the circuit')

        args = parser.parse_args()
        subject_name = args.subject_name

        data_dir = args.data_dir_root + '/' + subject_name + '/' + subject_name + '_' + str(util).split('.')[0] + 'P' + str(util).split('.')[
            1] + '/netlists/' + subject_name + '__' + 'CLIB_NAME-saed32__CLK_PERIOD-' + str(timing) + '00__CORE_UTIL-' + str(
            util) + '00/' + subject_name + '_mapped.v'
        feature_dir = args.data_dir_root + '/' + subject_name + '/' + subject_name + '_' + str(util).split('.')[0] + 'P' + str(util).split('.')[
            1] + '/Features/features_' + str(timing).split('.')[0] + 'P' + str(timing).split('.')[1] + '00.txt'
        placement_dir = args.data_dir_root + '/' + subject_name + '/' + subject_name + '_' + str(util).split('.')[0] + 'P' + str(util).split('.')[
            1] + '/ICC2_initialization/ICC2_Init_' + str(timing).split('.')[0] + 'P' + str(timing).split('.')[
                            1] + '00.def'
        lef_dir = args.data_dir_root + '/' + subject_name + '/celllist.txt'

        subject = data_dir.split('/')[-1].rstrip('.v')
        raw_lines = []
        with open(data_dir, 'r') as f:
            for line in f:
                raw_lines.append(line)

        # find the last wire assignment before the netlist begins
        last_wire_counter = 0
        counter = 0
        for line in raw_lines:
            if 'wire' in line or 'tri' in line or 'assign' in line:
                last_wire_counter = counter
            counter += 1
        # print(last_wire_counter, raw_lines[last_wire_counter])

        # find the first line of the netlist
        counter = 0
        first_line = ''
        cut_off = 0
        for line_counter in range(last_wire_counter, len(raw_lines)):
            if ';' in raw_lines[line_counter]:
                cut_off = line_counter + 2
                first_line = raw_lines[cut_off]
                break

        print('Please check that the following line is the first line of the netlist:')
        print(subject, cut_off, first_line.rstrip('\n'))

        wires = []
        flag = 0
        for line in raw_lines:
            if 'wire' in line or 'tri' in line:
                if len(line.split()) == 3 and ':' in line:
                    wires.append(line.split()[2].rstrip(';'))
                if len(line.split()) == 4 and ':' in line:
                    wires.append(line.split()[2] + ' ')

        lines = []
        dummy = ''
        line_index = {}
        start_counter = cut_off
        counter = 0
        for line in raw_lines[cut_off:-2]:
            if line == '\n':
                counter += 1
                continue
            dummy += line.rstrip('\n')
            if line.rstrip('\n')[-1] == ';':
                lines.append(dummy)
                dummy = ''
                line_index[len(lines) - 1] = [start_counter, start_counter + counter + 1]
                start_counter += counter + 1
                counter = 0
            else:
                counter += 1
        data = {}
        counter = 0
        for line in lines:
            kind = line.split()[0]
            name = line.split()[1]
            rest = ''.join(line.split()[2:]).lstrip('(').rstrip(';').rstrip(')').split('.')[1:]
            ports = [x.split('(')[0] for x in rest]
            count = line.count('.')
            connections = [x.split('(')[1].rstrip('),') for x in rest]
            if '' in connections:
                connections.remove('')
            data[counter] = {'kind': kind, 'name': name, 'con': count, 'ports': ports, 'nets': connections,
                             'time': counter}
            counter += 1

        if 'aq' in placement_types or 'both' in placement_types:
            data = aq_placement(data, raw_lines)

        gatenet = defaultdict(lambda: [])
        netgate = defaultdict(lambda: [])

        for i in data:
            gate = data[i]['name']
            net = data[i]['nets']
            if len(net) == 1:
                net = [net]
            gatenet[gate].extend(net)
            for n in net:
                netgate[n].extend([gate])
        lens = []
        nets = []
        for n in netgate:
            lens.append(len(netgate[n]))
            nets.append(n)
        print(np.max(lens), nets[np.argmax(lens)])

        gategate = {}
        gategateport = defaultdict(lambda: [])
        for i in data:
            gate = data[i]['name']
            net = data[i]['nets']
            port = data[i]['ports']

            if len(net) == 1:
                net = [net]
            connected_gates = []
            for n in range(len(net)):
                current_net = net[n]
                current_port = port[n]
                connected_gate = netgate[current_net][:]
                connected_gate.remove(gate)
                # connected_gate = netgate[current_net].remove(gate)
                # connected_gates.append(connected_gate)
                connected_gates.extend(connected_gate)
                if len(connected_gate) == 0:
                    continue
                for g in connected_gate:
                    # if g != gate:
                    if (g, gate) not in gategateport:
                        gategateport[(gate, g)].append(current_port)
                    else:
                        gategateport[(g, gate)].append(current_port)

            data[i]['connected_gates'] = connected_gates
            gategate[gate] = connected_gates

        gatemapping = {}
        gatecounter = {}
        gatetypes = {}
        onehop = {}
        twohop = {}
        placement = {}
        for i in range(len(data)):
            gatemapping[data[i]['name']] = data[i]['time']
            gatecounter[data[i]['time']] = data[i]['name']
            gatetypes[data[i]['time']] = data[i]['kind']
            onehop[data[i]['time']] = [math.log(len(data[i]['connected_gates']), 10)]
            if 'aq' in placement_types or 'both' in placement_types:
                placement[data[i]['time']] = [data[i]['x'], data[i]['y']]
        for i in range(len(data)):
            dummy_twohop = 0
            for connected_gate in data[i]['connected_gates']:
                dummy_twohop += len(data[gatemapping[connected_gate]]['connected_gates'])
            twohop[data[i]['time']] = [math.log(dummy_twohop, 10)]

        feature_lines = []
        with open(feature_dir) as f:
            for line in f:
                # if '.' not in line:
                # print(line)
                if '.' in line:
                    feature_lines.append([line.split(' ')[0], line.split(' ')[1], line.split(' ')[2].rstrip('\n')])
        feature_df = pd.DataFrame(feature_lines, columns=['gate', 'feature_1', 'feature_2'])
        feature_names = np.array(feature_df['gate'])
        feature_array = np.array(feature_df[['feature_1', 'feature_2']], dtype=float)
        feature_map = {}
        for i in range(len(feature_df)):
            feature_map[feature_names[i]] = feature_array[i]

        placement_lines = []
        with open(placement_dir) as f:
            for line in f:
                # if not line.startswith(' - '):
                #     print(line)
                if line.startswith(' - '):
                    line = line.strip()
                    placement_lines.append([line.split(' ')[1], line.split(' ')[6], line.split(' ')[7]])
        placement_df = pd.DataFrame(placement_lines, columns=['gate', 'placement_1', 'placement_2'])
        placement_names = np.array(placement_df['gate'])
        placement_array = np.array(placement_df[['placement_1', 'placement_2']], dtype=float)
        placement_array = 100 * (placement_array - placement_array.min(axis=0)) / (
                    placement_array.max(axis=0) - placement_array.min(axis=0))
        placement_map = {}
        for i in range(len(placement_df)):
            placement_map[placement_names[i]] = placement_array[i]

        for placement_type in placement_types:
            for timing_type in timing_types:
                for hop_type in hop_types:
                    print('Util:', util, ', Clock:', timing, ', Placement:', placement_type, ', Timing:', timing_type,
                          ', Hop:',
                          hop_type)
                    # dataset = Reddit(root='data/')
                    graph_reddit = None
                    el = None
                    graph_reddit = Graph()
                    el = defaultdict(  #target_id
                                        lambda: defaultdict( #source_id(
                                            lambda: int # time
                                        ))
                    # for i, j in tqdm(dataset.data.edge_index.t()):
                    #     el[i.item()][j.item()] = 1


                    allgates = list(gatemapping.keys())
                    allgatetypes = []
                    for i in range(len(gatetypes)):
                        allgatetypes.append(gatetypes[i])
                    for gate in allgates:
                        connections = gategate[gate]
                        for connected_gate in connections:
                            el[gatemapping[connected_gate]][gatemapping[gate]] = 1

                    target_type = 'def'
                    graph_reddit.edge_list['def']['def']['def'] = el
                    n = list(el.keys())
                    degree = np.zeros(np.max(n)+1)
                    for i in n:
                        degree[i] = len(el[i])
                    # print(dataset.data.x.numpy()[0])

                    print(len(el))

                    onehot_encoded = []
                    for i in range(len(data)):
                        current_name = data[i]['name']
                        timing_feature_dummy = []
                        if current_name in feature_map:
                            timing_feature_dummy = list(feature_map[current_name])
                        else:
                            timing_feature_dummy = list(np.mean(feature_array, axis=0))
                        if current_name in placement_map:
                            placement_feature_dummy = list(placement_map[current_name])
                        else:
                            placement_feature_dummy = list(np.mean(placement_array, axis=0))
                        if 'aq' in placement_types or 'both' in placement_types:
                            aq_placement_dummy = placement[gatemapping[current_name]]
                        if placement_type == 'none':
                            aq_placement_dummy = []
                            placement_feature_dummy = []
                        if placement_type == 'icc2':
                            aq_placement_dummy = []
                        if placement_type == 'aq':
                            placement_feature_dummy = []
                        if hop_type == 'none':
                            onehop_dummy = []
                            twohop_dummy = []
                        else:
                            onehop_dummy = onehop[gatemapping[current_name]]
                            twohop_dummy = twohop[gatemapping[current_name]]
                        if timing_type == 'none':
                            timing_feature_dummy = []
                        onehot_encoded.append(timing_feature_dummy + placement_feature_dummy
                                              + onehop_dummy + twohop_dummy +
                                              aq_placement_dummy)
                    onehot_encoded = np.array(onehot_encoded)

                    enc = OneHotEncoder(handle_unknown='ignore')
                    values = np.array(allgatetypes)
                    label_encoder = LabelEncoder()
                    integer_encoded = label_encoder.fit_transform(values)
                    # print(integer_encoded)
                    # # binary encode
                    # onehot_encoder = OneHotEncoder(sparse=False)
                    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
                    # onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
                    print('Feature shape:', onehot_encoded.shape)

                    # x = np.concatenate((dataset.data.x.numpy(), np.log(degree).reshape(-1, 1)), axis=-1)
                    x = np.concatenate((onehot_encoded, np.log(degree).reshape(-1, 1)), axis=-1)

                    graph_reddit.node_feature['def'] = pd.DataFrame({'emb': list(x)})

                    idx = np.arange(len(graph_reddit.node_feature[target_type]))
                    np.random.seed(42)
                    np.random.shuffle(idx)

                    graph_reddit.pre_target_nodes   = idx[ : int(len(idx) * 0.7)]
                    graph_reddit.train_target_nodes = idx[int(len(idx) * 0.7) : int(len(idx) * 0.8)]
                    graph_reddit.valid_target_nodes = idx[int(len(idx) * 0.8) : int(len(idx) * 0.9)]
                    graph_reddit.test_target_nodes  = idx[int(len(idx) * 0.9) : ]

                    integer_encoded = integer_encoded.reshape(len(integer_encoded))
                    # graph_reddit.y = dataset.data.y
                    graph_reddit.y = integer_encoded

                    # dill.dump(graph_reddit, open('data/graph_reddit.pk', 'wb'))
                    # dill.dump(graph_reddit, open('data/graph_reddit_netlist.pk', 'wb'))
                    try:
                        os.makedirs('graphs/')
                    except OSError as error:
                        print(error)
                    dill.dump(graph_reddit, open('graphs/graph_'+subject+'_'+str(timing)+'_util'+str(util)+'_placement_'+placement_type+'_hop_'+hop_type+'_timing_'+timing_type+'.pk', 'wb'))
