import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import solve
import random
import re
import argparse

random.seed(42)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Partitioning algorithm for 3D IC design.")
    parser.add_argument(
        "--data_dir",
        help="Directory to the netlist",
        type=str,
        required=True,
        default='data/netlist_sample.v'
    )
    parser.add_argument(
        "--lef_dir",
        help="Directory to the lef file",
        type=str,
        required=False,
        default='files/saed32_rvt.lef'
    )
    parser.add_argument(
        "--tier_n",
        help="number of tiers, currently not supported for reinforcement",
        type=int,
        required=False,
        default=2
    )
    parser.add_argument(
        "--out_dir",
        help="Directory for the output images and netlists to be saved",
        type=str,
        required=True,
        default='results/'
    )
    parser.add_argument(
        "--reinforcement",
        help="Use bruteforce reinforcement learning partitioning, 0 or 1",
        type=int,
        required=False,
        default=0
    )
    parser.add_argument(
        "--do_perm",
        help="Agent type in reinforcement learning, 0 for shuffle agent, 1 for tile agent",
        type=int,
        required=False,
        default=0
    )
    parser.add_argument(
        "--drop_threshold",
        help="Minimum number of connections for a net so that the net is not included and is dropped from computations",
        type=int,
        required=False,
        default=10000
    )
    args = vars(parser.parse_args())
    return args


args = parse_arguments()
data_dir = args.get('data_dir')
lef_dir = args.get('lef_dir')
tier_ns = args.get('tier_n')
out_dir = args.get('out_dir')
reinforcement = args.get('reinforcement')
do_perm = args.get('do_perm')
drop_threshold = args.get('drop_threshold')

subject = data_dir.split('/')[-1].rstrip('.v')
subjects = [subject]
# subjects = ['netlist', 'new', 'ldpc_decoder', 'aes_cipher', 'fpu', 'FFT128','aes_cipher2','aes_cipher3']
# subjects = ['aes_cipher_new', 'ldpc_decoder_new', 'fpu_new', 'FFT128_new']
# tier_n = 4

for subject in subjects:
    for tier_n in [tier_ns]:
        my_time = []
        my_time.append(time.time())
        # load netlist
        raw_lines = []
        with open(data_dir, 'r') as f:
            for line in f:
                raw_lines.append(line)
        my_time.append(time.time())

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
        print(subject, cut_off, first_line)


        wires = []
        flag = 0
        for line in raw_lines:
            if 'wire' in line or 'tri' in line:
                if len(line.split()) == 3 and ':' in line:
                    wires.append(line.split()[2].rstrip(';'))
                if len(line.split()) == 4 and ':' in line:
                    wires.append(line.split()[2]+' ')

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
            data[counter] = {'kind': kind, 'name': name, 'con': count, 'ports': ports, 'nets': connections}
            counter += 1
        # gates_names = []
        # for i in range(len(data)):
        #     gates_names.append(data[i]['name'])

        gates_num = len(data)

        all_nets_dirty = []
        for gate_counter in range(gates_num):
            all_nets_dirty.extend(data[gate_counter]['nets'])

        print(len(np.unique(all_nets_dirty)))
        my_time.append(time.time())
        nets_name_inverted = {}
        counter = 0
        for net in np.unique(all_nets_dirty):
            nets_name_inverted[net] = counter
            counter += 1
        del all_nets_dirty

        nets_num = len(nets_name_inverted)
        nets_name = dict([[v,k] for k,v in nets_name_inverted.items()])
        # find connections of gates to nets
        gates = {}
        gates_names = {}
        gates_names_original = {}
        for gate_counter in range(0, gates_num):
            gate_name = gate_counter
            dummy = []
            for net in data[gate_counter]['nets']:
                dummy.extend([nets_name_inverted[net]])
            gates[gate_counter] = dummy
            gates_names[gate_counter] = gate_name
            gates_names_original[gate_counter] = data[gate_counter]['name']


        # read the input and output pins from the netlist
        input_pins = []
        output_pins = []
        for line in raw_lines:
            if 'input' in line:
                input_pins.extend(line.replace(" ", "") .lstrip('input').rstrip(';\n').split(','))
            if 'output' in line:
                output_pins.extend(line.replace(" ", "") .lstrip('output').rstrip(';\n').split(','))

        # check if any pin name has been changed in the netlist with "assign"
        assignments = {}
        for line in raw_lines:
            if 'assign' in line:
                dummy = line.replace(" ", "") .lstrip('assign').rstrip(';\n').split('=')
                assignments[dummy[1]] = dummy[0]

        for i in range(len(input_pins)):
            if input_pins[i] in assignments:
                input_pins[i] = assignments[input_pins[i]]
        for i in range(len(output_pins)):
            if output_pins[i] in assignments:
                output_pins[i] = assignments[output_pins[i]]

        # read pins from netlist and add it to the connections
        pins_num_in = len(input_pins)
        pins_num_out = len(output_pins)
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
                    pins_dummy.append(name + '[' + str(i) + ']')
                    cords.append([x, y])
                    pins_num += 1
                pins_cleaned.extend(pins_dummy)
            else:
                pins_cleaned.append(pin)
                cords.append([x, y])
                pins_num += 1

        pins_cord = {}
        pins = {}
        for pin_counter in range(gates_num, gates_num + len(pins_cleaned)):
            pin = pins_cleaned[pin_counter - gates_num]
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
        my_time.append(time.time())

        # make k-nets into 2-nets
        k_lcm = 1

        extra_nets = 0
        remove_nets = 0
        index = []
        duo = []
        old_net_counter = 0

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
                    weights[unassigned_net] = k_lcm/(con_num - 1)
                    unassigned_net += 1

        my_time.append(time.time())


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
        my_time.append(time.time())


        # find connection matrix between gates
        c = np.zeros((gates_num + pins_num, gates_num + pins_num))
        b_x = np.zeros(gates_num)
        b_y = np.zeros(gates_num)
        all_pins = sorted(list(pins.keys()))
        my_time.append(time.time())
        gate_gate = {}
        # for gate in range(len(con_wpin_ext)):
        for gate in all_gates:
            if 'FFT128' in subject:
                if gate == 102928:
                    continue
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

        my_time.append(time.time())

        del weights

        print('Calculating placement.')
        a_wopin = c[:-pins_num, :-pins_num]

        my_time.append(time.time())

        c_sum = -np.sum(c, axis=0)
        for gate in range(len(a_wopin)):
            a_wopin[gate, gate] = c_sum[gate]

        my_time.append(time.time())

        x = solve(a_wopin, b_x)
        y = solve(a_wopin, b_y)
        my_time.append(time.time())

        print('Generating output.', subject)
        output = np.zeros((gates_num, 3))
        for i in range(len(output)):
            output[i][0] = i
            output[i][1] = x[i]
            output[i][2] = y[i]
        # np.savetxt('data/'+subject+'_output.txt', output)
        my_time.append(time.time())

        # del b_x, b_y, output, x, y, connected_pins

        for i in range(1, len(my_time)):
            print(i, my_time[i] - my_time[i-1])
        print(my_time[-1] - my_time[0])


        output_wpin = np.zeros((gates_num+pins_num, 3))
        output_wpin[:gates_num] = output

        for pin in all_pins:
            dummy = [pin, pins_cord[pin][0], pins_cord[pin][1]]
            output_wpin[pin] = dummy
        # np.savetxt('data/'+subject+'_output_wpin.txt', output_wpin)

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


        fig, ax = plt.subplots()
        scale = 50.0
        plt.xlim([-10, 110])
        plt.ylim([-10, 110])
        ax.scatter(x_gate, y_gate, c='blue', s=scale, label='Blocks', edgecolors='darkblue')
        ax.scatter(x_pin, y_pin, c='red', s=scale, label='Pins', edgecolors='darkred')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir + subject + '.png', dpi=300)
        # plt.show()
        # plt.close()

        try:
            lef_lines = []
            with open('files/saed32_rvt.lef', 'r') as f:
                for line in f:
                    for i in line.split('\n'):
                        if i != '':
                            lef_lines.append(i.lstrip().rstrip(';').rstrip())

            dummy = ''
            dimensions = {}
            dimensions_x = {}
            dimensions_y = {}
            flag = 0
            for line in lef_lines:
                if 'MACRO' in line:
                    name = line.lstrip('MACRO').lstrip()
                    flag = 1
                if flag:
                    if 'SIZE' in line:
                        x = float(line.split()[1])
                        y = float(line.split()[3])
                        size_elements = x * y
                        dimensions[name] = size_elements
                        dimensions_x[name] = x
                        dimensions_y[name] = y
        except:
            print('LEF file not found, all gates are noew considered to be of the same size.')
            dimensions = {}

        for i in range(len(data)):
            data[i]['x'] = x_gate[i]
            data[i]['y'] = y_gate[i]
            try:
                data[i]['size'] = dimensions[data[i]['kind']]
            except:
                data[i]['size'] = 1

        total_size = 0
        for i in range(len(data)):
            total_size += data[i]['size']

        print('Partitioning.')

        # x_m = np.mean(x_gate)
        # y_m = np.mean(y_gate)
        # p1 = np.array((x_m, y_m))
        # threshold = 0.50
        # best_miv = 9999999
        # for j in range(10):
        #     target = random.randint(0, len(data) - 1)
        #     x_t = data[target]['x']
        #     y_t = data[target]['y']
        #     p2 = np.array((x_t, y_t))
        #     distance = np.zeros(len(data))
        #     for i in range(len(data)):
        #         data[i]['tier'] = 1
        #         x = data[i]['x']
        #         y = data[i]['y']
        #         p3 = np.array((x, y))
        #         distance[i] = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
        #
        #
        #     tier_size = 0
        #     for i in np.argsort(distance):
        #         tier_size += data[i]['size']
        #         data[i]['tier'] = 0
        #         if tier_size >= threshold * total_size:
        #             break
        #
        #     divided = 0
        #     tier_gate = np.zeros(len(all_gates), dtype=bool)
        #     for i in range(len(data)):
        #         divided += data[i]['tier']
        #         tier_gate[i] = data[i]['tier']
        #     print(divided/gates_num)
        #
        #     # tier_gate = np.zeros(len(all_gates), dtype=bool)
        #     # for gate in all_gates:
        #     #     if gate in all_pins:
        #     #         tier_gate[gate] = 0
        #     #     else:
        #     #         if np.sum(tier_gate) / len(all_gates) > 0.50:
        #     #             tier_gate[gate] = 0
        #     #         else:
        #     #             if gate % 2 == 0:
        #     #                 tier_gate[gate] = 0
        #     #             else:
        #     #                 tier_gate[gate] = 1
        #     # print(np.sum(tier_gate) / len(all_gates))
        #
        #
        #     # np.savetxt('data/'+subject+'_output_scaled_0.txt', output_scaled_0)
        #     # np.savetxt('data/'+subject+'_output_scaled_1.txt', output_scaled_1)
        #
        #     # Find gates that are in different tiers
        #     ok_gates = []
        #     faulty_gates = []
        #     for gate in all_gates:
        #         faulty_flag = 0
        #         current_tier = tier_gate[gate]
        #         for connected_gate in gate_gate[gate]:
        #             if tier_gate[connected_gate] != current_tier:
        #                 faulty_flag = 1
        #                 faulty_gates.append(gate)
        #                 break
        #         if faulty_flag == 0:
        #             ok_gates.append(gate)
        #
        #     # insert MIVs
        #     all_nets = sorted(nets_wpin.keys())
        #     free_gate = all_gates[-1] + 1
        #     miv_number = free_gate
        #     all_mivs = []
        #     nets_wpin_0 = {}
        #     nets_wpin_1 = {}
        #     gates_to_mivs_0 = {}
        #     gates_to_mivs_1 = {}
        #     nets_to_mivs = {}
        #     for net in all_nets:
        #         connected_gates_to_net = np.array(nets_wpin[net])
        #         connected_gates_tiers = tier_gate[connected_gates_to_net]
        #         gates_tier_0 = connected_gates_to_net[~connected_gates_tiers]
        #         gates_tier_1 = connected_gates_to_net[connected_gates_tiers]
        #         if len(gates_tier_0) != 0 and len(gates_tier_1) != 0:
        #             nets_wpin_0[net] = list(np.concatenate((gates_tier_0, [miv_number]), axis=0))
        #             nets_wpin_1[net] = list(np.concatenate((gates_tier_1, [miv_number]), axis=0))
        #             all_mivs.append(miv_number)
        #             gates_to_mivs_0[miv_number] = list(gates_tier_0)
        #             gates_to_mivs_1[miv_number] = list(gates_tier_1)
        #             nets_to_mivs[miv_number] = net
        #             miv_number += 1
        #         elif len(gates_tier_0) == 0:
        #             nets_wpin_1[net] = list(gates_tier_1)
        #         elif len(gates_tier_1) == 0:
        #             nets_wpin_0[net] = list(gates_tier_0)
        #
        #
        #     tier_0_gates = []
        #     tier_1_gates = []
        #     tier_0_nets = []
        #     tier_1_nets = []
        #     for gate in all_gates:
        #         if gate not in all_pins:
        #             connected_nets = gates_wpin[gate]
        #             dummy = [gates_names_original[gate], str(gates_names[gate]), str(len(connected_nets))]
        #             connected_nets_original = [str(nets_name[net]) for net in connected_nets]
        #             dummy.extend(connected_nets_original)
        #             if tier_gate[gate] == 0:
        #                 tier_0_nets.extend(connected_nets)
        #                 tier_0_gates.append(dummy)
        #             if tier_gate[gate] == 1:
        #                 tier_1_gates.append(dummy)
        #     if len(all_mivs) < best_miv:
        #         best_target = target
        #     print(j, target, len(tier_0_gates), len(tier_1_gates), len(all_mivs))
        #     if len(all_mivs) < 12300:
        #         break

        if reinforcement:
            tier_n = 8
            threshold = [0]
            for i in range(tier_n):
                threshold.append(threshold[-1]+1/tier_n)
            print(threshold)

            for i in range(len(data)):
                data[i]['tier'] = 7

            tier_size = 0
            target_tier = 0
            current_threshold = threshold[1]
            flag = 0
            for i in np.argsort(x_gate):
                tier_size += data[i]['size']
                data[i]['tier'] = target_tier
                if tier_size >= current_threshold * total_size:
                    target_tier += 1
                    if target_tier == tier_n - 1:
                        break
                    current_threshold = threshold[target_tier + 1]
            dummy_dict = {k:0 for k in range(tier_n)}
            for i in range(len(data)):
                dummy_dict[data[i]['tier']] += 1
            print(dummy_dict)


            tier_gate = np.zeros(len(all_gates), dtype=int)
            for i in range(len(data)):
                tier_gate[i] = data[i]['tier']


            # insert MIVs
            all_nets = sorted(nets_wpin.keys())
            free_gate = all_gates[-1] + 1
            miv_number = free_gate
            all_mivs = []
            nets_to_mivs_0 = {}
            nets_to_mivs_1 = {}
            nets_to_mivs_2 = {}
            dummy_dict = {}
            for i in range(tier_n):
                for j in range(tier_n):
                    dummy_dict[(i, j)] = 0
            for net in all_nets:
                connected_gates_to_net = np.array(nets_wpin[net])
                connected_gates_tiers = tier_gate[connected_gates_to_net]
                gates_tier_dummy = []
                gates_tier_dummy.append(connected_gates_to_net[connected_gates_tiers == 0])
                gates_tier_dummy.append(connected_gates_to_net[connected_gates_tiers == 1])
                gates_tier_dummy.append(connected_gates_to_net[connected_gates_tiers == 2])
                gates_tier_dummy.append(connected_gates_to_net[connected_gates_tiers == 3])
                gates_tier_dummy.append(connected_gates_to_net[connected_gates_tiers == 4])
                gates_tier_dummy.append(connected_gates_to_net[connected_gates_tiers == 5])
                gates_tier_dummy.append(connected_gates_to_net[connected_gates_tiers == 6])
                gates_tier_dummy.append(connected_gates_to_net[connected_gates_tiers == 7])


                gates_lens = [len(k) for k in gates_tier_dummy]
                # print(gates_lens)
                nonzero_lens = [k for k in range(len(gates_lens)) if gates_lens[k] != 0]
                for i in range(tier_n):
                    if i in nonzero_lens:
                        for j in range(tier_n):
                            if j in nonzero_lens:
                                dummy_dict[(i,j)] += 1
            # print(np.random.random_integers(0,1,tier_n))
            print(dummy_dict)

            if not do_perm:
                tiers = []
                for _ in range(10000):
                    dummy = list(np.random.random_integers(0,1,tier_n))
                    if dummy not in tiers:
                        tiers.append(dummy)
                tiers_cleaned = [t for t in tiers if np.sum(t) == tier_n / 2]
                print(tiers, len(tiers))
                print(tiers_cleaned, len(tiers_cleaned))

                distances = []
                for tier in tiers_cleaned:
                    dummy_distance = 0
                    for i in range(tier_n):
                        for j in range(i+1, tier_n):
                            if tier[j] != tier[i]:
                                dummy_distance += dummy_dict[(i, j)]
                            else:
                                dummy_distance += 10 * dummy_dict[(i, j)]
                    distances.append(dummy_distance)
                print(np.min(distances), np.argmin(distances), tiers_cleaned[np.argmin(distances)])
                best_tier_arrangement = tiers_cleaned[np.argmin(distances)]

            if do_perm:
                from itertools import permutations

                perm = permutations(np.arange(tier_n))

                distances = []
                distance_k = {(0,1): 10, (0,2): 10, (0,3):15, (0,4): 1 , (0,5): 12, (0,6): 12, (0,7): 18, (1,2): 15, (1,3): 10, (1,4):12, (1,5): 1, (1,6):18, (1,7):12, (2,3):10, (2,4):12, (2,5):18, (2,6):1, (2,7):12, (3,4):18,
                                (3,5):12, (3,6):12, (3,7):1, (4,5):10, (4,6):10, (4,7):15, (5,6):15, (5,7):10, (6,7):10}
                all_p = []
                for p in perm:
                    dummy_distance = 0
                    tier = list(p)
                    all_p.append(p)
                    for i in range(tier_n):
                        for j in range(i+1, tier_n):
                            tier_1 = np.min([tier[i], tier[j]])
                            tier_2 = np.max([tier[i], tier[j]])
                            dummy_distance += dummy_dict[(i, j)] * distance_k[(tier_1, tier_2)]
                    distances.append(dummy_distance)
                print(np.min(distances), np.argmin(distances), all_p[np.argmin(distances)])
                best_tier_arrangement = all_p[np.argmin(distances)]
                best_tier_arrangement = [0 if i < 4 else 1 for i in best_tier_arrangement]

            for i in range(len(data)):
                data[i]['tier'] = best_tier_arrangement[data[i]['tier']]
            tier_n = 2

        if tier_n == 2:
            if not reinforcement:
                # tier_n = 2
                threshold = 0.50

                for i in range(len(data)):
                    data[i]['tier'] = 1


                tier_size = 0
                for i in np.argsort(x_gate):
                    tier_size += data[i]['size']
                    data[i]['tier'] = 0
                    if tier_size >= threshold * total_size:
                        break
            divided = 0
            tier_gate = np.zeros(len(all_gates), dtype=bool)
            for i in range(len(data)):
                divided += data[i]['tier']
                tier_gate[i] = data[i]['tier']
            print(divided / gates_num)
            # Find gates that are in different tiers
            ok_gates = []
            faulty_gates = []
            for gate in all_gates:
                if 'FFT128' in subject:
                    if gate == 102928:
                        continue
                faulty_flag = 0
                current_tier = tier_gate[gate]
                for connected_gate in gate_gate[gate]:
                    if 'FFT128' in subject:
                        if gate == 102928:
                            continue
                    if tier_gate[connected_gate] != current_tier:
                        faulty_flag = 1
                        faulty_gates.append(gate)
                        break
                if faulty_flag == 0:
                    ok_gates.append(gate)

            # insert MIVs
            all_nets = sorted(nets_wpin.keys())
            free_gate = all_gates[-1] + 1
            miv_number = free_gate
            all_mivs = []
            nets_wpin_0 = {}
            nets_wpin_1 = {}
            gates_to_mivs_0 = {}
            gates_to_mivs_1 = {}
            nets_to_mivs = {}
            for net in all_nets:
                connected_gates_to_net = np.array(nets_wpin[net])
                connected_gates_tiers = tier_gate[connected_gates_to_net]
                gates_tier_0 = connected_gates_to_net[~connected_gates_tiers]
                gates_tier_1 = connected_gates_to_net[connected_gates_tiers]
                if len(gates_tier_0) != 0 and len(gates_tier_1) != 0:
                    nets_wpin_0[net] = list(np.concatenate((gates_tier_0, [miv_number]), axis=0))
                    nets_wpin_1[net] = list(np.concatenate((gates_tier_1, [miv_number]), axis=0))
                    all_mivs.append(miv_number)
                    gates_to_mivs_0[miv_number] = list(gates_tier_0)
                    gates_to_mivs_1[miv_number] = list(gates_tier_1)
                    nets_to_mivs[miv_number] = net
                    miv_number += 1
                elif len(gates_tier_0) == 0:
                    nets_wpin_1[net] = list(gates_tier_1)
                elif len(gates_tier_1) == 0:
                    nets_wpin_0[net] = list(gates_tier_0)

            tier_0_gates = []
            tier_1_gates = []
            tier_0_nets = []
            tier_1_nets = []
            for gate in all_gates:
                if gate not in all_pins:
                    connected_nets = gates_wpin[gate]
                    dummy = [gates_names_original[gate], str(gates_names[gate]), str(len(connected_nets))]
                    connected_nets_original = [str(nets_name[net]) for net in connected_nets]
                    dummy.extend(connected_nets_original)
                    if tier_gate[gate] == 0:
                        tier_0_nets.extend(connected_nets)
                        tier_0_gates.append(dummy)
                    if tier_gate[gate] == 1:
                        tier_1_gates.append(dummy)
            print(len(tier_0_gates), len(tier_1_gates), len(all_mivs))

            file1 = open(out_dir + str(tier_n)+'tier_0_' + subject + '.v', "w+")
            for i in raw_lines[0:cut_off]:
                file1.write(i)

            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 0:
                    for j in range(line_index[i][0], line_index[i][1]):
                        file1.write(raw_lines[j])

            for miv in all_mivs:
                con = nets_name[nets_to_mivs[miv]]
                if con.split('[')[0] in wires:
                    file1.write(
                        '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                elif con[0] == '\\':
                    file1.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                else:
                    file1.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')
            file1.write('endmodule\n\n')

            file1.close()

            file2 = open(out_dir + str(tier_n)+'tier_1_' + subject + '.v', "w+")
            for i in raw_lines[0:cut_off]:
                file2.write(i)

            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 1:
                    for j in range(line_index[i][0], line_index[i][1]):
                        file2.write(raw_lines[j])

            for miv in all_mivs:
                con = nets_name[nets_to_mivs[miv]]
                if con.split('[')[0] in wires:
                    file2.write(
                        '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                elif con[0] == '\\':
                    file2.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                else:
                    file2.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')
            file2.write('endmodule\n\n')

            file2.close()

            print("Number of MIVs for ", subject, " is ", len(all_mivs))


            file3 = open(out_dir + str(tier_n)+'tier_0_' + subject + '.txt', "w+")
            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 0:
                    file3.write(current['name']+'\n')
            file3.close()

            file4 = open(out_dir + str(tier_n)+'tier_1_' + subject + '.txt', "w+")
            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 1:
                    file4.write(current['name']+'\n')
            file4.close()

            cons = []
            for miv in all_mivs:
                con = nets_name[nets_to_mivs[miv]]
                if con not in cons:
                    cons.append(con)

            file5 = open(out_dir + str(tier_n)+'tier_cut_nets_' + subject + '.txt', "w+")
            for i in range(len(cons)):
                file5.write(cons[i]+'\n')
            file5.close()



        # 3 tiers
        if tier_n == 3:
            # tier_n = 3
            threshold_1 = 0.33333333333
            threshold_2 = 0.66666666

            for i in range(len(data)):
                data[i]['tier'] = 2

            tier_size = 0
            target_tier = 0
            threshold = threshold_1
            flag = 0
            for i in np.argsort(x_gate):
                tier_size += data[i]['size']
                data[i]['tier'] = target_tier
                if tier_size >= threshold * total_size:
                    if flag == 1:
                        break
                    target_tier = 1
                    threshold = threshold_2
                    flag = 1
            divided_0 = 0
            divided_1 = 0
            divided_2 = 0
            tier_gate = np.zeros(len(all_gates), dtype=int)
            for i in range(len(data)):
                if data[i]['tier'] == 1:
                    divided_1 += 1
                elif data[i]['tier'] == 2:
                    divided_2 += 1
                else:
                    divided_0 += 1

                tier_gate[i] = data[i]['tier']
            print(divided_0 / gates_num)
            print(divided_1 / gates_num)
            print(divided_2 / gates_num)


            # insert MIVs
            all_nets = sorted(nets_wpin.keys())
            free_gate = all_gates[-1] + 1
            miv_number = free_gate
            all_mivs = []
            nets_to_mivs_0 = {}
            nets_to_mivs_1 = {}
            for net in all_nets:
                connected_gates_to_net = np.array(nets_wpin[net])
                connected_gates_tiers = tier_gate[connected_gates_to_net]
                gates_tier_0 = connected_gates_to_net[connected_gates_tiers == 0]
                gates_tier_1 = connected_gates_to_net[connected_gates_tiers == 1]
                gates_tier_2 = connected_gates_to_net[connected_gates_tiers == 2]

                if len(gates_tier_0) != 0 and len(gates_tier_1) != 0 and len(gates_tier_2) == 0:
                    nets_to_mivs_0[miv_number] = net
                    all_mivs.append(miv_number)
                    miv_number += 1
                elif len(gates_tier_0) == 0 and len(gates_tier_1) != 0 and len(gates_tier_2) != 0:
                    nets_to_mivs_1[miv_number] = net
                    all_mivs.append(miv_number)
                    miv_number += 1
                elif len(gates_tier_0) != 0 and len(gates_tier_1) != 0 and len(gates_tier_2) != 0:
                    nets_to_mivs_0[miv_number] = net
                    all_mivs.append(miv_number)
                    miv_number += 1
                    nets_to_mivs_1[miv_number] = net
                    all_mivs.append(miv_number)
                    miv_number += 1
            tier_0_gates = []
            tier_1_gates = []
            tier_2_gates = []
            tier_0_nets = []
            tier_1_nets = []
            for gate in all_gates:
                if gate not in all_pins:
                    connected_nets = gates_wpin[gate]
                    dummy = [gates_names_original[gate], str(gates_names[gate]), str(len(connected_nets))]
                    connected_nets_original = [str(nets_name[net]) for net in connected_nets]
                    dummy.extend(connected_nets_original)
                    if tier_gate[gate] == 0:
                        tier_0_nets.extend(connected_nets)
                        tier_0_gates.append(dummy)
                    if tier_gate[gate] == 1:
                        tier_1_gates.append(dummy)
                    if tier_gate[gate] == 2:
                        tier_2_gates.append(dummy)
            print(len(tier_0_gates), len(tier_1_gates), len(tier_2_gates), len(all_mivs))

            file1 = open(out_dir + str(tier_n)+'tier_0_' + subject + '.v', "w+")
            for i in raw_lines[0:cut_off]:
                file1.write(i)

            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 0:
                    for j in range(line_index[i][0], line_index[i][1]):
                        file1.write(raw_lines[j])

            for miv in list(nets_to_mivs_0.keys()):
                con = nets_name[nets_to_mivs_0[miv]]
                if con.split('[')[0] in wires:
                    file1.write(
                        '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                elif con[0] == '\\':
                    file1.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                else:
                    file1.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')
            file1.write('endmodule\n\n')

            file1.close()

            file2 = open(out_dir + str(tier_n)+'tier_1_' + subject + '.v', "w+")
            for i in raw_lines[0:cut_off]:
                file2.write(i)

            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 1:
                    for j in range(line_index[i][0], line_index[i][1]):
                        file2.write(raw_lines[j])

            for miv in list(nets_to_mivs_1.keys()):
                con = nets_name[nets_to_mivs_1[miv]]
                if con.split('[')[0] in wires:
                    file2.write(
                        '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                elif con[0] == '\\':
                    file2.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                else:
                    file2.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')

            for miv in list(nets_to_mivs_0.keys()):
                con = nets_name[nets_to_mivs_0[miv]]
                if con.split('[')[0] in wires:
                    file2.write(
                        '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                elif con[0] == '\\':
                    file2.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                else:
                    file2.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')
            file2.write('endmodule\n\n')

            file2.close()

            file3 = open(out_dir + str(tier_n)+'tier_2_' + subject + '.v', "w+")
            for i in raw_lines[0:cut_off]:
                file3.write(i)

            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 2:
                    for j in range(line_index[i][0], line_index[i][1]):
                        file3.write(raw_lines[j])

            for miv in list(nets_to_mivs_1.keys()):
                con = nets_name[nets_to_mivs_1[miv]]
                if con.split('[')[0] in wires:
                    file3.write(
                        '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                elif con[0] == '\\':
                    file3.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                else:
                    file3.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')
            file3.write('endmodule\n\n')

            file3.close()


            print("Number of MIVs for ", subject, " is ", len(all_mivs))




        # 4 tiers
        if tier_n == 4:
            # tier_n = 4
            threshold_1 = 0.25
            threshold_2 = 0.5
            threshold_3 = 0.75


            for i in range(len(data)):
                data[i]['tier'] = 3

            tier_size = 0
            target_tier = 0
            threshold = threshold_1
            flag = 0
            for i in np.argsort(x_gate):
                tier_size += data[i]['size']
                data[i]['tier'] = target_tier
                if tier_size >= threshold * total_size:
                    if flag == 2:
                        break
                    if flag == 1:
                        target_tier = 2
                        threshold = threshold_3
                        flag = 2
                    if flag == 0:
                        target_tier = 1
                        threshold = threshold_2
                        flag = 1
            divided_0 = 0
            divided_1 = 0
            divided_2 = 0
            divided_3 = 0
            tier_gate = np.zeros(len(all_gates), dtype=int)
            for i in range(len(data)):
                if data[i]['tier'] == 1:
                    divided_1 += 1
                elif data[i]['tier'] == 2:
                    divided_2 += 1
                elif data[i]['tier'] == 3:
                    divided_3 += 1
                else:
                    divided_0 += 1

                tier_gate[i] = data[i]['tier']
            print(divided_0 / gates_num)
            print(divided_1 / gates_num)
            print(divided_2 / gates_num)
            print(divided_3 / gates_num)


            # insert MIVs
            all_nets = sorted(nets_wpin.keys())
            free_gate = all_gates[-1] + 1
            miv_number = free_gate
            all_mivs = []
            nets_to_mivs_0 = {}
            nets_to_mivs_1 = {}
            nets_to_mivs_2 = {}

            for net in all_nets:
                connected_gates_to_net = np.array(nets_wpin[net])
                connected_gates_tiers = tier_gate[connected_gates_to_net]
                gates_tier_0 = connected_gates_to_net[connected_gates_tiers == 0]
                gates_tier_1 = connected_gates_to_net[connected_gates_tiers == 1]
                gates_tier_2 = connected_gates_to_net[connected_gates_tiers == 2]
                gates_tier_3 = connected_gates_to_net[connected_gates_tiers == 3]
                smallest_tier = -1
                largest_tier = -1
                if len(gates_tier_0) != 0:
                    smallest_tier = 0
                elif len(gates_tier_1) != 0:
                    smallest_tier = 1
                elif len(gates_tier_2) != 0:
                    smallest_tier = 2
                elif len(gates_tier_3) != 0:
                    smallest_tier = 3

                if len(gates_tier_3) != 0:
                    largest_tier = 3
                elif len(gates_tier_2) != 0:
                    largest_tier = 2
                elif len(gates_tier_1) != 0:
                    largest_tier = 1
                elif len(gates_tier_0) != 0:
                    largest_tier = 0

                if smallest_tier == largest_tier:
                    continue
                else:
                    if smallest_tier == 0:
                        nets_to_mivs_0[miv_number] = net
                        all_mivs.append(miv_number)
                        miv_number += 1
                        if largest_tier >= 2:
                            nets_to_mivs_1[miv_number] = net
                            all_mivs.append(miv_number)
                            miv_number += 1
                        if largest_tier == 3:
                            nets_to_mivs_2[miv_number] = net
                            all_mivs.append(miv_number)
                            miv_number += 1
                    elif smallest_tier == 1:
                        nets_to_mivs_1[miv_number] = net
                        all_mivs.append(miv_number)
                        miv_number += 1
                        if largest_tier == 3:
                            nets_to_mivs_2[miv_number] = net
                            all_mivs.append(miv_number)
                            miv_number += 1
                    elif smallest_tier == 2:
                        nets_to_mivs_2[miv_number] = net
                        all_mivs.append(miv_number)
                        miv_number += 1

                # if len(gates_tier_0) != 0 and len(gates_tier_1) != 0 and len(gates_tier_2) == 0:
                #     nets_to_mivs_0[miv_number] = net
                #     all_mivs.append(miv_number)
                #     miv_number += 1
                # elif len(gates_tier_0) == 0 and len(gates_tier_1) != 0 and len(gates_tier_2) != 0:
                #     nets_to_mivs_1[miv_number] = net
                #     all_mivs.append(miv_number)
                #     miv_number += 1
                # elif len(gates_tier_0) != 0 and len(gates_tier_1) != 0 and len(gates_tier_2) != 0:
                #     nets_to_mivs_0[miv_number] = net
                #     all_mivs.append(miv_number)
                #     miv_number += 1
                #     nets_to_mivs_1[miv_number] = net
                #     all_mivs.append(miv_number)
                #     miv_number += 1
            tier_0_gates = []
            tier_1_gates = []
            tier_2_gates = []
            tier_3_gates = []

            tier_0_nets = []
            tier_1_nets = []
            tier_2_nets = []

            for gate in all_gates:
                if gate not in all_pins:
                    connected_nets = gates_wpin[gate]
                    dummy = [gates_names_original[gate], str(gates_names[gate]), str(len(connected_nets))]
                    connected_nets_original = [str(nets_name[net]) for net in connected_nets]
                    dummy.extend(connected_nets_original)
                    if tier_gate[gate] == 0:
                        tier_0_nets.extend(connected_nets)
                        tier_0_gates.append(dummy)
                    if tier_gate[gate] == 1:
                        tier_1_gates.append(dummy)
                    if tier_gate[gate] == 2:
                        tier_2_gates.append(dummy)
                    if tier_gate[gate] == 3:
                        tier_3_gates.append(dummy)
            print(len(tier_0_gates), len(tier_1_gates), len(tier_2_gates), len(tier_3_gates), len(all_mivs))

            file1 = open(out_dir + str(tier_n)+'tier_0_' + subject + '.v', "w+")
            for i in raw_lines[0:cut_off]:
                file1.write(i)

            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 0:
                    for j in range(line_index[i][0], line_index[i][1]):
                        file1.write(raw_lines[j])

            for miv in list(nets_to_mivs_0.keys()):
                con = nets_name[nets_to_mivs_0[miv]]
                if con.split('[')[0] in wires:
                    file1.write(
                        '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                elif con[0] == '\\':
                    file1.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                else:
                    file1.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')
            file1.write('endmodule\n\n')

            file1.close()

            file2 = open(out_dir + str(tier_n)+'tier_1_' + subject + '.v', "w+")
            for i in raw_lines[0:cut_off]:
                file2.write(i)

            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 1:
                    for j in range(line_index[i][0], line_index[i][1]):
                        file2.write(raw_lines[j])

            for miv in list(nets_to_mivs_1.keys()):
                con = nets_name[nets_to_mivs_1[miv]]
                if con.split('[')[0] in wires:
                    file2.write(
                        '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                elif con[0] == '\\':
                    file2.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                else:
                    file2.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')

            for miv in list(nets_to_mivs_0.keys()):
                con = nets_name[nets_to_mivs_0[miv]]
                if con.split('[')[0] in wires:
                    file2.write(
                        '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                elif con[0] == '\\':
                    file2.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                else:
                    file2.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')
            file2.write('endmodule\n\n')

            file2.close()

            file3 = open(out_dir + str(tier_n)+'tier_2_' + subject + '.v', "w+")
            for i in raw_lines[0:cut_off]:
                file3.write(i)

            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 2:
                    for j in range(line_index[i][0], line_index[i][1]):
                        file3.write(raw_lines[j])

            for miv in list(nets_to_mivs_1.keys()):
                con = nets_name[nets_to_mivs_1[miv]]
                if con.split('[')[0] in wires:
                    file3.write(
                        '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                elif con[0] == '\\':
                    file3.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                else:
                    file3.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')
            for miv in list(nets_to_mivs_2.keys()):
                con = nets_name[nets_to_mivs_2[miv]]
                if con.split('[')[0] in wires:
                    file3.write(
                        '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                elif con[0] == '\\':
                    file3.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                else:
                    file3.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')
            file3.write('endmodule\n\n')

            file3.close()
            file4 = open(out_dir + str(tier_n)+'tier_3_' + subject + '.v', "w+")
            for i in raw_lines[0:cut_off]:
                file4.write(i)

            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 3:
                    for j in range(line_index[i][0], line_index[i][1]):
                        file4.write(raw_lines[j])

            for miv in list(nets_to_mivs_2.keys()):
                con = nets_name[nets_to_mivs_2[miv]]
                if con.split('[')[0] in wires:
                    file4.write(
                        '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                elif con[0] == '\\':
                    file4.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                else:
                    file4.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')
            file4.write('endmodule\n\n')

            file4.close()


            print("Number of MIVs for ", subject, " is ", len(all_mivs))


            file5 = open(out_dir + str(tier_n)+'tier_0_' + subject + '.txt', "w+")
            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 0:
                    file5.write(current['name']+'\n')
            file5.close()

            file6 = open(out_dir + str(tier_n)+'tier_1_' + subject + '.txt', "w+")
            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 1:
                    file6.write(current['name']+'\n')
            file6.close()

            file7 = open(out_dir + str(tier_n)+'tier_2_' + subject + '.txt', "w+")
            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 2:
                    file7.write(current['name']+'\n')
            file7.close()

            file8 = open(out_dir + str(tier_n)+'tier_3_' + subject + '.txt', "w+")
            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 3:
                    file8.write(current['name']+'\n')
            file8.close()

            cons = []
            for miv in list(nets_to_mivs_0.keys()):
                con = nets_name[nets_to_mivs_0[miv]]
                if con not in cons:
                    cons.append(con)

            file9 = open(out_dir + str(tier_n)+'tier_0_cut_nets_' + subject + '.txt', "w+")
            for i in range(len(cons)):
                file9.write(cons[i]+'\n')
            file9.close()

            cons = []
            for miv in list(nets_to_mivs_1.keys()):
                con = nets_name[nets_to_mivs_1[miv]]
                if con not in cons:
                    cons.append(con)

            file10 = open(out_dir + str(tier_n)+'tier_1_cut_nets_' + subject + '.txt', "w+")
            for i in range(len(cons)):
                file10.write(cons[i]+'\n')
            file10.close()

            cons = []
            for miv in list(nets_to_mivs_2.keys()):
                con = nets_name[nets_to_mivs_2[miv]]
                if con not in cons:
                    cons.append(con)

            file11 = open(out_dir + str(tier_n)+'tier_2_cut_nets_' + subject + '.txt', "w+")
            for i in range(len(cons)):
                file11.write(cons[i]+'\n')
            file11.close()


        # 5 tiers
        if tier_n == 5:
            # tier_n = 5
            threshold_1 = 0.2
            threshold_2 = 0.4
            threshold_3 = 0.6
            threshold_4 = 0.8


            for i in range(len(data)):
                data[i]['tier'] = 4

            tier_size = 0
            target_tier = 0
            threshold = threshold_1
            flag = 0
            for i in np.argsort(x_gate):
                tier_size += data[i]['size']
                data[i]['tier'] = target_tier
                if tier_size >= threshold * total_size:
                    if flag == 3:
                        break
                    if flag == 2:
                        target_tier = 3
                        threshold = threshold_4
                        flag = 3
                    if flag == 1:
                        target_tier = 2
                        threshold = threshold_3
                        flag = 2
                    if flag == 0:
                        target_tier = 1
                        threshold = threshold_2
                        flag = 1
            divided_0 = 0
            divided_1 = 0
            divided_2 = 0
            divided_3 = 0
            divided_4 = 0
            tier_gate = np.zeros(len(all_gates), dtype=int)
            for i in range(len(data)):
                if data[i]['tier'] == 1:
                    divided_1 += 1
                elif data[i]['tier'] == 2:
                    divided_2 += 1
                elif data[i]['tier'] == 3:
                    divided_3 += 1
                elif data[i]['tier'] == 4:
                    divided_4 += 1
                else:
                    divided_0 += 1

                tier_gate[i] = data[i]['tier']
            print(divided_0 / gates_num)
            print(divided_1 / gates_num)
            print(divided_2 / gates_num)
            print(divided_3 / gates_num)
            print(divided_4 / gates_num)


            # insert MIVs
            all_nets = sorted(nets_wpin.keys())
            free_gate = all_gates[-1] + 1
            miv_number = free_gate
            all_mivs = []
            nets_to_mivs_0 = {}
            nets_to_mivs_1 = {}
            nets_to_mivs_2 = {}
            nets_to_mivs_3 = {}

            for net in all_nets:
                connected_gates_to_net = np.array(nets_wpin[net])
                connected_gates_tiers = tier_gate[connected_gates_to_net]
                gates_tier_0 = connected_gates_to_net[connected_gates_tiers == 0]
                gates_tier_1 = connected_gates_to_net[connected_gates_tiers == 1]
                gates_tier_2 = connected_gates_to_net[connected_gates_tiers == 2]
                gates_tier_3 = connected_gates_to_net[connected_gates_tiers == 3]
                gates_tier_4 = connected_gates_to_net[connected_gates_tiers == 4]
                smallest_tier = -1
                largest_tier = -1
                if len(gates_tier_0) != 0:
                    smallest_tier = 0
                elif len(gates_tier_1) != 0:
                    smallest_tier = 1
                elif len(gates_tier_2) != 0:
                    smallest_tier = 2
                elif len(gates_tier_3) != 0:
                    smallest_tier = 3
                elif len(gates_tier_4) != 0:
                    smallest_tier = 4

                if len(gates_tier_4) != 0:
                    largest_tier = 4
                elif len(gates_tier_3) != 0:
                    largest_tier = 3
                elif len(gates_tier_2) != 0:
                    largest_tier = 2
                elif len(gates_tier_1) != 0:
                    largest_tier = 1
                elif len(gates_tier_0) != 0:
                    largest_tier = 0

                if smallest_tier == largest_tier:
                    continue
                else:
                    if smallest_tier == 0:
                        nets_to_mivs_0[miv_number] = net
                        all_mivs.append(miv_number)
                        miv_number += 1
                        if largest_tier >= 2:
                            nets_to_mivs_1[miv_number] = net
                            all_mivs.append(miv_number)
                            miv_number += 1
                        if largest_tier >= 3:
                            nets_to_mivs_2[miv_number] = net
                            all_mivs.append(miv_number)
                            miv_number += 1
                        if largest_tier == 4:
                            nets_to_mivs_3[miv_number] = net
                            all_mivs.append(miv_number)
                            miv_number += 1
                    elif smallest_tier == 1:
                        nets_to_mivs_1[miv_number] = net
                        all_mivs.append(miv_number)
                        miv_number += 1
                        if largest_tier >= 3:
                            nets_to_mivs_2[miv_number] = net
                            all_mivs.append(miv_number)
                            miv_number += 1
                        if largest_tier == 4:
                            nets_to_mivs_3[miv_number] = net
                            all_mivs.append(miv_number)
                            miv_number += 1
                    elif smallest_tier == 2:
                        nets_to_mivs_2[miv_number] = net
                        all_mivs.append(miv_number)
                        miv_number += 1
                        if largest_tier == 4:
                            nets_to_mivs_3[miv_number] = net
                            all_mivs.append(miv_number)
                            miv_number += 1
                    elif smallest_tier == 3:
                        nets_to_mivs_3[miv_number] = net
                        all_mivs.append(miv_number)
                        miv_number += 1

            tier_0_gates = []
            tier_1_gates = []
            tier_2_gates = []
            tier_3_gates = []
            tier_4_gates = []

            tier_0_nets = []
            tier_1_nets = []
            tier_2_nets = []
            tier_3_nets = []

            for gate in all_gates:
                if gate not in all_pins:
                    connected_nets = gates_wpin[gate]
                    dummy = [gates_names_original[gate], str(gates_names[gate]), str(len(connected_nets))]
                    connected_nets_original = [str(nets_name[net]) for net in connected_nets]
                    dummy.extend(connected_nets_original)
                    if tier_gate[gate] == 0:
                        tier_0_nets.extend(connected_nets)
                        tier_0_gates.append(dummy)
                    if tier_gate[gate] == 1:
                        tier_1_gates.append(dummy)
                    if tier_gate[gate] == 2:
                        tier_2_gates.append(dummy)
                    if tier_gate[gate] == 3:
                        tier_3_gates.append(dummy)
                    if tier_gate[gate] == 4:
                        tier_4_gates.append(dummy)
            print(len(tier_0_gates), len(tier_1_gates), len(tier_2_gates), len(tier_3_gates), len(tier_4_gates), len(all_mivs))

            file1 = open(out_dir + str(tier_n)+'tier_0_' + subject + '.v', "w+")
            for i in raw_lines[0:cut_off]:
                file1.write(i)

            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 0:
                    for j in range(line_index[i][0], line_index[i][1]):
                        file1.write(raw_lines[j])

            for miv in list(nets_to_mivs_0.keys()):
                con = nets_name[nets_to_mivs_0[miv]]
                if con.split('[')[0] in wires:
                    file1.write(
                        '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                elif con[0] == '\\':
                    file1.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                else:
                    file1.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')
            file1.write('endmodule\n\n')

            file1.close()

            file2 = open(out_dir + str(tier_n)+'tier_1_' + subject + '.v', "w+")
            for i in raw_lines[0:cut_off]:
                file2.write(i)

            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 1:
                    for j in range(line_index[i][0], line_index[i][1]):
                        file2.write(raw_lines[j])

            for miv in list(nets_to_mivs_1.keys()):
                con = nets_name[nets_to_mivs_1[miv]]
                if con.split('[')[0] in wires:
                    file2.write(
                        '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                elif con[0] == '\\':
                    file2.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                else:
                    file2.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')

            for miv in list(nets_to_mivs_0.keys()):
                con = nets_name[nets_to_mivs_0[miv]]
                if con.split('[')[0] in wires:
                    file2.write(
                        '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                elif con[0] == '\\':
                    file2.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                else:
                    file2.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')
            file2.write('endmodule\n\n')

            file2.close()

            file3 = open(out_dir + str(tier_n)+'tier_2_' + subject + '.v', "w+")
            for i in raw_lines[0:cut_off]:
                file3.write(i)

            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 2:
                    for j in range(line_index[i][0], line_index[i][1]):
                        file3.write(raw_lines[j])

            for miv in list(nets_to_mivs_1.keys()):
                con = nets_name[nets_to_mivs_1[miv]]
                if con.split('[')[0] in wires:
                    file3.write(
                        '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                elif con[0] == '\\':
                    file3.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                else:
                    file3.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')
            for miv in list(nets_to_mivs_2.keys()):
                con = nets_name[nets_to_mivs_2[miv]]
                if con.split('[')[0] in wires:
                    file3.write(
                        '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                elif con[0] == '\\':
                    file3.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                else:
                    file3.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')
            file3.write('endmodule\n\n')

            file3.close()
            file4 = open(out_dir + str(tier_n)+'tier_3_' + subject + '.v', "w+")
            for i in raw_lines[0:cut_off]:
                file4.write(i)

            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 3:
                    for j in range(line_index[i][0], line_index[i][1]):
                        file4.write(raw_lines[j])

            for miv in list(nets_to_mivs_2.keys()):
                con = nets_name[nets_to_mivs_2[miv]]
                if con.split('[')[0] in wires:
                    file4.write(
                        '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                elif con[0] == '\\':
                    file4.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                else:
                    file4.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')
            file4.write('endmodule\n\n')

            file4.close()

            file40 = open(out_dir + str(tier_n)+'tier_4_' + subject + '.v', "w+")
            for i in raw_lines[0:cut_off]:
                file40.write(i)

            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 4:
                    for j in range(line_index[i][0], line_index[i][1]):
                        file40.write(raw_lines[j])

            for miv in list(nets_to_mivs_3.keys()):
                con = nets_name[nets_to_mivs_3[miv]]
                if con.split('[')[0] in wires:
                    file40.write(
                        '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                elif con[0] == '\\':
                    file40.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                else:
                    file40.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')
            file40.write('endmodule\n\n')

            file40.close()

            print("Number of MIVs for ", subject, " is ", len(all_mivs))


            file5 = open(out_dir + str(tier_n)+'tier_0_' + subject + '.txt', "w+")
            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 0:
                    file5.write(current['name']+'\n')
            file5.close()

            file6 = open(out_dir + str(tier_n)+'tier_1_' + subject + '.txt', "w+")
            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 1:
                    file6.write(current['name']+'\n')
            file6.close()

            file7 = open(out_dir + str(tier_n)+'tier_2_' + subject + '.txt', "w+")
            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 2:
                    file7.write(current['name']+'\n')
            file7.close()

            file8 = open(out_dir + str(tier_n)+'tier_3_' + subject + '.txt', "w+")
            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 3:
                    file8.write(current['name']+'\n')
            file8.close()

            file80 = open(out_dir + str(tier_n)+'tier_4_' + subject + '.txt', "w+")
            for i in range(len(data)):
                current = data[i]
                if current['tier'] == 4:
                    file80.write(current['name']+'\n')
            file80.close()


            cons = []
            for miv in list(nets_to_mivs_0.keys()):
                con = nets_name[nets_to_mivs_0[miv]]
                if con not in cons:
                    cons.append(con)

            file9 = open(out_dir + str(tier_n)+'tier_0_cut_nets_' + subject + '.txt', "w+")
            for i in range(len(cons)):
                file9.write(cons[i]+'\n')
            file9.close()

            cons = []
            for miv in list(nets_to_mivs_1.keys()):
                con = nets_name[nets_to_mivs_1[miv]]
                if con not in cons:
                    cons.append(con)

            file10 = open(out_dir + str(tier_n)+'tier_1_cut_nets_' + subject + '.txt', "w+")
            for i in range(len(cons)):
                file10.write(cons[i]+'\n')
            file10.close()

            cons = []
            for miv in list(nets_to_mivs_2.keys()):
                con = nets_name[nets_to_mivs_2[miv]]
                if con not in cons:
                    cons.append(con)

            file11 = open(out_dir + str(tier_n)+'tier_2_cut_nets_' + subject + '.txt', "w+")
            for i in range(len(cons)):
                file11.write(cons[i]+'\n')
            file11.close()

            cons = []
            for miv in list(nets_to_mivs_3.keys()):
                con = nets_name[nets_to_mivs_3[miv]]
                if con not in cons:
                    cons.append(con)

            file12 = open(out_dir + str(tier_n)+'tier_3_cut_nets_' + subject + '.txt', "w+")
            for i in range(len(cons)):
                file12.write(cons[i]+'\n')
            file12.close()
