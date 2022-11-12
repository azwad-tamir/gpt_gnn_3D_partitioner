import sys

import pandas as pd

from example_reddit.GPT_GNN.data import *
from example_reddit.GPT_GNN.model import *
from warnings import filterwarnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from kneed import KneeLocator

filterwarnings("ignore")

import argparse

# Parsing command line arguments
# data_dir_root = sys.argv[1]
# subject_name = sys.argv[2]
# out_dir = sys.argv[3]


def node_classification_sample(seed, nodes, time_range):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers) and their time.
    '''
    np.random.seed(seed)
    samp_nodes = np.random.choice(nodes, args.batch_size, replace=False)
    feature, times, edge_list, _, texts = sample_subgraph(graph, time_range, \
                                                          inp={target_type: np.concatenate(
                                                              [samp_nodes, np.ones(args.batch_size)]).reshape(2,
                                                                                                              -1).transpose()}, \
                                                          sampled_depth=args.sample_depth,
                                                          sampled_number=args.sample_width,
                                                          feature_extractor=feature_reddit)

    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
        to_torch(feature, times, edge_list, graph)

    x_ids = np.arange(args.batch_size)
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, graph.y[samp_nodes]


def all_nodes_sample(seed, nodes, time_range):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers) and their time.
    '''
    samp_nodes = nodes
    feature, times, edge_list, _, texts = sample_subgraph(graph, time_range, \
                                                          inp={target_type: np.concatenate(
                                                              [samp_nodes, np.ones(len(samp_nodes))]).reshape(2,
                                                                                                              -1).transpose()}, \
                                                          sampled_depth=args.sample_depth,
                                                          sampled_number=args.sample_width,
                                                          feature_extractor=feature_reddit)

    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
        to_torch(feature, times, edge_list, graph)

    x_ids = np.arange(len(samp_nodes))
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, graph.y[samp_nodes]


def prepare_data(pool):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    for batch_id in np.arange(args.n_batch):
        p = pool.apply_async(node_classification_sample, args=(randint(), train_target_nodes, {1: True}))
        jobs.append(p)
    p = pool.apply_async(node_classification_sample, args=(randint(), valid_target_nodes, {1: True}))
    jobs.append(p)
    return jobs


class unique_element:
    def __init__(self,value,occurrences):
        self.value = value
        self.occurrences = occurrences


def perm_unique(elements):
    eset=set(elements)
    listunique = [unique_element(i,elements.count(i)) for i in eset]
    u=len(elements)
    return perm_unique_helper(listunique,[0]*u,u-1)


def perm_unique_helper(listunique,result_list,d):
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d]=i.value
                i.occurrences-=1
                for g in  perm_unique_helper(listunique,result_list,d-1):
                    yield g
                i.occurrences+=1




timings = [0.7]
utils = [0.7]
# placement_types = ['none', 'aq', 'icc2', 'both']
placement_types = ['icc2']
timing_types = ['none', 'included']
# hop_types = ['none', 'included']
hop_types = ['included']
partition_types = ['minmiv', 'middle']
subject = 'dhm_mini_mapped'
for util in utils:
    for timing in timings:
        for placement_type in placement_types:
            for timing_type in timing_types:
                for hop_type in hop_types:
                    print(util, timing, placement_type, timing_type, hop_type)

                    # data_dir_root = sys.argv[1]
                    # subject_name = sys.argv[2]
                    # out_dir = sys.argv[3]
                    parser = argparse.ArgumentParser(description='Fine-Tuning on Reddit classification task')
                    parser.add_argument('--data_dir_root', type=str, default='data',
                                        help='The address of the initial data')
                    parser.add_argument('--subject_name', type=str, default='aes_cipher',
                                        help='The name of the circuit')
                    parser.add_argument('--out_dir', type=str, default='results',
                                        help='The output directory')

                    args = parser.parse_args()
                    data_dir_root = args.data_dir_root
                    subject_name = args.subject_name
                    out_dir = args.out_dir

                    netlist_dir = data_dir_root + '/' + subject_name + '/' + subject_name + '_' + str(util).split('.')[0] + 'P' + \
                                  str(util).split('.')[1] + '/netlists/' + subject_name + '__' + 'CLIB_NAME-saed32__CLK_PERIOD-' + \
                                  str(timing) + '00__CORE_UTIL-' + str(util) + '00/' + subject_name + '_mapped.v'
                    lef_dir = data_dir_root + '/' + subject_name + '/celllist.txt'
                    # out_dir = 'results/'
                    try:
                        os.makedirs(out_dir + subject + '/' + str(util) + '/' + str(timing) + '/')
                    except OSError as error:
                        print(error)
                    current_outdir = out_dir + subject + '/' + str(util) + '/' + str(timing) + '/' + str(timing) + '_util' + str(
                        util) + '_placement_' + placement_type + '_hop_' + hop_type + '_timing_' + timing_type


                    '''
                        Dataset arguments
                    '''
                    # parser.add_argument('--data_dir', type=str, default='data/graph_reddit_netlist.pk',
                    #                     help='The address of preprocessed graph.')
                    parser.add_argument('--data_dir', type=str, default='graphs/graph_' + subject + '_' + str(timing) + '_util' + str(
                        util) + '_placement_' + placement_type + '_hop_' + hop_type + '_timing_' + timing_type + '.pk',
                                        help='The address of preprocessed graph.')
                    parser.add_argument('--use_pretrain', default=True,  help='Whether to use pre-trained model', action='store_true')
                    parser.add_argument('--pretrain_model_dir', type=str,
                                        default='built_models/model_' + subject + '_' + str(timing) + '_util' + str(
                                            util) + '_placement_' + placement_type + '_hop_' + hop_type + '_timing_' + timing_type,
                                        help='The address for storing the pre-trained models.')
                    # parser.add_argument('--model_dir', type=str, default='built_models/gpt_all_reddit_finetuned',
                    #                     help='The address for storing the models and optimization results.')
                    # parser.add_argument('--model_dir', type=str, default='built_models/dhm_mini_gpt_all_reddit_finetuned',
                    #                     help='The address for storing the models and optimization results.')
                    parser.add_argument('--task_name', type=str, default='reddit',
                                        help='The name of the stored models and optimization results.')
                    parser.add_argument('--cuda', type=int, default=-1,
                                        help='Avaiable GPU ID')
                    parser.add_argument('--sample_depth', type=int, default=6,
                                        help='How many numbers to sample the graph')
                    parser.add_argument('--sample_width', type=int, default=128,
                                        help='How many nodes to be sampled per layer per type')
                    '''
                       Model arguments 
                    '''
                    parser.add_argument('--conv_name', type=str, default='hgt',
                                        choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                                        help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
                    parser.add_argument('--n_hid', type=int, default=400,
                                        help='Number of hidden dimension')
                    parser.add_argument('--n_heads', type=int, default=8,
                                        help='Number of attention head')
                    parser.add_argument('--n_layers', type=int, default=3,
                                        help='Number of GNN layers')
                    parser.add_argument('--prev_norm', help='Whether to add layer-norm on the previous layers', action='store_true')
                    parser.add_argument('--last_norm', help='Whether to add layer-norm on the last layers',     action='store_true')
                    parser.add_argument('--dropout', type=int, default=0.2,
                                        help='Dropout ratio')


                    '''
                        Optimization arguments
                    '''
                    parser.add_argument('--optimizer', type=str, default='adamw',
                                        choices=['adamw', 'adam', 'sgd', 'adagrad'],
                                        help='optimizer to use.')
                    parser.add_argument('--scheduler', type=str, default='cosine',
                                        help='Name of learning rate scheduler.' , choices=['cycle', 'cosine'])
                    parser.add_argument('--data_percentage', type=int, default=0.1,
                                        help='Percentage of training and validation data to use')
                    parser.add_argument('--n_epoch', type=int, default=50,
                                        help='Number of epoch to run')
                    parser.add_argument('--n_pool', type=int, default=8,
                                        help='Number of process to sample subgraph')
                    parser.add_argument('--n_batch', type=int, default=16,
                                        help='Number of batch (sampled graphs) for each epoch')
                    parser.add_argument('--batch_size', type=int, default=256,
                                        help='Number of output nodes for training')
                    parser.add_argument('--clip', type=int, default=0.5,
                                        help='Gradient Norm Clipping')


                    jupyter = False
                    if jupyter:
                        args = parser.parse_args(args=[])
                    else:
                        args = parser.parse_args()
                    args_print(args)


                    if args.cuda != -1:
                        device = torch.device("cuda:" + str(args.cuda))
                    else:
                        device = torch.device("cpu")

                    graph = dill.load(open(args.data_dir, 'rb'))

                    target_type = 'def'
                    train_target_nodes = graph.train_target_nodes
                    valid_target_nodes = graph.valid_target_nodes
                    test_target_nodes = graph.test_target_nodes
                    pre_target_nodes = graph.pre_target_nodes
                    all_nodes = np.array(list(pre_target_nodes) + list(train_target_nodes) + list(valid_target_nodes) + list(test_target_nodes))

                    types = graph.get_types()
                    criterion = nn.NLLLoss()


                    stats = []
                    res = []
                    best_val   = 0
                    train_step = 0

                    pool = mp.Pool(args.n_pool)
                    st = time.time()
                    jobs = prepare_data(pool)


                    '''
                        Initialize GNN (model is specified by conv_name) and Classifier
                    '''
                    gnn = None
                    gnn = GNN(conv_name = args.conv_name, in_dim = len(graph.node_feature[target_type]['emb'].values[0]), n_hid = args.n_hid, \
                              n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout, num_types = len(types), \
                              num_relations = len(graph.get_meta_graph()) + 1, prev_norm = args.prev_norm, last_norm = args.last_norm, use_RTE = False)
                    if args.use_pretrain:
                        gnn.load_state_dict(load_gnn(torch.load(args.pretrain_model_dir)), strict = False)
                        print('Load Pre-trained Model from (%s)' % args.pretrain_model_dir)

                    gnn.eval()
                    with torch.no_grad():
                        # for _ in range(10):
                        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = \
                                    all_nodes_sample(randint(), all_nodes, {1: True})
                        paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                    edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]

                    print(paper_rep.shape)
                    scaler = None
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(paper_rep)

                    kmeans_kwargs = {
                    "init": "k-means++",
                     "n_init": 10,
                    "max_iter": 1000,
                    "random_state": 42 }

                    # sse = []
                    # k = 2
                    # kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
                    # kmeans.fit(scaled_features)
                    # kmeans.cluster_centers_
                    # # recommend number of tiers
                    # sse.append(kmeans.inertia_)

                    sse = []
                    total_k = 11
                    for k in range(1, total_k):
                        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
                        kmeans.fit(scaled_features)
                        sse.append(kmeans.inertia_)
                        print(k)
                    plt.style.use("fivethirtyeight")
                    plt.plot(range(1, total_k), sse)
                    plt.xticks(range(1, total_k))
                    plt.xlabel("Number of Clusters")
                    plt.ylabel("SSE")
                    plt.tight_layout()
                    plt.savefig(current_outdir + '_knee.png', format='png', dpi=300)
                    plt.show()
                    plt.close()

                    kl = KneeLocator(range(1, total_k), sse, curve="convex", direction="decreasing")
                    optimum_k = kl.elbow
                    print(optimum_k)
                    kmeans = None
                    kmeans = KMeans(n_clusters=optimum_k * 2, **kmeans_kwargs)
                    kmeans.fit(scaled_features)

                    from sklearn.decomposition import PCA
                    pca = None
                    pca = PCA(n_components=2, random_state=42)
                    pca_features = pca.fit_transform(scaled_features)


                    if '.lef' in lef_dir:
                        lef_lines = []
                        with open(lef_dir, 'r') as f:
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
                    elif 'txt' in lef_dir:
                        lef_lines = []
                        with open(lef_dir, 'r') as f:
                            for line in f:
                                for i in line.split('\n'):
                                    if i != '':
                                        lef_lines.append(i)

                        dimensions = {}
                        for line in lef_lines:
                            if '/' in line:
                                name = line.split()[0].split('/')[1]
                                size_elements = float(line.split()[1])
                                dimensions[name] = size_elements
                    dimension_placeholder = np.mean(np.array(list(dimensions.values())))
                    # subject = netlist_dir.split('/')[-1].rstrip('.v')


                    raw_lines = []
                    with open(netlist_dir, 'r') as f:
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
                        data[counter] = {'kind': kind, 'name': name, 'con': count, 'ports': ports, 'nets': connections, 'time': counter}
                        counter += 1

                    for i in range(len(data)):
                        try:
                            data[i]['size'] = dimensions[data[i]['kind']]
                        except:
                            data[i]['size'] = dimension_placeholder

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
                                if (g,gate) not in gategateport:
                                    gategateport[(gate,g)].append(current_port)
                                else:
                                    gategateport[(g,gate)].append(current_port)

                        data[i]['connected_gates'] = connected_gates
                        gategate[gate] = connected_gates



                    cluster_centers = np.array(pca.transform(kmeans.cluster_centers_))
                    cluster_place = np.argsort(cluster_centers[:,0])
                    for partition_type in partition_types:
                        if partition_type == 'minmiv':
                            dummy_tiers = []
                            for dummy_tier_i in range(2):
                                dummy_tiers.extend([dummy_tier_i] * int(optimum_k))
                            perm = perm_unique(dummy_tiers)
                            # tiers = [list(dummy) for dummy in list(permutations(dummy_tiers))]
                            # tiers_cleaned = []
                            # for dummy in tiers:
                            #     if dummy not in tiers_cleaned:
                            #         tiers_cleaned.append(dummy)
                            # print(tiers, len(tiers))
                            # print(tiers_cleaned, len(tiers_cleaned))

                            perm_counter = 0
                            best_cluster_order = []
                            lowest_miv_num = 999999999
                            for p in perm:
                                cluster_order = list(p)
                                if cluster_order[0] != 0 or cluster_order[-1] != 1:
                                    continue
                                cluster_map = {}
                                current_tier = 0
                                for c in range(len(cluster_place)):
                                    cluster_map[cluster_place[c]] = cluster_order[c]

                                # for c in range(len(cluster_order)):
                                #     cluster_map[cluster_order[c]] = c % 2

                                tier_dict = {}
                                tiers = [cluster_map[t] for t in kmeans.labels_]
                                for i in range(len(all_nodes)):
                                    tier_dict[all_nodes[i]] = tiers[i]

                                print(np.sum(tiers) / len(tiers))

                                counter_gate = {}
                                for i in data:
                                    gate = data[i]['name']
                                    counter_gate[i] = gate
                                    data[i]['tier'] = tier_dict[i]
                                tier_gate = {}
                                for i in tier_dict:
                                    tier_gate[counter_gate[i]] = tier_dict[i]

                                total_size = 0
                                tier_0_size = 0
                                tier_1_size = 0
                                for i in range(len(data)):
                                    total_size += data[i]['size']
                                    if data[i]['tier'] == 0:
                                        tier_0_size += data[i]['size']
                                    elif data[i]['tier'] == 1:
                                        tier_1_size += data[i]['size']
                                print(tier_0_size, tier_1_size, total_size, tier_0_size/total_size)
                                if 0.4 < tier_0_size/total_size < 0.6:
                                    miv_count = 0
                                    for net in nets:
                                        connected_gates_to_net = np.array(netgate[net])
                                        connected_gates_tiers = np.array([tier_gate[c] for c in connected_gates_to_net])
                                        # gates_tier_0 = connected_gates_to_net[connected_gates_tiers == 0]
                                        # gates_tier_1 = connected_gates_to_net[connected_gates_tiers == 1]
                                        # if len(gates_tier_0) != 0 and len(gates_tier_1) != 0:
                                        if 0 in connected_gates_tiers and 1 in connected_gates_tiers:
                                            # all_mivs.append(miv_number)
                                            # gates_to_mivs_0[miv_number] = list(gates_tier_0)
                                            # gates_to_mivs_1[miv_number] = list(gates_tier_1)
                                            # nets_to_mivs[miv_number] = net
                                            # miv_number += 1
                                            miv_count += 1
                                    print(timing, perm_counter, miv_count)
                                    if miv_count < lowest_miv_num:
                                        best_cluster_order = cluster_order
                                        lowest_miv_num = miv_count
                                perm_counter += 1
                        if partition_type == 'middle':
                            current_tier = 0
                            best_cluster_order = []
                            for c in range(len(cluster_place)):
                                if c >= optimum_k * (current_tier + 1):
                                    current_tier += 1
                                best_cluster_order.append(current_tier)
                        cluster_map = {}
                        for c in range(len(cluster_place)):
                            cluster_map[cluster_place[c]] = best_cluster_order[c]
                        # for c in range(len(cluster_order)):
                        #     cluster_map[cluster_order[c]] = c % 2

                        tier_dict = {}
                        tiers = [cluster_map[t] for t in kmeans.labels_]
                        for i in range(len(all_nodes)):
                            tier_dict[all_nodes[i]] = tiers[i]

                        print(np.sum(tiers) / len(tiers))

                        type_dict = []
                        for i in range(len(all_nodes)):
                            type_dict.append(data[all_nodes[i]]['kind'][:2])
                        print(np.unique(type_dict))

                        counter_gate = {}
                        for i in data:
                            gate = data[i]['name']
                            counter_gate[i] = gate
                            data[i]['tier'] = tier_dict[i]
                        tier_gate = {}
                        for i in tier_dict:
                            tier_gate[counter_gate[i]] = tier_dict[i]

                        total_size = 0
                        tier_0_size = 0
                        tier_1_size = 0
                        for i in range(len(data)):
                            total_size += data[i]['size']
                            if data[i]['tier'] == 0:
                                tier_0_size += data[i]['size']
                            elif data[i]['tier'] == 1:
                                tier_1_size += data[i]['size']
                        print(tier_0_size, tier_1_size, total_size, tier_0_size/total_size)


                        tier_n = 2

                        import seaborn as sns

                        label_encoder = LabelEncoder()
                        true_labels = label_encoder.fit_transform(type_dict)
                        # all_markers = ['3', '4', '+', 'x', 'o', 'v', '^', '*', 'd', 's', '1', '2']
                        all_markers = ['o'] * optimum_k * 2
                        pcadf = pd.DataFrame(pca_features, columns=["component_1", "component_2"])
                        pcadf["predicted_cluster"] = kmeans.labels_
                        pcadf["true_label"] = label_encoder.inverse_transform(true_labels)

                        plt.style.use("fivethirtyeight")
                        plt.figure(figsize=(10, 8))
                        # fix color wheel
                        # scat = sns.scatterplot("component_1", "component_2", s=50, data=pcadf,
                        #                        hue="predicted_cluster", style="true_label",
                        #                        palette=sns.color_palette("tab10", optimum_k * 2))
                        scat = sns.scatterplot("component_1", "component_2", s=50, data=pcadf,
                                               hue="true_label", style="predicted_cluster",
                                               palette=sns.color_palette("tab10", len(np.unique(type_dict))), markers=all_markers[:optimum_k*2])
                        scat.set_title("Clustering results")
                        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
                        plt.legend(fontsize=7, ncol=1)
                        plt.title('Area of tier 0 equals to '+str(round(tier_0_size/total_size * 100, 2))+' percent of total area.')
                        plt.tight_layout()

                        plt.savefig(current_outdir + '_partition' + partition_type + '_clusters_detailed.png', format='png', dpi=300)
                        plt.show()
                        plt.close()

                        pcadf = pd.DataFrame(pca_features, columns=["component_1", "component_2"])
                        pcadf["predicted_cluster"] = kmeans.labels_
                        pcadf["true_label"] = tiers

                        plt.style.use("fivethirtyeight")
                        plt.figure(figsize=(10, 8))
                        # fix color wheel
                        scat = sns.scatterplot("component_1", "component_2", s=50, data=pcadf,
                                               hue="predicted_cluster", style="true_label",
                                               palette=sns.color_palette("tab10", optimum_k * 2))
                        scat.set_title("Clustering results")
                        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
                        plt.legend(fontsize=7, ncol=1)
                        plt.title('Area of tier 0 equals to '+str(round(tier_0_size/total_size * 100, 2))+' percent of total area.')
                        plt.tight_layout()
                        plt.savefig(current_outdir + '_partition' + partition_type + '_clusters.png', format='png', dpi=300)
                        plt.show()
                        plt.close()

                        # insert MIVs
                        free_gate = len(data)
                        miv_number = free_gate
                        all_mivs = []
                        gates_to_mivs_0 = {}
                        gates_to_mivs_1 = {}
                        nets_to_mivs = {}
                        for net in nets:
                            connected_gates_to_net = np.array(netgate[net])
                            connected_gates_tiers = np.array([tier_gate[c] for c in connected_gates_to_net])
                            gates_tier_0 = connected_gates_to_net[connected_gates_tiers == 0]
                            gates_tier_1 = connected_gates_to_net[connected_gates_tiers == 1]
                            if len(gates_tier_0) != 0 and len(gates_tier_1) != 0:
                                all_mivs.append(miv_number)
                                gates_to_mivs_0[miv_number] = list(gates_tier_0)
                                gates_to_mivs_1[miv_number] = list(gates_tier_1)
                                nets_to_mivs[miv_number] = net
                                miv_number += 1
                        print(len(all_mivs))

                        file1 = open(current_outdir + '_partition' + partition_type + '_' + str(
                            tier_n)+'tier_0.v', 'w+')

                        # file1 = open(out_dir + subject + '/' + str(tier_n)+'tier_0_' + subject + '.v', "w+")
                        for i in raw_lines[0:cut_off]:
                            file1.write(i)

                        for i in range(len(data)):
                            current = data[i]
                            if current['tier'] == 0:
                                for j in range(line_index[i][0], line_index[i][1]):
                                    file1.write(raw_lines[j])

                        for miv in all_mivs:
                            con = nets_to_mivs[miv]
                            if con.split('[')[0] in wires:
                                file1.write(
                                    '  ANTENNA_RVT M' + str(miv) + ' (.INP(' + con.split('[')[0] + ' [' + con.split('[')[1] + '));\n')
                            elif con[0] == '\\':
                                file1.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + ' ));\n')
                            else:
                                file1.write('  ANTENNA_RVT M' + str(miv) + ' (.INP(' + str(con) + '));\n')
                        file1.write('endmodule\n\n')

                        file1.close()

                        file2 = open(current_outdir + '_partition' + partition_type + '_' + str(
                            tier_n) + 'tier_1.v', 'w+')
                        # file2 = open(out_dir + subject + '/' + str(tier_n)+'tier_1_' + subject + '.v', "w+")
                        for i in raw_lines[0:cut_off]:
                            file2.write(i)

                        for i in range(len(data)):
                            current = data[i]
                            if current['tier'] == 1:
                                for j in range(line_index[i][0], line_index[i][1]):
                                    file2.write(raw_lines[j])

                        for miv in all_mivs:
                            con = nets_to_mivs[miv]
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

                        file3 = open(current_outdir + '_partition' + partition_type + '_' + str(
                            tier_n) + 'tier_0.txt', 'w+')
                        # file3 = open(out_dir + subject + '/' + str(tier_n)+'tier_0_' + subject + '.txt', "w+")
                        for i in range(len(data)):
                            current = data[i]
                            if current['tier'] == 0:
                                file3.write(current['name']+'\n')
                        file3.close()

                        file4 = open(current_outdir + '_partition' + partition_type + '_' + str(
                            tier_n) + 'tier_1.txt', 'w+')
                        # file4 = open(out_dir + subject + '/' + str(tier_n)+'tier_1_' + subject + '.txt', "w+")
                        for i in range(len(data)):
                            current = data[i]
                            if current['tier'] == 1:
                                file4.write(current['name']+'\n')
                        file4.close()

                        cons = []
                        for miv in all_mivs:
                            con = nets_to_mivs[miv]
                            if con not in cons:
                                cons.append(con)

                        file5 = open(current_outdir + '_partition' + partition_type + '_' + str(
                            tier_n) + 'tier_cut_nets.v', 'w+')
                        # file5 = open(out_dir + subject + '/' + str(tier_n)+'tier_cut_nets_' + subject + '.txt', "w+")
                        for i in range(len(cons)):
                            file5.write(cons[i]+'\n')
                        file5.close()

