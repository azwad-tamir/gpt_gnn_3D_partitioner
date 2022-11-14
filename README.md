# GPT_GNN_3D_partitioner
## A GPT-GNN based verilog netlist partitioner for 3D IC design
Authors: Azwad Tamir & Milad Salem
[Only a part of the entire work is presented here due to copywrite issues as it was a company funded project]

## Background: 
As the channel length of modern IC technology shrinks down to nanometer scale, it is no longer possible to keep up with the trend of increased performance of digital ICs like processors and GPUs with transistor dimensionality reduction alone. This warrants a paradigm shift and one of the most promising advances that is likely to take up central stage is three dimensional ICs. Already several 3D IC technologies have been proposed in the scientific community but with the lack of fully automated 3D commercial P&R tool, it is very difficult to fabricate and test these ICs in the real world. 
The purpose of this project is to develop a partitioner tool using deep learning which would be able to segment a gate level netlist into multiple tiers to facilitate the implementation of 3D IC physical design. After the netlist has been partitioned, any conventional 2D IC design tool like the Synopsys ICC2 could be used to realize the full 3D IC physical design.
First, the gate-level Verilog netlist is converted into a graph with the nodes representing the standard cells and the edges representing the connections between them. Then, a deeplearning framework named GPT-GNN (Generative Pre-Training of Graph Neural Networks) is used to latent vector representation of the input graphs. The node features used in the model are timing variables like worst slack, pin slew and worst delay while the edge features include the distance between each standard cell instances in the 2D version of the placement. Other features like module hierarchy information and n hop neighbors could also be put into the model as necessary. The resulting latent vectors are then used to partition the graph into n tiers.

## Usage:
The tool operation is divided into three parts: preprocessing, pretraining and inference. Each of them needs to be executed into succession to obtain the final results. The preprocessing step generates the input graph and adds the features. The pretraining step manages the training of the model on the graphs. Finally, the inference step computes the latent vectors and partitions the graph into tiers. 
The following step outlines the implementation:

* Clone the GPT-GNN repository from https://github.com/UCLA-DM/GPT-GNN and paste it into the root directory of the project.

* Download and install the required python libraries by running 
    >> pip install -r requirements.txt
  
* Copy the input verilog gatelevel netlist, features, cell list and initial 2D placement into the data directory.

* Run:
  python3 preprocess_verilog.py <data_dir_root> <subject_name>
  example:
    >> python3 preprocess_verilog.py data_dir_root=data subject_name=aes_cipher
    
* Run: 
  python3 pretrain_verilog.py <subject_name>
  example:
    >> python3 preprocess_verilog.py subject_name=aes_cipher

* Run: 
  python3 inference_verilog.py <data_dir_root> <subject_name> <out_dir>
  example:
    >> python3 inference_verilog.py data_dir_root=data subject_name=aes_cipher out_dir=results

