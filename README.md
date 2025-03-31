# Spatio-Temporal Intent Modeling for Sequential Recommendation

This is our Pytorch implementation for the STIRec


## Requirement

- **Platforms**: Ubuntu 20.04
- **Device**: RTX 3090, Driver Version: 535.154.05, CUDA Version: 12.2
- Python 3.11
- Torch 2.2
- Cuda 12.1
- numpy<2

The full list is detailed in [requirements](https://github.com/TosakRin/PTSR/blob/main/requirements.txt).

> [!TIP]
> The code may contain some version-specific code. It's recommended to follow our verified environment. You may take some extra effort for older version of Python & PyTorch, but the modification will not be troublesome. It's up to you.



## Data Preparaion

Please refer to [data/README.md](https://github.com/TosakRin/PTSR/tree/main/data) for data preparation.

We have already provided original sequence files:

- `Beauty.txt`
- `m1-1m.txt`
- `Sports.txt`
- `Toys.txt`
- `Yelp.txt`

To construct **SIG** in paper, just run the following command:

```sh
python graph.py --msg gen
```

The final data organization:

```sh
$ tree ../data

../data
├── Beauty_graph_50.pkl
├── Beauty_subseq_50.txt
├── Beauty_t_50.pkl
├── Beauty.txt
├── ml-1m_graph_50.pkl
├── ml-1m_subseq_50.txt
├── ml-1m_t_50.pkl
├── ml-1m.txt
├── README.md
├── Sports_graph_50.pkl
├── Sports_subseq_50.txt
├── Sports_t_50.pkl
├── Sports.txt
├── Toys_graph_50.pkl
├── Toys_subseq_50.txt
├── Toys_t_50.pkl
└── Toys.txt
```

## Training/Testing

```sh
python main.py --data_name Beauty --do_test --do_eval --scheduler warmup+multistep --milestones "[25, 50]" --gamma 0.1 --warm_up_epochs 5 --loader_type new --gcn_mode batch --gpu_id 0 --log_root logs --gnn_layer 4 --msg training
```

- `data_name`: Beauty/Sports/Toys/ml-1m.
- `gpu_id`: Device ID.
- `log_root`: Root directory for python logging & TensorBoard.
- `msg`: Custom message dentifiers for console output & log file.

