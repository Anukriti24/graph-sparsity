# Graph Sparsity, Information Propagation & Node Classification in GNNs

This project investigates how **graph sparsity** affects information propagation and node-classification performance in Graph Neural Networks (GNNs). It combines controlled experiments on synthetic Stochastic Block Model graphs with a real-world experiment on the Cora citation network.

The study focuses on the trade-off between:

- **under-reaching / under-squashing in very sparse graphs**, where nodes receive too little useful neighbourhood information; and
- **over-smoothing in deep or highly connected graphs**, where node representations become increasingly similar.

The notebook compares several GNN architectures, measures representation smoothing and influence propagation, and tests spectral and graph-rewiring explanations.

## Research question

> How does graph sparsity affect information propagation and node-classification performance in Graph Neural Networks?

## Hypotheses

| ID | Hypothesis | Result in the current run |
|---|---|---|
| H1 | Very sparse graphs limit useful information propagation. | Supported |
| H2 | Denser graphs over-smooth more quickly as depth increases. | Supported by the structural smoothing metrics |
| H3 | Classification accuracy has an intermediate sparsity “sweet spot.” | Not clearly supported; accuracy mostly improves and then saturates |
| H4 | GNN architectures respond differently to sparsity. | Partially supported, especially in the sparse heterophilous setting |
| H5 | The best GNN depth depends on graph density. | Supported |

## Project overview

The notebook contains three main parts.

### 1. Controlled synthetic graphs

Graphs are generated with a **Stochastic Block Model (SBM)**. Community structure is fixed while the requested average degree is varied, allowing graph density to be treated as the main experimental variable.

Default synthetic configuration:

- 4 communities;
- 120 nodes per community;
- 480 nodes in total;
- 24-dimensional node features;
- weak noisy class-dependent feature signal;
- 20 labelled training nodes per class;
- the remaining nodes split equally between validation and test sets.

### 2. GNN models

The following architectures are evaluated:

- Graph Convolutional Network (**GCN**);
- GraphSAGE;
- Graph Attention Network (**GAT**);
- Graph Isomorphism Network (**GIN**);
- a graph-independent **MLP baseline**.

The common training configuration uses:

- hidden dimension: `64`;
- dropout: `0.5`;
- Adam optimizer;
- learning rate: `0.01`;
- weight decay: `5e-4`;
- 200 epochs;
- validation-based selection of the reported test accuracy.

### 3. Propagation and structural diagnostics

The analysis uses:

- test accuracy;
- Dirichlet energy;
- Mean Average Distance (MAD);
- Jacobian-based influence by shortest-path distance;
- normalized-Laplacian spectral gap;
- the second-largest propagation eigenvalue magnitude;
- structure-aware graph rewiring.

## Experiments

| Section | Experiment | Main purpose | Generated output |
|---|---|---|---|
| 1.1 | SBM sparsity visualisation | Show sparse, medium and dense graph regimes | `figures/01_sbm_sparsity_levels.png` |
| 4 | Sparsity vs. accuracy | Compare GNN architectures over graph density | `figures/02_accuracy_vs_sparsity.png` |
| 5 | Over-smoothing vs. depth | Measure representation collapse at different densities | `figures/03_oversmoothing_vs_depth.png` |
| 6 | Influence vs. distance | Measure structural receptive-field influence | `figures/04_influence_vs_distance.png` |
| 7 | Depth × density | Test whether useful model depth depends on density | `figures/05_depth_density_heatmap.png` |
| 8 | Cora sparsification | Validate the sparsity effect on a real graph | `figures/06_cora_sparsification.png` |
| 10 | Over-smoothing law | Compare measured convergence with the spectral prediction | `figures/07_oversmoothing_law.png` |
| 11 | Spectral gap vs. edge count | Test whether spectral gap explains accuracy better than density | `figures/08_disentangle_gap_vs_degree.png` |
| 12 | Rewiring interventions | Compare different ways of improving graph connectivity | `figures/09_rewiring_interventions.png` |
| 13 | Heterophily robustness | Compare architectures under homophily and heterophily | `figures/10_heterophily.png` |

Each experiment also writes its underlying values to a CSV file in `figures/`.

## Main findings from the current notebook run

### Sparse graphs are the clearest performance bottleneck

On the homophilous SBM graphs, all message-passing models improve as average degree increases. At low degree, nodes have fewer useful same-community neighbours, and performance is lower and less stable. At medium and high density, most models approach near-perfect accuracy.

This supports **H1**, but the current results do not show a clear intermediate optimum. Instead, performance generally improves and then saturates, so **H3 is not supported in the present synthetic setting**.

### Representation smoothing increases strongly with depth

Both Dirichlet energy and MAD decrease as additional GCN layers are applied. The decay is fastest on the denser graphs, showing that repeated propagation makes node representations increasingly similar.

This provides structural support for **H2**.

### Depth and density interact

The depth-density heatmap shows that deep models are most harmful in sparse graphs:

- at requested average degree `2`, accuracy falls from approximately `0.92` with one layer to `0.78` with eight layers;
- medium-density graphs remain strong across several depths;
- at average degree `30`, performance remains near `1.00` through six layers but drops to about `0.93` at eight layers.

The useful depth is therefore density-dependent, supporting **H5**.

### Cora is sensitive to aggressive edge removal

Randomly removing edges from Cora causes a steady decline in test accuracy:

- approximately `0.81` with the full graph;
- approximately `0.71` after dropping 60% of edges;
- approximately `0.59` after dropping 90% of edges.

This confirms that severe sparsification can remove information required for node classification.

### The spectral over-smoothing mechanism is visible

Repeated application of the normalized propagation operator produces approximately geometric convergence toward its leading eigenspace. The mean absolute difference between the predicted slope, `log(mu)`, and the measured residual slope is about `0.09`.

This is the strongest direct evidence in the notebook for a spectral explanation of over-smoothing.

### Spectral gap is not a universal predictor of accuracy in the current experiment

The combined density-and-community-contrast analysis produces:

- `R² ≈ 0.18` for accuracy against average degree;
- `R² ≈ 0.00` for accuracy against spectral gap.

Therefore, the current run does **not** support the claim that spectral gap alone explains classification accuracy better than edge count. Community-label alignment and homophily affect accuracy independently of global graph mixing, so a single spectral statistic is insufficient across both sweeps.

This negative result is still useful: graph connectivity and task-relevant connectivity are not the same thing.

### Rewiring changes connectivity more than accuracy

At the tested edge budget:

| Method | Mean accuracy | Mean spectral gap |
|---|---:|---:|
| Sparse baseline | 0.950 | 0.106 |
| Random edge addition | 0.935 | 0.125 |
| Curvature-guided rewiring | 0.937 | 0.113 |
| Effective-resistance rewiring | 0.943 | 0.106 |
| Virtual node | 0.950 | 0.280 |

The virtual node substantially increases the spectral gap without improving accuracy over the already strong baseline. This again shows that better global mixing does not automatically improve classification.

### Architecture differences are clearest under sparse heterophily

In the sparse heterophilous regime, GraphSAGE performs substantially better than GCN in the current run. At higher degrees, all models approach perfect accuracy.

The near-perfect high-density heterophily results should be interpreted cautiously because the synthetic node features remain class-informative and the two-block graph is highly regular.

## Important interpretation notes

1. **Experiments 2 and 3 use untrained GCNs.**  
   They isolate the structural behaviour of propagation, but they do not measure smoothing or influence in learned representations.

2. **Raw Dirichlet energy is affected by the number of retained edges.**  
   In the Cora experiment, the energy formula is divided by the number of nodes rather than the number of edges. Removing edges therefore lowers the total mechanically. Cross-sparsity comparisons should use an edge-normalized quantity before drawing a strong smoothing conclusion.

3. **The best-validation model is not restored.**  
   The reported accuracy corresponds to the epoch with the best validation score, but the model remains at its final epoch. Any embedding metric computed after training may therefore describe a different model state from the reported accuracy.

4. **Nominal and realized average degree should be kept separate.**  
   SBM samples have slightly different realized degrees across seeds. For clean mean curves, results should be grouped by the requested degree before plotting; otherwise seed-specific points can be connected into misleading zig-zag lines.

5. **The number of random seeds is small.**  
   Most experiments use two or three seeds. More seeds and confidence intervals would provide stronger statistical evidence.

6. **The rewiring methods are lightweight experimental implementations.**  
   They demonstrate the ideas but should not be treated as exact reproductions of full SDRF or optimized resistance-based algorithms.

## Repository structure

```text
graph-sparsity/
├── Graph Sparsity.ipynb      # Complete experiment notebook
├── README.md                 # Project documentation
├── data/
│   └── Cora/                 # Downloaded automatically by PyTorch Geometric
└── figures/                  # Generated plots and CSV result files
```

The `data/` and `figures/` directories are created when the notebook is executed.

## Installation

Python 3.10 or newer is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install torch torch_geometric networkx numpy pandas scipy matplotlib seaborn jupyter
```

PyTorch Geometric must be compatible with the installed PyTorch version and CUDA setup.

## Running the project

Open the notebook in VS Code or Jupyter:

```bash
jupyter notebook "Graph Sparsity.ipynb"
```

Run all cells from top to bottom.

The first Cora execution downloads the dataset into `data/Cora/`. An internet connection is required for that initial download.

For a faster test run, reduce the seed lists to a single value:

```python
SEEDS = [0]
GRID_SEEDS = [0]
SEEDS5 = [0]
SEEDS2 = [0]
HET_SEEDS = [0]
```

The most computationally expensive sections are:

- sparsity vs. architecture;
- depth × density;
- heterophily robustness;
- effective-resistance rewiring.

## Reproducibility

The notebook seeds:

- Python `random`;
- NumPy;
- PyTorch;
- CUDA random generators.

The execution device is selected automatically:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

GPU execution is supported, but the full notebook can also run on CPU.

## Suggested next improvements

- restore and evaluate the best-validation checkpoint;
- normalize Dirichlet energy for comparisons across different edge counts;
- plot means against requested degree instead of connecting realized-degree seed values;
- increase the number of random seeds;
- report confidence intervals and statistical tests;
- weaken or remove the node-feature signal in the heterophily study;
- compare with heterophily-specific models;
- evaluate additional real-world datasets;
- separate graph mixing, homophily and label informativeness in the spectral analysis;
- add a final notebook discussion section that automatically summarizes all generated CSV files.

## Technologies

- Python
- PyTorch
- PyTorch Geometric
- NetworkX
- NumPy
- Pandas
- SciPy
- Matplotlib
- Seaborn
