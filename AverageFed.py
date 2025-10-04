import torch
from torch import amp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib
# matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import copy
import traceback
import itertools
import pickle
from pathlib import Path
# Custom imports; ensure these modules are in your PYTHONPATH
from UtilityGraph import *
from Defence import *
from Corruption import *
from UtilityMLP import *
import gc
import os
from contextlib import contextmanager
import warnings
# --------------------- results-persistence helpers ---------------------
from collections import defaultdict
from datetime import datetime

def append_result(entry, path="resultsFedAVG.pkl"):
    """
    Append *one* experiment result to the pickle file.
    Each call writes an independent pickle frame.
    """
    with open(path, "ab") as fh:           # binary-append mode
        pickle.dump(entry, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _nested_dict() -> defaultdict:          # top-level ? picklable
    return defaultdict(dict)

_RESULTS = defaultdict(_nested_dict)      

def _to_plain(obj):
    "Recursively turn defaultdicts into plain dicts."
    if isinstance(obj, defaultdict):
        obj = {k: _to_plain(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        obj = {k: _to_plain(v) for k, v in obj.items()}
    return obj


def save_results(path: str = "resultsFedAVG.pkl") -> None:
    """
    Persist global _RESULTS; merge under a timestamp if file exists.
    """
    ts_key = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ? convert to plain dict to avoid defaultdict pickling woes
    clean_results = _to_plain(_RESULTS)

    payload = {ts_key: clean_results}

    p = Path(path)
    if p.exists():
        with p.open("rb") as fh:
            try:
                existing = pickle.load(fh)
            except Exception:
                existing = {}
        existing.update(payload)
        payload = existing

    tmp = p.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)  # highest proto ? smaller/faster
    tmp.replace(p)

    n_priv = len(clean_results)
    n_atk  = sum(len(a) for a in clean_results.values())
    n_noise = sum(len(n) for a in clean_results.values() for n in a.values())
    print(f"Saved {n_priv=} {n_atk=} {n_noise=} to {p} under key '{ts_key}'.")



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gc.collect()
torch.cuda.empty_cache()

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,expandable_segments:True"
)


# Define the MLP model
class Current(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Current, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, input_size)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
class FlipLabelDataset(Dataset):
    """Dataset wrapper that flips labels according to a fixed map."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.label_map = {0:3,1:4,2:7,3:5,4:8,5:0,6:9,7:6,8:2,9:1}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        flipped_label = self.label_map[label]
        return image, flipped_label

def split_dataset(dataset, num_clients):
    dataset_size = len(dataset)
    indices = np.random.permutation(dataset_size)  # Shuffle all indices
    data_per_client = dataset_size // num_clients
    split_sizes = [data_per_client] * num_clients

    # Distribute any remaining data among the first few clients
    for i in range(dataset_size % num_clients):
        split_sizes[i] += 1

    datasets = []
    start = 0
    for size in split_sizes:
        datasets.append(Subset(dataset, indices[start:start + size]))
        start += size  # Move the start index forward

    return datasets

def train_local_model(dataset, global_model, num_epochs):

    # --- model clone & optimiser -------------------------------------------
    model = type(global_model)(input_size, hidden_size, num_classes).to(device)
    model.load_state_dict(global_model.state_dict())
    model.train()

    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optim     = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # --- metrics to collect -------------------------------------------------
    epoch_losses, epoch_accs = [], []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct, total = 0, 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # forward / backward / step
            out  = model(x)
            loss = criterion(out, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # accumulate metrics
            running_loss += loss.item()
            preds   = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

        # epoch-level averages
        avg_loss = running_loss / len(loader)
        acc      = 100.0 * correct / total

        epoch_losses.append(avg_loss)
        epoch_accs.append(acc)

        print(f"Epoch {epoch+1}/{num_epochs}  loss={avg_loss:.4f}  acc={acc:.2f}%")

    return model, epoch_losses, epoch_accs

def aggregate_models2(local_models, input_size, hidden_size, num_classes, num_nodesSelected):
    """
    Compute the averaged model parameters from local models and return a new model.

    Args:
        local_models (list): List of PyTorch models (local models).
        input_size (int): Input size for the model.
        hidden_size (int): Hidden layer size.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: A new model with averaged parameters.
    """
    # Extract state_dicts from local models
    local_dicts = [model.state_dict() for model in local_models]

    # Initialize global_dict with zero tensors of the same shape as the first model's parameters
    global_dict = {key: torch.zeros_like(local_dicts[0][key]) for key in local_dicts[0].keys()}

    # Compute the average over all local models
    for key in global_dict.keys():
        global_dict[key] = sum(local_dicts[i][key] for i in num_nodesSelected) / len(num_nodesSelected)

    averaged_model = type(local_models[0])(input_size, hidden_size, num_classes).to(device)  
    averaged_model.load_state_dict(global_dict)

    return averaged_model


def aggregate_models(True_avg, node_models, G, tolerance, c, max_iters, rejection_threshold, K_decision, averaging, when, CorruptValue, true_nodes, Print_Val, noise_STD, PrivacyMethod, p, log_filename= None, perm_threshold=0.5):
    # Initialize variables
    num_nodes = len(node_models)
    converged = False
    count = 0
    Error = []
    Track = 0
    mask_history = []
    # lying_nodes = lying_nodes or set()
    nodes = list(G.nodes())
    True_avg_dict = True_avg.state_dict()

    # Detection 
    mask = np.ones((num_nodes, num_nodes), dtype=int)  # 1: active, -1: blocked
    reject_count = np.zeros((num_nodes, num_nodes), dtype=int)
    sum_check = np.zeros((num_nodes, num_nodes))
    thresholds = np.zeros(num_nodes)
    MAD = np.zeros(num_nodes)
    D = {i: {j: 0 for j in G.neighbors(i)} for i in G.nodes()}  # Suspicion scores
    ignored = {i: set() for i in G.nodes()}  # Ignored neighbors

    # Model parameter initialization
    local_dicts = [model.state_dict() for model in node_models]
    global_dict = node_models[0].state_dict() 
    # param_keys = True_avg.keys()  # all models have the same parameters
    A_ij = calc_incidence_nested(G)
    x_history = []
    # Initialize PDMM variables with tensor support
    x = [{} for _ in range(num_nodes)]
    z = [{} for _ in range(num_nodes)]
    y = [{} for _ in range(num_nodes)]
    y_transmit = [{} for _ in range(num_nodes)]
    # Initialize x with local models and move to device
    for i in range(num_nodes):
        for key in True_avg_dict.keys():
            # if PrivacyMethod == 1:
            #     x[i][key] = local_dicts[i][key].clone() + torch.randn_like(local_dicts[i][key]) * noise_STD
            # else:
                x[i][key] = local_dicts[i][key].clone()

    # Initialize z and y
    for i in range(num_nodes):
        z[i] = {}
        y[i] = {}
        y_transmit[i] = {}
        for j in G.neighbors(i):
            z[i][j] = {}
            y[i][j] = {}
            y_transmit[i][j] = {}
            for key in True_avg_dict.keys():
                if PrivacyMethod == 3:
                    z[i][j][key] = torch.randn_like(True_avg_dict[key]) * noise_STD
                else:
                    z[i][j][key] = torch.zeros_like(True_avg_dict[key])
                y[i][j][key] = torch.zeros_like(True_avg_dict[key])
                y_transmit[i][j][key] = torch.zeros_like(True_avg_dict[key])
         
    if PrivacyMethod == 2:
        smpc_masks = {}
        for i in range(num_nodes):
            for j in G.neighbors(i):
                if i < j:
                    smpc_masks[(i, j)] = {}
                    for key in True_avg_dict:
                        smpc_masks[(i, j)][key] = torch.randn_like(True_avg_dict[key])

    # print(lying_nodes)
    # Synchronous PDMM with detection
    while not converged and count < max_iters:
        # --------------------
        # 1. Synchronous x-update for all nodes
        # --------------------
        x_new = [{} for _ in range(num_nodes)]
        for i in range(num_nodes):
            # Count corrupt neighbors once for node i
            corrupt_neighbors = sum(1 for j in G.neighbors(i) if mask[i][j] == -1)
            effective_degree = G.degree[i] - corrupt_neighbors

            # Update each parameter using the same effective degree
            for key in True_avg_dict:
                x_new[i][key] = local_dicts[i][key].clone()
                for j in G.neighbors(i):
                    if mask[i][j] != -1:
                        x_new[i][key] -= A_ij[i][j] * z[i][j][key]
                x_new[i][key] = x_new[i][key] / (1 + c * effective_degree)

        x = x_new
        x_history.append(x.copy())

        # --------------------
        # 2. Dual variable update (y)
        # --------------------
        for i in range(num_nodes):
            for j in G.neighbors(i):
                for key in True_avg_dict:
                    y[i][j][key] = z[i][j][key] + 2 * c * A_ij[i][j] * x[i][key]
                    if PrivacyMethod == 1:
                        y[i][j][key] = y[i][j][key].clone() + torch.randn_like(local_dicts[i][key]) * noise_STD

        if PrivacyMethod == 2:
            # SMPC: Apply pairwise masks for secure aggregation
            for i in range(num_nodes):
                for j in G.neighbors(i):
                    for key in True_avg_dict:
                        if i < j:
                            y_transmit[i][j][key] = y[i][j][key] + smpc_masks[(i, j)][key]
                        else:
                            y_transmit[i][j][key] = y[i][j][key] - smpc_masks[(j, i)][key]
        else:
            # Original: no additional masking
            for i in range(num_nodes):
                for j in G.neighbors(i):
                    for key in True_avg_dict:
                        y_transmit[i][j][key] = y[i][j][key].clone()


        # --------------------
        # 3. Detection logic (executed periodically)
        # --------------------
        # if count > when:
        # print(f"Detecting at count {count}")
        for i in range(num_nodes):
            neighbors = [j for j in G.neighbors(i) if j not in ignored[i]]
            if not neighbors:
                continue

            # Precompute absolute values of y variables (for PDMM minus-sign handling)
            abs_y = {j: {key: torch.abs(y_transmit[j][i][key]) for key in True_avg_dict} for j in neighbors}

            # 1. Compute element-wise median (m_i)
            medians = {}
            for key in True_avg_dict:
                # Stack all neighbors' parameters for this key
                params = torch.stack([abs_y[j][key] for j in neighbors])
                medians[key] = torch.median(params, dim=0).values  # Element-wise median

            # 2. Compute Delta Y_{i,j} using infinity norm
            delta_ys = []
            for j in neighbors:
                max_diff = -float('inf')
                for key in True_avg_dict:
                    diff = torch.max(torch.abs(abs_y[j][key] - medians[key])).item()
                    if diff > max_diff:
                        max_diff = diff
                delta_ys.append(max_diff)

            # 3. Compute MAD and threshold
            median_delta = np.median(delta_ys)
            deviations = np.abs(delta_ys - median_delta)
            MAD_val = np.median(deviations)
            threshold = rejection_threshold * MAD_val
            epsilon = 1e-2  
            threshold = max(threshold, epsilon) # TO avaoid zero threshold

            # 4. Update suspicion scores
            for idx, j in enumerate(neighbors):
                # if j in lying_nodes:
                # print(f"Node {i} check {j}: ?Y={delta_ys[idx]:.2f}, threshold={threshold:.2f}")
                if delta_ys[idx] > threshold:
                    # print(f"To {i} Value of {j}: ?Y={delta_ys[idx]:.2f}, threshold={threshold:.2f}, D = {D[i][j]}")
                # if j in lying_nodes:
                    D[i][j] += 1
                    # if Print_Val:
                    # print(f"Node {i} suspicious of {j}: ?Y={delta_ys[idx]:.2f}, threshold={threshold:.2f}, D = {D[i][j]}")

            # 5. Periodic mitigation check
            if count % K_decision == 0 and count > 0:
                for j in list(D[i].keys()):  # Iterate over copy to allow modification
                    if D[i][j] > K_decision/2:
                        # print(f"Node {i} ignoring node {j} for next {K_decision} iterations")
                        mask[i][j] = -1
                    else:
                        mask[i][j] = 1
                    D[i][j] = 0
        mask_history.append(mask.copy())

        if PrivacyMethod == 2:
            for i in range(num_nodes):
                for j in G.neighbors(i):
                    if mask[i][j] == -1:
                        for key in True_avg_dict:
                            z[i][j][key] = (1 - averaging) * z[i][j][key] 
                    if mask[i][j] == 1:
                        for key in True_avg_dict:
                            if j < i:
                                unmasked = y_transmit[j][i][key] + smpc_masks[(j, i)][key]
                            else:
                                unmasked = y_transmit[j][i][key] - smpc_masks[(i, j)][key]
                            z[i][j][key] = (1 - averaging) * z[i][j][key] + (averaging) * unmasked
        else:                        
          for i in range(num_nodes):
              for j in G.neighbors(i):
                if mask[i][j] == -1:
                      # Apply noise to blocked channels
                    for key in True_avg_dict:
                        z[i][j][key] = (1 - averaging) * z[i][j][key] 
                if mask[i][j] == 1:
                    # Normal update from y values
                    for key in True_avg_dict:
                        z[i][j][key] = (1 - averaging) * z[i][j][key]  + (averaging) * y_transmit[j][i][key].clone()

        # --------------------
        # 5. Update global model and check convergence
        # --------------------
        # Initialize global dictionary as zero tensors of the same shape
        # for key in True_avg_dict.keys():
        #     True_avg_dict[key] = sum(x[i][key] for i in true_nodes) / len(true_nodes)
        # total_elements = 0
        # param_keys = list(True_avg.state_dict().keys())  # Get the list of parameter keys

        avg_error = 0
        total_elements = 0

        for i in true_nodes:
            node_error = 0
            for key in True_avg_dict:
                diff = x[i][key] - True_avg_dict[key]
                norm_diff = torch.norm(diff).item()
                node_error += norm_diff
            avg_error += node_error
            total_elements += 1

        # Final norm: average error divided by the total number of nodes
        avg_error /= total_elements

        # Store the computed average error
        Error.append(avg_error)
        #if count % 100 == 0:
        
          #with open(log_filename, "a") as fh:
          #  fh.write(
          #      f"[PDMM iter {count:4d}]  "
          #      f"privacy={PrivacyMethod}  attack={typeAttack}  "
          #      f"Error={avg_error:.8f}%\n"
          #  )
        # print(f"cunt {count} and error {Error[-1]}" )

        if avg_error < tolerance:
            print(f'Converged at iteration {count}')
            converged = True
        elif count % 100 == 0 and Print_Val:
            print(f'Iter {count}: Error {Error[-1]:.4f}')

        # Update parameters for next iteration
        count += 1
    # Update final models
    for i in range(num_nodes):
        model_dict = node_models[i].state_dict()
        for key in True_avg_dict:
            model_dict[key] = x[i][key].clone()
        node_models[i].load_state_dict(model_dict)

    # Create an averaged model with the final True_avg_dict
    averaged_model = type(node_models[0])(input_size, hidden_size, num_classes).to(device)  # Pass necessary arguments directly
    averaged_model.load_state_dict(True_avg_dict)


    return node_models, Error, mask_history

def evaluate(models, test_dataset, criterion):
    """Evaluate either a single model or list of models"""

    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # if not isinstance(models, (list, tuple)):
    #     models = [models]
    # avg_loss = 0.0
    # avg_accuracy = 0.0
    
    # for model in models:
    #     model.to(device)  # Move model to the correct device
    #     model.eval()
    #     with torch.no_grad():
    #         correct = 0
    #         total = 0
    #         running_loss = 0.0
    #         for images, labels in test_loader:
    #             images = images.to(device)
    #             labels = labels.to(device)
    #             outputs = model(images)
    #             loss = criterion(outputs, labels)
    #             running_loss += loss.item()

    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()

    #         avg_loss += running_loss / len(test_loader)
    #         avg_accuracy += 100 * correct / total

    # # Average across all models
    # avg_loss /= len(models)
    # avg_accuracy /= len(models)
    return 0, 0

def main():
    # Set random seeds
    Seed = 42
    random.seed(Seed)
    np.random.seed(Seed) 
    torch.manual_seed(Seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Seed)
        torch.cuda.manual_seed_all(Seed)

    # Build graph
    required_probability = 0.9999
    Amount_Clients = 20
    num_nodes, G, A, pos, r_c = build_random_graph(Amount_Clients, required_probability, fix_num_nodes=True)
    print("num_nodes:", num_nodes)

    # Hyperparameters
    global input_size, hidden_size, num_classes, num_epochs, batch_size, learning_rate, typeAttack, PrivacyMethod
    global num_clients, num_rounds, threshold, percentageCorrupt, corrupt
    input_size = 28 * 28
    hidden_size = 128
    num_classes = 10
    num_epochs = 5
    batch_size = 64
    learning_rate = 0.01
    num_clients = num_nodes
    num_rounds = 50
    threshold = 0.02
    typeAttack = 4              # 0: No attack, 1: Gaussian noise, 2: Copycat attack, 3: Gaussian addative noise attack, 4: LIE attack, 5 Sign flip attack, 6: Label Flipping attack
    if typeAttack == 0:
        percentageCorrupt = 0 /num_nodes
    else:
        percentageCorrupt = 2 /num_nodes
    
    corrupt = True
    save = False
    # Generate corrupt clients
    CorruptClients = CorruptGeneration(percentageCorrupt, corrupt, num_clients)
    lying_nodes = np.where(CorruptClients == 1)[0]
    true_nodes = [i for i in range(num_nodes) if i not in lying_nodes]
    all_nodes = np.union1d(lying_nodes, true_nodes)
    print("Corrupt Clients:", lying_nodes)
    tolerance=-1                #PDMM tolerance
    c=0.5                       #PDMM c
    max_iters=300             #PDMM max iterations
    when = 0
    CorruptValue = -100000000000000000
    rejection_threshold = 6
    K_decision = 5
    averaging = 1
    noise_levels = [0, 10**-1,10**-2,10**-3,10**2]  
    var = 10**2                 # Standard deviation for Gaussian noise
    mean = 0                   # Mean for Gaussian noise
    Target = np.random.randint(1, num_clients) # Target client for copycat attack
    scale = 1
    PrivacyMethod = 1      # 0: No privacy, 1: DP, 2: SMPC, 3: Subspace
    PrimModulo = [2**61 - 1] 

    neighbors_dict = {}
    for ln in lying_nodes:
        neighbors_dict[ln] = list(G.neighbors(ln))
    save = False
    print("Neighbors of lying nodes:", neighbors_dict)
    plt.figure(figsize=(8,6))

    color_map = []
    for node in range(num_nodes):
        if node in lying_nodes:
            color_map.append('red')   # corrupt
        else:
            color_map.append('blue')  # honest

    nx.draw(G, pos, with_labels=True, node_color=color_map)
    plt.title("Graph with Lying (Red) and Honest (Blue) Nodes")
    plt.show()

    # Create a subgraph containing only honest nodes
    remaining_nodes = [n for n in G.nodes() if n not in lying_nodes]
    G_sub = G.subgraph(remaining_nodes)

    still_connected = nx.is_connected(G_sub)
    print("Are the honest-only nodes still forming a connected subgraph?", still_connected)

    # Check that every honest node has a majority of honest neighbors
    all_honest = [n for n in G.nodes() if n not in lying_nodes]
    all_good = True  # Flag to track if all honest nodes pass the check

    for node in all_honest:
        neighbors = list(G.neighbors(node))
        total_neighbors = len(neighbors)
        # Count honest neighbors (neighbors that are not in lying_nodes)
        honest_neighbors = sum(1 for neighbor in neighbors if neighbor not in lying_nodes)
        
        # Only check nodes that have at least one neighbor; if a node is isolated, it might need special handling
        if total_neighbors > 0:
            if honest_neighbors <= total_neighbors / 2:
                print(f"Honest node {node} does not have majority honest neighbors: {honest_neighbors} honest out of {total_neighbors} neighbors")
                all_good = False
                
    print("Are all honest nodes connected to a majority of honest neighbors?", all_good)
    # Load dataset
    # CIFAR-10 adapted for MLP: grayscale + resize to 28×28
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, transform=transform, download=True  # False if already downloaded
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, transform=transform, download=True
    )


    # Split and corrupt datasets
    client_datasets = split_dataset(train_dataset, num_clients)
    for client_idx in lying_nodes:
        original_dataset = client_datasets[client_idx]
        if typeAttack == 6:  # Label flipping attack
            client_datasets[client_idx] = FlipLabelDataset(original_dataset)

    # Initialize models
    local_intialized_models = [Current(input_size, hidden_size, num_classes).to(device) 
                              for _ in range(num_clients)]
    criterion = nn.CrossEntropyLoss()
    # Lists to keep track of metrics

    results = [[] for _ in range(num_rounds)]
    if PrivacyMethod == 2:
        loop_params = PrimModulo
        param_name  = 'p'
    else:
        loop_params = noise_levels
        param_name  = 'noise_STD'
    for param in loop_params:
        global_train_losses = []
        test_losses = []
        test_accuracies = []
        PDMM_errors_rounds = []
        test_losses2 = []
        test_accuracies2 = []
        train_losses = []
        train_accuracies = []
        FAR_rounds, MDR_rounds = [], [] 
        epoch_losses = [None] * num_clients
        epoch_accs = [None] * num_clients
        local_models = copy.deepcopy(local_intialized_models)
        # print(f"\n--- Doing noise level {(0 if PrivacyMethod==2 else param)} with {num_rounds} rounds---")
        for round in range(num_rounds):
            # print(f'\n--- Round {round+1} of {num_rounds} ---')
               
                # Each client trains on its local data
                # local_models = [None] * num_clients

                # print(local_models)
            # Train all clients locally
            for client_idx in range(num_clients):
                local_models[client_idx], epoch_losses[client_idx], epoch_accs[client_idx] = \
                    train_local_model(client_datasets[client_idx], local_models[client_idx], num_epochs)

            # Compute per-client averages over epochs
            client_loss_means = [float(np.mean(losses)) for losses in epoch_losses]  # length = num_clients
            client_acc_means  = [float(np.mean(accs))   for accs   in epoch_accs]    # length = num_clients

            # Average across clients (this is your per-round training metric)
            avg_t_loss = float(np.mean(client_loss_means))
            avg_t_acc  = float(np.mean(client_acc_means))

            train_losses.append(avg_t_loss)
            train_accuracies.append(avg_t_acc)
            
            log_filename = None
            #log_filename = f"pdmm_iter_priv{PrivacyMethod}_atk{typeAttack}_{param_name}{param}.txt"
            #txt_name = (f"trainlog_Attack{typeAttack}_Priv{PrivacyMethod}_"
            #f"{param_name}{param}.txt")

            # ? - every 50 rounds -----------------------------------------------
            #if (round) % 50 == 0:          # +1 ? human-friendly counting
            #    with open(txt_name, "a") as fh:
            #        fh.write(
            #            f"[ROUND {round+1:4d}] "
            #            f"privacy={PrivacyMethod}  attack={typeAttack}  "
            #            f"{param_name}={param}  "
            #            f"train-loss={avg_t_loss:.4f}  train-acc={avg_t_acc:.2f}%\n"
            #        )



            Local_models_trained = copy.deepcopy(local_models)
            Local_models_trained = local_models.copy()

            Local_models_trained = CorruptData_update(CorruptClients, Local_models_trained, typeAttack, var, mean, Target,num_clients,scale)
                # print(global_model)
            global_model2 = aggregate_models2(Local_models_trained, input_size, hidden_size, num_classes, true_nodes)
            global_model22 = aggregate_models2(Local_models_trained, input_size, hidden_size, num_classes, all_nodes)

            Local_models_trained, PDMM_error, mask_history = aggregate_models(global_model2, Local_models_trained, G, tolerance, c, max_iters, rejection_threshold, K_decision, averaging, when, CorruptValue, true_nodes, False, (0 if PrivacyMethod==2 else param), PrivacyMethod, (param if PrivacyMethod==2 else 0),log_filename=log_filename)
                # print(global_model)
            PDMM_errors_rounds.append(PDMM_error)  # append per-round PDMM_error
                # Calculate edges for metrics
            edges = list(G.edges())
            honest_nodes = np.where(CorruptClients == 0)[0]
            lying_nodes = np.where(CorruptClients == 1)[0]

                # Identify target edges
            edges_honest_target = [(i, j) for (i, j) in edges if j in honest_nodes]
            edges_corrupt_target = [(i, j) for (i, j) in edges if j in lying_nodes]

                # Calculate total edges for normalization
            total_honest_edges = len(edges_honest_target)
            total_corrupt_edges = len(edges_corrupt_target)

                # Initialize metrics
            FAR_list, MDR_list = [], []
                # Calculate metrics for each iteration
            for step in range(len(mask_history)):
                mask = mask_history[step]
                    
                    # False Alarm Rate (honest edges rejected)
                false_alarms = sum(1 for (i, j) in edges_honest_target if mask[i, j] == -1)
                far = false_alarms / total_honest_edges if total_honest_edges > 0 else 0
                FAR_list.append(far)
                    
                    # Missed Detection Rate (corrupt edges not rejected)
                missed_detections = sum(1 for (i, j) in edges_corrupt_target if mask[i, j] != -1)
                mdr = missed_detections / total_corrupt_edges if total_corrupt_edges > 0 else 0
                MDR_list.append(mdr)

            # Print metrics for the current round
            FAR_rounds.append(FAR_list)
            MDR_rounds.append(MDR_list)  
            if   round < 50:      eval_every = 5
            elif round < 200:     eval_every = 10
            else:                 eval_every = 20

            if round % eval_every == 0:
                honest_models = [Local_models_trained[i] for i in true_nodes]
                loss1, acc = evaluate(honest_models, test_dataset, criterion)
                test_losses.append(loss1)
                test_accuracies.append(acc)
                test_loss2, test_accuracy2 = evaluate(global_model2, test_dataset, criterion)
                test_losses2.append(test_loss2)
                test_accuracies2.append(test_accuracy2)


        result_dict = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "privacy":   PrivacyMethod,
                "attack":    typeAttack,
                param_name:  param,
                "FAR":       FAR_rounds,
                "MDR":       MDR_rounds,
                "Error":     PDMM_errors_rounds,
                "train_loss":      train_losses,
                "train_accuracy":  train_accuracies,
                "test_loss": test_losses,
                "test_accuracy": test_accuracies,
                "test_loss2": test_losses2,
                "test_accuracy2": test_accuracies2,
        }

        # 1) keep it in-memory if you still want the combined figure
        results.append(result_dict)

        # 2) write it permanently right now
        append_result(result_dict)  

if __name__ == "__main__":
    main()