import os
import shutil
import random
import networkx as nx
import torch
import numpy as np
import umap
import open3d as o3d

from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

from source.generative.model import PointCloudAE
from source.semseg.model import PointNet2SemSegMsg
from source.generative.dataset import get_train_val_file_paths
from source.generative.dataset import PointCloudInstancesDataset


SEM_CLASSES = {
    'bus': 0,
    'car': 1,
    'motorcycle': 2,
    'airplane': 3,
    'boat': 4,
    'tower': 5
}

PALETTE = np.array([
    [200,  50,  50],  # 0 bus
    [ 35, 142,  35],  # 1 car
    [  0,   0, 255],  # 2 motorcycle
    [255, 165,   0],  # 3 airplane
    [255,   0, 255],  # 4 boat
    [120, 120, 120],  # 5 train
], dtype=np.uint8)


"""
[200, 50, 50] dark red
[35, 142, 35] forest green
[0, 0, 255]  pure blue
[255, 165, 0] orange
[255, 0, 255]  magenta
[120, 120, 120]  medium gray
[255, 255, 0]  yellow
"""

def plot_tradeoff(x, y, z, color='b', marker='o', output_dir=''):
    if not (len(x) == len(y) == len(z)):
        raise ValueError("x, y, z must have the same length.")
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=color, marker=marker, s=30)

    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Validity")
    ax.set_zlabel("Similarity")
    ax.set_title("Trade-Off-3D")

    plt.savefig(f'{output_dir}/trade-off-3D-plot.jpg')
    plt.close()

def compute_sparsity_nn(original: np.ndarray, counterfactual: np.ndarray, eps: float) -> float:
    if original.shape != counterfactual.shape or original.ndim != 2 or original.shape[1] != 3:
        raise ValueError("Inputs must both be (N, 3) arrays of the same shape.")
    if original.size == 0:
        return 0.0

    eps2 = eps * eps
    try:
        tree = cKDTree(counterfactual)
        dists, _ = tree.query(original, k=1, workers=-1)
        return float((dists * dists > eps2).mean())
    except Exception:
        diff = original[:, None, :] - counterfactual[None, :, :]
        sq = np.einsum("ijk,ijk->ij", diff, diff)
        min_sq = sq.min(axis=1)
        return float((min_sq > eps2).mean())

def compute_validity_nn(original_points: np.ndarray, counterfactual_points: np.ndarray, softmax_orig: np.ndarray, softmax_cf: np.ndarray) -> float:
    if original_points.shape[0] != softmax_orig.shape[0]:
        raise ValueError("original_xyz and softmax_orig must align in N")
    if counterfactual_points.shape[0] != softmax_cf.shape[0]:
        raise ValueError("counterfactual_xyz and softmax_cf must align in M")

    c_star = np.argmax(softmax_orig, axis=1)

    tree = KDTree(counterfactual_points)
    nn_idx = tree.query(original_points, k=1, return_distance=False).ravel()

    p_orig = softmax_orig[np.arange(len(c_star)), c_star]
    p_cf   = softmax_cf[nn_idx, c_star]

    return float(np.mean(np.abs(p_cf - p_orig)))

def compute_similarity_nn(original: np.ndarray, counterfactual: np.ndarray) -> float:
    if original.ndim != 2 or counterfactual.ndim != 2 or original.shape[1] != counterfactual.shape[1]:
        raise ValueError("Both point clouds must have the same dimensions (N,3) and (M,3).")
    if original.size == 0 or counterfactual.size == 0:
        return 0.0

    tree1 = KDTree(original)
    dists1, _ = tree1.query(counterfactual, k=1, return_distance=True)
    tree2 = KDTree(counterfactual)
    dists2, _ = tree2.query(original, k=1, return_distance=True)

    cd = np.mean(dists1**2) + np.mean(dists2**2)
    return cd / (1.0 + cd)


def main(seed: int, first_class: str, second_class: str, results_dir: str = ''):
    print(f'Running experiment with random seed: {seed} for classes: {first_class} and {second_class}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.cuda.empty_cache()
    if results_dir == '':
        results_dir = 'results/'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    OUTPUT_DIR: str = os.path.join(results_dir, f'{first_class}_{second_class}_{seed}/')
    SOURCE_CLASS: int = SEM_CLASSES[first_class]
    TARGET_CLASS: int = SEM_CLASSES[second_class]

    DATA_DIR: str = '../data/processed/'
    
    batch_size: int = 32

    def clear_directory(dir_path: str):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    clear_directory(OUTPUT_DIR)

    train_paths, val_paths = get_train_val_file_paths(DATA_DIR, split=0.4)

    train_dataset = PointCloudInstancesDataset()
    train_dataset.set_file_paths(train_paths)

    test_dataset = PointCloudInstancesDataset()
    test_dataset.set_file_paths(val_paths)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    print(f'Found {len(train_dataset)} train samples!')
    print(f'Found {len(test_dataset)} test samples!')

    AUTOENCODER_WEIGHTS_PATH = 'weights/AE-PCD-epoch-800.pth'
    CLASSIFIER_WEIGHTS_PATH = 'weights/PointNet2SemSegMsg-epoch-100.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes: int = 6
    num_points: int = 4096
    latent_dim: int = 1024
    in_channels: int = 3
    ALPHA: float = 1.0
    BETA: float = 1.0
    N_OTHER: int = 10
    overwrite: bool = False

    latents_path = os.path.join(OUTPUT_DIR, 'train_latents.pth')
    logits_path = os.path.join(OUTPUT_DIR, 'train_logits.pth')

    autoencoder = PointCloudAE(in_channels=in_channels, num_points=num_points, latent_dim=latent_dim)
    ae_state = torch.load(AUTOENCODER_WEIGHTS_PATH, map_location=device)
    autoencoder.load_state_dict(ae_state)
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    classifier = PointNet2SemSegMsg(num_classes=num_classes)
    cls_state = torch.load(CLASSIFIER_WEIGHTS_PATH, map_location=device)
    classifier.load_state_dict(cls_state)
    classifier = classifier.to(device)
    classifier.eval()

    if not os.path.exists(latents_path) and not os.path.exists(logits_path) or overwrite:
        train_latents = []
        train_logits = []

        with torch.no_grad():
            for points, _ in tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9):
                points = points.float().contiguous().to(device)
                points = points.transpose(2, 1)

                _, z = autoencoder(points)

                for batch_idx in range(z.size(0)):
                    single_z = z[batch_idx, :]
                    train_latents.append(single_z.detach().cpu())

                softmax_probabilities = classifier(points)
                for batch_idx in range(softmax_probabilities.size(0)):
                    single_softmax_probability = softmax_probabilities[batch_idx]
                    train_logits.append(single_softmax_probability.detach().cpu())

            
        train_latents = torch.stack(train_latents, dim=0)
        train_logits = torch.stack(train_logits, dim=0)
        train_logits = torch.mean(train_logits, dim=1)

        print(f'Shape Train Latents: {train_latents.shape}')
        print(f'Shape Train logits: {train_logits.shape}')

        torch.save(train_latents, os.path.join(OUTPUT_DIR, 'train_latents.pth'))
        torch.save(train_logits, os.path.join(OUTPUT_DIR, 'train_logits.pth'))
    else:
        train_latents = torch.load(latents_path).cpu()
        train_logits = torch.load(logits_path).cpu()

    umap_model = umap.UMAP(n_components=2, n_neighbors=5, metric='cosine')
    embedding = umap_model.fit_transform(train_latents)

    triangulation = Delaunay(embedding)

    kde = KernelDensity(bandwidth=0.1).fit(embedding)
    log_density = kde.score_samples(embedding)
    density = np.exp(log_density)
    scaler = MinMaxScaler()
    normalized_densities = scaler.fit_transform(density.reshape(-1, 1)).flatten()

    train_logits = train_logits.data.numpy()

    target_class = TARGET_CLASS
    mask = np.all(
        train_logits[:, target_class][:, None] > np.delete(train_logits, target_class, axis=1),
        axis=1
    )

    valid_target_indices = np.where(mask)[0]

    G = nx.Graph()
    for idx in range(len(embedding)):
        G.add_node(idx, logits=train_logits[idx],
                   logit_target=train_logits[idx][TARGET_CLASS],
                   ypred=np.argmax(train_logits[idx]))

    for simplex in triangulation.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                p1, p2 = simplex[i], simplex[j]
                d = np.linalg.norm(embedding[p1] - embedding[p2])

                weight_density_edge = 1 - ((density[p1] + density[p2]) / 2.0)
                weight_logit_edge = 1 - ((train_logits[p1][TARGET_CLASS] + train_logits[p2][TARGET_CLASS]) / 2.0)

                G.add_edge(p1, p2, distance_edge=d, weight_density=weight_density_edge,
                           weight_logit=weight_logit_edge,
                           weight=ALPHA * weight_density_edge + BETA * weight_logit_edge)

    # Set threshold to the 99th percentile of distances
    DISTANCE_THRESHOLD = np.percentile([G.edges[e]['distance_edge'] for e in G.edges()], 99)
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['distance_edge'] > DISTANCE_THRESHOLD]
    G.remove_edges_from(edges_to_remove)

###################################################################################################################
#
###################################################################################################################
    from matplotlib.collections import LineCollection

    # -----------------------------
    # Build edge segments from pruned NX graph (left panel)
    # -----------------------------
    pos = embedding  # (N,2) array
    edges = np.array(list(G.edges()))
    segs = np.stack([pos[edges[:,0]], pos[edges[:,1]]], axis=1)  # (E,2,2)

    # Optional: use edge weights for alpha (lighter for heavier weight)
    edge_w = np.array([G[u][v]['weight'] for u, v in edges])
    edge_w_norm = (edge_w - edge_w.min()) / (edge_w.max() - edge_w.min() + 1e-12)
    edge_alpha = 0.15 + 0.65 * (1.0 - edge_w_norm)  # higher weight → lighter

    # -----------------------------
    # Evaluate KDE on a grid (right panel heatmap)
    # -----------------------------
    pad = 0.05
    (xmin, ymin), (xmax, ymax) = pos.min(0) - pad, pos.max(0) + pad
    grid_res = 300
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, grid_res),
        np.linspace(ymin, ymax, grid_res)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    logp = kde.score_samples(grid)
    p = np.exp(logp).reshape(xx.shape)
    p_norm = (p - p.min()) / (p.max() - p.min() + 1e-12)

    # -----------------------------
    # Figure
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # (1) Left: pruned NX graph edges + nodes colored by predicted class (unchanged)
    ax = axes[0]
    lc = LineCollection(segs, linewidths=0.5, alpha=None)  # set globally next line
    lc.set_alpha(edge_alpha if edge_alpha.ndim == 1 else 0.5)
    ax.add_collection(lc)
    ax.scatter(pos[:,0], pos[:,1], s=10, c=(PALETTE[np.argmax(train_logits, axis=1)]/255.0), edgecolors='none')
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Pruned graph on UMAP space")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    # (2) Right: KDE heatmap + points colored by *their own* KDE value
    ax = axes[1]
    im = ax.imshow(
        p_norm, origin='lower', extent=[xmin, xmax, ymin, ymax],
        interpolation='bilinear', aspect='equal'
    )

    # color each point by KDE at its location (no class colors)
    # use the already computed per-point normalized densities
    ax.scatter(pos[:,0], pos[:,1], s=10, c=normalized_densities, cmap='viridis',
            edgecolors='black', linewidths=0.2, alpha=0.9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("KDE-estimated plausibility (normalized)")
    ax.set_title("KDE heatmap over UMAP space")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    plt.savefig(os.path.join(OUTPUT_DIR, f"{first_class}_{second_class}_umap_graphKDE_side_by_side.jpg"), dpi=300)
    plt.close()

########################################################################################################################
    def_valid_target_indices = []

    for node in valid_target_indices:
        neighbors = list(G.neighbors(node))
        n_other = 1
        if len(neighbors) < n_other:
            continue
        count = sum(1 for neighbor in neighbors if neighbor in valid_target_indices)
        if count >= N_OTHER:
            def_valid_target_indices.append(node)

    test_latents = []
    test_logits = []

    test_latents_path = os.path.join(OUTPUT_DIR, 'test_latents.pth')
    test_logits_path = os.path.join(OUTPUT_DIR, 'test_logits.pth')

    if not os.path.exists(test_logits_path) or overwrite:
        with torch.no_grad():
            for points, _ in tqdm(test_dataloader, total=len(test_dataloader), smoothing=0.9):
                points = points.float().contiguous().to(device)
                points = points.transpose(2, 1)

                softmax_probabilities = classifier(points)
                for batch_idx in range(softmax_probabilities.size(0)):
                    test_logits.append(softmax_probabilities[batch_idx].detach().cpu())
        
        test_logits = torch.stack(test_logits, dim=0)
        test_logits = torch.mean(test_logits, dim=1)
        torch.save(test_logits, test_logits_path)
    else:
        test_logits = torch.load(test_logits_path).cpu()

    test_predictions = test_logits.argmax(dim=1).data.numpy()

    indices_source_class = np.where(test_predictions == SOURCE_CLASS)[0]

    if len(indices_source_class) == 0:
        raise ValueError(f"No test instances found with predicted class {SOURCE_CLASS}")

    # Randomly select one test instance from the source class
    INDEX_TEST = np.random.choice(indices_source_class)
    x_test = test_dataset[INDEX_TEST]
    input_tensor = x_test[0].float().contiguous().unsqueeze(0).transpose(2, 1).to(device)

    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(x_test[0].data.numpy())
    o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, f'original.pcd'), original_pcd)

    y_pred_test = classifier(input_tensor).mean(dim=1).argmax(dim=1)
    y_pred_test = y_pred_test.detach().cpu().numpy()

    print(f'y_pred_test: {y_pred_test}')
    print(f'y_pred_test shape: {y_pred_test.shape}')
    _, x_test_latent = autoencoder(input_tensor)
    x_test_latent = x_test_latent.detach().cpu().numpy()
    x_test_embedding = umap_model.transform(x_test_latent)

    distances = np.linalg.norm(embedding - x_test_embedding, axis=1)
    closest_index = np.argmin(distances)

    # Shortest paths from source to nodes with target class
    all_paths = nx.single_source_dijkstra_path(G, source=closest_index, weight='weight')
    all_costs = nx.single_source_dijkstra_path_length(G, source=closest_index, weight='weight')

    # Filter valid class-dominant targets based on indices with the target class
    target_paths = {i: all_paths[i] for i in all_paths if all_paths[i][-1] in def_valid_target_indices}
    target_costs = {i: all_costs[i] for i in all_paths if all_paths[i][-1] in def_valid_target_indices}

    # Find the min cost path to the target class
    if target_costs:
        closest_target = min(target_costs, key=target_costs.get)
        closest_path = target_paths[closest_target]
        path = embedding[target_paths[closest_target]]
    else:
        path = None

    try:
        print(f'LENGTH OF PATH VALUES: {len(path)}')
    
        plt.figure(figsize=(10, 10))
        pos = {i: embedding[i] for i in range(len(embedding))}
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        color_palette = PALETTE / 255.0
        node_colors = [color_palette[np.argmax(G.nodes[node]['logits'])] for node in G.nodes()]
        nx.draw(G, pos, with_labels=False, node_size=10, node_color=node_colors, edge_color=edge_weights,
                edge_cmap=plt.cm.viridis, width=0.5, alpha=0.7)
        plt.scatter(x_test_embedding[0, 0], x_test_embedding[0, 1], color=color_palette[int(y_pred_test[0])], edgecolors='black', s=100,
                    label='Test Point', zorder=5)
        """
        if path is not None:
        """

        # Plot the edges along the shortest path
        for i in range(0, len(closest_path) - 1):
            p1 = embedding[closest_path[i]]
            p2 = embedding[closest_path[i + 1]]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='blue', linewidth=3, zorder=6)

        plt.scatter(embedding[closest_target, 0], embedding[closest_target, 1],
                    color=node_colors[closest_target], s=100, edgecolors='black', label='Closest Valid Target',
                    zorder=5)
        plt.title("Shortest Path to Class-Dominant Points (Target Class: {})".format(target_class))
        plt.legend()
        plt.xlabel("UMAP Component 1")
        plt.ylabel("UMAP Component 2")
        plt.savefig(f'{OUTPUT_DIR}/{first_class}_{second_class}_umap_cfe_path.jpg')
        plt.close()

        if path is not None:
            n_steps = 3
            interpolated_path = []

            path_latent_indices = closest_path
            path = train_latents[path_latent_indices]

            # Get the test point latent
            p1 = torch.tensor(x_test_latent.squeeze(0))
            p2 = path[0]  # First latent in path

            for t in np.linspace(0, 1, n_steps + 1)[:-1]:
                interpolated_point = (1 - t) * p1 + t * p2
                interpolated_path.append(interpolated_point)

            # Intermediate counterfactuals
            for i in range(len(path) - 2):
                p1 = path[i]
                p2 = path[i + 1]
                for t in np.linspace(0, 1, n_steps + 2)[1:-1]:
                    interpolated_point = (1 - t) * p1 + t * p2
                    interpolated_path.append(interpolated_point)

            # last latent in path
            p1 = path[-2]
            p2 = path[-1]
            for t in np.linspace(0, 1, n_steps + 1)[1:]:
                interpolated_point = (1 - t) * p1 + t * p2
                interpolated_path.append(interpolated_point)

            interpolated_path = torch.stack(interpolated_path).to(device)
            
            decoder = autoencoder.decoder

            source_instance: np.ndarray = np.asarray([])
            source_softmax: np.ndarray = np.asarray([])

            similarities = []
            validities = []
            sparsities = []

            for idx in range(interpolated_path.shape[0]):
                z = interpolated_path[idx].unsqueeze(0)

                with torch.no_grad():
                    output = decoder(z)

                    output_np = output.transpose(2, 1).squeeze(0).detach().cpu().numpy()

                    logits = classifier(output)

                labels = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
                logits = logits.squeeze(0).detach().cpu().numpy()

                colors = PALETTE[labels].astype(np.float32) / 255.0

                if idx == 0:
                    source_instance = output_np
                    source_softmax = logits

                sparsity_val = compute_sparsity_nn(source_instance, output_np, eps=0.1)
                sparsities.append(sparsity_val)
                similarity_val = compute_similarity_nn(source_instance, output_np)
                similarities.append(similarity_val)
                validity_val = compute_validity_nn(source_instance, output_np, source_softmax, logits)
                validities.append(validity_val)

                print(f'Similarity: {similarity_val} for counterfactual: {idx}')
                print(f'Validity: {validity_val} for counterfactual: {idx}')
                print(f'Sparsity: {sparsity_val} for counterfactual: {idx}')

                pcd_output = o3d.geometry.PointCloud()
                pcd_output.points = o3d.utility.Vector3dVector(output_np)
                pcd_output.colors = o3d.utility.Vector3dVector(colors)

                o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, f'output_{idx}.pcd'), pcd_output)
            
            similarities = np.asarray(similarities)
            validities = np.asarray(validities)
            sparsities = np.asarray(sparsities)

            N = len(similarities)
            t = np.arange(1, N + 1)

            plt.plot(t, similarities, marker="o", label="Similarity")
            plt.plot(t, validities, marker="s", label="Validity")
            plt.plot(t, sparsities, marker="^", label="Sparsity")

            plt.xlabel("Interpolation step")
            plt.ylabel("Metric Score")
            plt.legend()
            plt.tight_layout()

            plt.savefig(f"{OUTPUT_DIR}/{first_class}_{second_class}_metrics.jpg", dpi=300)
            plt.close()

            plot_tradeoff(sparsities, validities, similarities, output_dir=OUTPUT_DIR)

            return similarities, sparsities, validities
    except:
        print(f'A path was not found unfortunately!')
        return [], [], []

#############################################################
# For quanititative analysis - multiple runs ################
# ###########################################################    
import numpy as np
import matplotlib.pyplot as plt

def _interp_to_grid(series_list, G=100):
    g = np.linspace(0, 1, G)
    Y = []
    for y in series_list:
        y = np.asarray(y, float)
        yi = np.full_like(g, y[0]) if len(y)==1 else np.interp(g, np.linspace(0,1,len(y)), y)
        Y.append(yi)
    return g, np.vstack(Y)

def _band(Y, mode="iqr"):
    if mode == "sem":
        c = np.nanmean(Y, 0)
        s = np.nanstd(Y, 0, ddof=1) / np.sqrt(max(Y.shape[0],1))
        lo, hi = c - s, c + s
    elif mode == "std":
        c = np.nanmean(Y, 0)
        s = np.nanstd(Y, 0, ddof=1)
        lo, hi = c - s, c + s
    else:  # "iqr"
        c = np.nanmedian(Y, 0)
        lo = np.nanpercentile(Y, 25, 0); hi = np.nanpercentile(Y, 75, 0)
    return c, np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)

def plot_metrics_clean(runs, metrics=("similarity","validity","sparsity"),
                       band="iqr", title=None, savepath=None):
    fig, axes = plt.subplots(len(metrics), 1, sharex=True, figsize=(7, 7))
    if len(metrics)==1: axes = [axes]
    g = np.linspace(0,1,100)

    for ax, m in zip(axes, metrics):
        series = [r[m] for r in runs if m in r and len(r[m])>0]
        if not series: 
            ax.set_visible(False); continue
        g, Y = _interp_to_grid(series, G=200)
        c, lo, hi = _band(Y, mode=band)

        # no spaghetti → much clearer
        ax.plot(g, c, lw=2, label=m)
        ax.fill_between(g, lo, hi, alpha=0.18, label=f"{m} {band.upper()}")

        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        ax.legend()

    axes[-1].set_xlabel("Interpolation progress")
    if title: fig.suptitle(title, y=0.98)
    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=300); plt.close(fig)

seeds: list[int] = [1, 8, 24, 42]
    
def run_experiments():
    runs = []
    first_class = 'car'
    second_class = 'bus'
    results_dir: str = 'results_multiple_runs/'
    for seed in seeds:
        run = {}
        similarity, sparsity, validity = main(seed, first_class, second_class, results_dir)
        if len(similarity) == 0 or len(sparsity) == 0 or len(validity) == 0:
            continue

        run['similarity'] = similarity
        run['sparsity'] = sparsity
        run['validity'] = validity
        runs.append(run)
    
    plot_metrics_clean(
    runs,
    metrics=("similarity","validity","sparsity"),
    band="sem",
    title=f"10 runs {first_class} vs {second_class}",
    savepath=f"{results_dir}/{first_class}_{second_class}_metrics_small_multiples.jpg")

    def summarize_runs(runs, metric):
        data = np.array([run[metric] for run in runs])  # shape: (n_runs, n_steps)
        mean = data.mean(axis=0)
        sem = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
        return mean, sem

    mean_sim, sem_sim = summarize_runs(runs, "similarity")
    mean_val, sem_val = summarize_runs(runs, "validity")
    mean_spar, sem_spar = summarize_runs(runs, "sparsity")

    print("Similarity (mean ± SEM):")
    for m, s in zip(mean_sim, sem_sim):
        print(f"{m:.3f} ± {s:.3f}")

    print("Validity (mean ± SEM):")
    for m, s in zip(mean_val, sem_val):
        print(f"{m:.3f} ± {s:.3f}")

    print("Similarity (mean ± SEM):")
    for m, s in zip(mean_spar, sem_spar):
        print(f"{m:.3f} ± {s:.3f}")
########################################################################################

def run_pair():
    seed: int = 42
    first_class = 'car'
    second_class = 'tower'

    main(seed, first_class, second_class)


if __name__ == "__main__":
    run_pair()
    #run_experiments()