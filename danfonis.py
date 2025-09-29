import numpy as np
import torch
from torch.utils.data import DataLoader
#In these script we program our client selection strategy (called FedSignal in the paper, here refered as danfonis)
class danfonis:
    def __init__(self, num_users, initial_weights=None):
        self.num_users = num_users
        self.selection_counts = np.zeros(num_users)
        self.initial_weights = initial_weights if initial_weights is not None else np.ones(num_users)
        self.current_weights = np.copy(self.initial_weights)
        self.signatures = []
    
    def compute_signature(self, data_loader):
        images = []
        for data, _ in data_loader:
            images.append(data.numpy())
        images = np.concatenate(images, axis=0)
        
        # Check if the images are 3x64x64
        assert images.shape[1] == 3, \
            f"Unexpected image dimensions: {images.shape}"
        print(images.shape)
        
        # if images.shape[1] == 3:
        #     # Shape is 3x64x64, we need to transpose to 64x64x3
        #     images = images.transpose(0, 2, 3, 1)

        # Separate the channels
        red_channel = images[:, 0, :, :]
        green_channel = images[:, 1, :, :]
        blue_channel = images[:, 2, :, :]
        
        # Calculate mean and standard deviation for each channel
        red_mean = np.mean(red_channel)
        red_std = np.std(red_channel)
        
        green_mean = np.mean(green_channel)
        green_std = np.std(green_channel)
        
        blue_mean = np.mean(blue_channel)
        blue_std = np.std(blue_channel)
        
        # Create the result vector
        signature = np.array([red_mean, green_mean, blue_mean, red_std, green_std, blue_std])
        
        return signature
    
    def update_weights(self, selected_clients):
        for client in selected_clients:
            self.selection_counts[client] += 1
        self.current_weights = self.initial_weights / (1 + self.selection_counts)
    
    def select_clients(self, user_groups, num_clients, data_loaders):
        if len(self.signatures) == 0:
            for i in range(self.num_users):
                signature = self.compute_signature(data_loaders[i])
                self.signatures.append((signature, i))
        
        # Compute the overall mean of the first three terms of all signatures
        overall_mean = np.mean([sig[0][:3] for sig in self.signatures], axis=0)
        # Normalize mean distance to the range [0, 1]
        mean_distances = [np.linalg.norm(sig[0][:3] - overall_mean) for sig in self.signatures]
        min_distance, max_distance = min(mean_distances), max(mean_distances)

        def normalize(distance):
            if max_distance == min_distance:
                return 1  # Avoid division by zero; return 1 if all distances are the same
            return (distance - min_distance) / (max_distance - min_distance)

        def client_score(sig):
            signature, idx = sig
            mean_distance = np.linalg.norm(signature[:3] - overall_mean[:3])  # Mean proximity
            dispersion = np.linalg.norm(signature[3:])  # Data dispersion
            selection_penalty = self.current_weights[idx]  # Selection frequency penalty
            
            # Normalize and invert mean distance for scoring
            normalized_mean_distance = normalize(mean_distance)
            mean_distance_score = np.exp(-normalized_mean_distance) # Closer to overall mean is better
            
            return mean_distance_score * dispersion * selection_penalty

        # Sort clients based on the combined score
        self.signatures.sort(key=client_score, reverse=True)

        # Select the top clients
        selected_clients = [client[1] for client in self.signatures[:num_clients]]
        print("selected clients: ", selected_clients)

        # Update weights for the importance
        self.update_weights(selected_clients)
        
        return selected_clients