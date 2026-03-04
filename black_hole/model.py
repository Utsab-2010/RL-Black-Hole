import torch
import torch.nn as nn
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out



class QNetwork(nn.Module):
    def __init__(self, pos_dim=4, val_dim=4, player_dim=3, hidden_dims=[128, 64], start_val=-1000):
        super(QNetwork, self).__init__()
        self.config = {
            'pos_dim': pos_dim,
            'val_dim': val_dim,
            'player_dim': player_dim,
            'hidden_dims': hidden_dims,
            'start_val': start_val
        }
        self.start_val = start_val 

        # ResNet Config (Downscaling)
        self.map_channels = 1 # Single channel: Self (1), Opponent (-1), Empty (0)
        
        # Initial Conv (6x6)
        # 1 -> 32
        self.conv_in = nn.Conv2d(self.map_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Downscaling Blocks (ResNet-18 style expansion)
        self.layer1 = ResBlock(32, 64, stride=2)   # 6x6 -> 3x3
        self.layer2 = ResBlock(64, 128, stride=2)  # 3x3 -> 2x2
        self.layer3 = ResBlock(128, 256, stride=2) # 2x2 -> 1x1
        
        # Final flattened size: 256 * 1 * 1 = 256
        self.flattened_res_size = 256
        
        # MLP Head
        mlp_input_dim = self.flattened_res_size
        
        layers = []
        prev_dim = mlp_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim)) # Added Normalization
            layers.append(nn.ReLU())
            prev_dim = h_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_head = nn.Linear(prev_dim, 21) 

        # Indices for 21 positions in 6x6 grid
        self.pos_map = []
        idx = 0
        for r in range(6):
            for c in range(r + 1):
                self.pos_map.append((r, c))
                idx += 1
        
    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        
        # 1. Transform state to 6x6x1
        board_data = x[:, :42].view(batch_size, 21, 2)
        players = board_data[:, :, 0] # (Batch, 21) => 1 for Self, 2 for Opponent
        values = board_data[:, :, 1]  # (Batch, 21) => 1 through 10
        
        grid = torch.full((batch_size, 1, 6, 6), self.start_val, device=device, dtype=torch.float32)
        
        rows = torch.tensor([r for r, c in self.pos_map], device=device)
        cols = torch.tensor([c for r, c in self.pos_map], device=device)
        
        # Fill Lower Triangle
        # We assume input is canonicalized: Self is 1, Opponent is 2
        # Tile weights are normalized from [1, 10] to [0.1, 1.0]
        # Self tiles are positive (+), Opponent tiles are negative (-)
        normalized_values = values.float() / 10.0
        
        p1_weights = (players == 1).float() * normalized_values
        p2_weights = (players == 2).float() * -normalized_values
        grid[:, 0, rows, cols] = p1_weights + p2_weights
        
        # 2. ResNet Encoding
        out = self.conv_in(grid)
        out = self.bn_in(out)
        out = self.relu(out)
        
        out = self.layer1(out) # 32 -> 64, 3x3
        out = self.layer2(out) # 64 -> 128, 2x2
        out = self.layer3(out) # 128 -> 256, 1x1
        
        # Flatten
        out = out.view(batch_size, -1) # (B, 256)
        
        # 4. MLP
        logits = self.mlp(out) 
        output = self.output_head(logits) # (B, 21)
        
        # 5. Masking and Softmax
        filled_mask = (players != 0)
        output = output.masked_fill(filled_mask, -float('inf'))
        
        probs = torch.softmax(output, dim=1)
        final_values = probs * 100.0
        
        return final_values

class AlphaBH(nn.Module):
    def __init__(self, pos_dim=4, val_dim=4, player_dim=3, hidden_dims=[128, 64], start_val=-1000):
        super(AlphaBH, self).__init__()
        self.config = {
            'pos_dim': pos_dim,
            'val_dim': val_dim,
            'player_dim': player_dim,
            'hidden_dims': hidden_dims,
            'start_val': start_val
        }
        self.start_val = start_val 

        # ResNet Config (Downscaling)
        self.map_channels = 1 # Opponent, Player
        
        # Initial Conv (6x6)
        # 2 -> 32
        self.conv_in = nn.Conv2d(self.map_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Downscaling Blocks (ResNet-18 style expansion)
        self.layer1 = ResBlock(32, 64, stride=2)   # 6x6 -> 3x3
        self.layer2 = ResBlock(64, 128, stride=2)  # 3x3 -> 2x2
        self.layer3 = ResBlock(128, 256, stride=2) # 2x2 -> 1x1
        
        # Final flattened size: 256 * 1 * 1 = 256
        self.flattened_res_size = 256
        
        # Common MLP Body
        mlp_input_dim = self.flattened_res_size
        
        layers = []
        prev_dim = mlp_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim)) # Added Normalization
            layers.append(nn.ReLU())
            prev_dim = h_dim
        
        self.body = nn.Sequential(*layers)
        
        # --- Dual Heads ---
        # Policy Head: Logits for 21 actions
        self.policy_head = nn.Linear(prev_dim, 21)
        
        # Value Head: Scalar Value [-1, 1]
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

        # Indices for 21 positions in 6x6 grid
        self.pos_map = []
        idx = 0
        for r in range(6):
            for c in range(r + 1):
                self.pos_map.append((r, c))
                idx += 1
        
    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        
        # 1. Transform state to 6x6x1
        board_data = x[:, :42].view(batch_size, 21, 2)
        players = board_data[:, :, 0] # (Batch, 21) => 1 for Self, 2 for Opponent
        values = board_data[:, :, 1]  # (Batch, 21) => 1 through 10
        
        grid = torch.full((batch_size, 1, 6, 6), self.start_val, device=device, dtype=torch.float32)
        
        rows = torch.tensor([r for r, c in self.pos_map], device=device)
        cols = torch.tensor([c for r, c in self.pos_map], device=device)
        
        # Fill Lower Triangle
        # We assume input is canonicalized: Self is 1, Opponent is 2
        # Tile weights are normalized from [1, 10] to [0.1, 1.0]
        # Self tiles are positive (+), Opponent tiles are negative (-)
        normalized_values = values.float() / 10.0
        
        p1_weights = (players == 1).float() * normalized_values
        p2_weights = (players == 2).float() * -normalized_values
        grid[:, 0, rows, cols] = p1_weights + p2_weights
        
        # 2. ResNet Encoding
        out = self.conv_in(grid)
        out = self.bn_in(out)
        out = self.relu(out)
        
        out = self.layer1(out) # 32 -> 64, 3x3
        out = self.layer2(out) # 64 -> 128, 2x2
        out = self.layer3(out) # 128 -> 256, 1x1
        
        # Flatten
        out = out.view(batch_size, -1) # (B, 256)
        
        # 4. Body
        features = self.body(out)
        
        # 5. Heads
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value

def preprocess_batch(obs, device):
    # obs is dictionary of stacked arrays from VectorEnv
    # board: (B, 21, 2)
    # The current_tile is ignored by the actual networks now, so we just
    # flatten the board identically padding behavior for legacy inputs.
    
    board = torch.tensor(obs["board"], dtype=torch.float32, device=device)
    board_flat = board.view(board.shape[0], -1) # (B, 42)
    
    # We still append a dummy 0 for current_tile so old shapes (43) don't break scripts
    # before they are updated, though the network actually only reads the first 42.
    current_tile = torch.zeros((board.shape[0], 1), dtype=torch.float32, device=device)
    
    features = torch.cat([board_flat, current_tile], dim=1) # (B, 43)
    return features

def get_action_mask_batch(obs, device):
    # obs["board"]: (B, 21, 2)
    board = torch.tensor(obs["board"], device=device)
    # Valid where idx 0 (player) is 0
    valid = (board[:, :, 0] == 0) # (B, 21)
    return valid

def preprocess_obs(obs, device):
    # Single obs version (for test script / non-vector train)
    # Input is dict with arrays (not batched)
    # Return (43,) torch tensor? Or (1, 43)?
    # Usually we want (43,) and unsqueeze later if needed, OR consistently return batch.
    
    # Original 'preprocess_obs' returned (43,).
    
    board_flat = torch.tensor(obs["board"].flatten(), dtype=torch.float32, device=device)
    current_tile = torch.tensor([obs["current_tile"]], dtype=torch.float32, device=device)
    features = torch.cat([board_flat, current_tile])
    return features

def get_action_mask(obs, device):
    # Single env mask
    board = torch.tensor(obs["board"], device=device)
    valid = (board[:, 0] == 0)
    return valid
