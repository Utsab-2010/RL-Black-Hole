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

def get_sinusoidal_embeddings(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    # div_term = 10000^(2i/d_model)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class QNetwork(nn.Module):
    def __init__(self, pos_dim=4, val_dim=4, player_dim=3, hidden_dims=[128, 64], start_val=-10):
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
        self.map_channels = 2 # Opponent, Player
        
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
        
        # Turn Embedding (Fixed Sinusoidal)
        self.turn_dim = 16
        # Register as buffer so it's saved with state_dict but not optimized
        self.register_buffer('turn_emb_table', get_sinusoidal_embeddings(22, self.turn_dim))
        
        # MLP Head
        mlp_input_dim = self.flattened_res_size + self.turn_dim
        
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
        
        # 1. Transform state to 6x6x2
        board_data = x[:, :42].view(batch_size, 21, 2)
        players = board_data[:, :, 0] # (Batch, 21)
        
        grid = torch.full((batch_size, 2, 6, 6), self.start_val, device=device, dtype=torch.float32)
        
        rows = torch.tensor([r for r, c in self.pos_map], device=device)
        cols = torch.tensor([c for r, c in self.pos_map], device=device)
        
        # Fill Lower Triangle
        # Plane 1: Player 1 (val 1), Plane 2: Player 2 (val 2)
        grid[:, 0, rows, cols] = (players == 1).float() * 1.0
        grid[:, 1, rows, cols] = (players == 2).float() * 2.0
        
        # 2. ResNet Encoding
        out = self.conv_in(grid)
        out = self.bn_in(out)
        out = self.relu(out)
        
        out = self.layer1(out) # 32 -> 64, 3x3
        out = self.layer2(out) # 64 -> 128, 2x2
        out = self.layer3(out) # 128 -> 256, 1x1
        
        # Flatten
        out = out.view(batch_size, -1) # (B, 256)
        
        # 3. Turn Encoding
        turn_count = (players != 0).sum(dim=1) # (B,)
        turn_emb = self.turn_emb_table[turn_count] # (B, 16)
        
        # Concatenate
        combined = torch.cat([out, turn_emb], dim=1)
        
        # 4. MLP
        logits = self.mlp(combined) 
        output = self.output_head(logits) # (B, 21)
        
        # 5. Masking and Softmax
        filled_mask = (players != 0)
        output = output.masked_fill(filled_mask, -float('inf'))
        
        probs = torch.softmax(output, dim=1)
        final_values = probs * 100.0
        
        return final_values

def preprocess_batch(obs, device):
    # obs is dictionary of stacked arrays from VectorEnv
    # board: (B, 21, 2)
    # current_tile: (B,)
    
    board = torch.tensor(obs["board"], dtype=torch.float32, device=device)
    board_flat = board.view(board.shape[0], -1) # (B, 42)
    
    current_tile = torch.tensor(obs["current_tile"], dtype=torch.float32, device=device).unsqueeze(1) # (B, 1)
    
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
