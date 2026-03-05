import torch
import torch.nn as nn
import numpy as np
from .game import DEFAULT_LAYERS

# Derived constants for the default configuration
_DEFAULT_LAYERS = DEFAULT_LAYERS
_DEFAULT_NUM_HEXES = (_DEFAULT_LAYERS * (_DEFAULT_LAYERS + 1)) // 2  # 45
_DEFAULT_TILES_PER_PLAYER = (_DEFAULT_NUM_HEXES - 1) // 2  # 22


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


def _compute_flattened_size(layers):
    """Compute the flattened spatial size after 3 stride-2 ResNet blocks on an LxL input."""
    size = layers
    for _ in range(3):
        size = (size + 1) // 2  # ceil division for stride=2
    return 256 * size * size


class QNetwork(nn.Module):
    def __init__(self, layers=_DEFAULT_LAYERS, hidden_dims=[128, 64], start_val=-1000):
        super(QNetwork, self).__init__()
        self.layers = layers
        self.num_hexes = (layers * (layers + 1)) // 2
        self.tiles_per_player = (self.num_hexes - 1) // 2
        self.start_val = start_val

        self.config = {
            'layers': layers,
            'hidden_dims': hidden_dims,
            'start_val': start_val
        }

        self.map_channels = 1

        # Initial Conv (LxL)
        self.conv_in = nn.Conv2d(self.map_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Downscaling ResNet Blocks
        self.res_layer1 = ResBlock(32, 64, stride=2)
        self.res_layer2 = ResBlock(64, 128, stride=2)
        self.res_layer3 = ResBlock(128, 256, stride=2)

        self.flattened_res_size = _compute_flattened_size(layers)

        # MLP Head
        mlp_input_dim = self.flattened_res_size
        mlp_layers = []
        prev_dim = mlp_input_dim
        for h_dim in hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, h_dim))
            mlp_layers.append(nn.BatchNorm1d(h_dim))
            mlp_layers.append(nn.ReLU())
            prev_dim = h_dim

        self.mlp = nn.Sequential(*mlp_layers)
        self.output_head = nn.Linear(prev_dim, self.num_hexes)

        # Triangular position map: maps hex index -> (row, col) in LxL grid
        self.pos_map = [(r, c) for r in range(layers) for c in range(r + 1)]

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device

        # 1. Transform state to LxLx1
        flat_board_size = self.num_hexes * 2
        board_data = x[:, :flat_board_size].view(batch_size, self.num_hexes, 2)
        players = board_data[:, :, 0]  # (B, num_hexes)
        values  = board_data[:, :, 1]  # (B, num_hexes)

        grid = torch.full((batch_size, 1, self.layers, self.layers), self.start_val, device=device, dtype=torch.float32)

        rows = torch.tensor([r for r, c in self.pos_map], device=device)
        cols = torch.tensor([c for r, c in self.pos_map], device=device)

        # Signed normalised tile weights
        normalized_values = values.float() / float(self.tiles_per_player)
        p1_weights = (players == 1).float() * normalized_values
        p2_weights = (players == 2).float() * -normalized_values
        grid[:, 0, rows, cols] = p1_weights + p2_weights

        # 2. ResNet Encoding
        out = self.relu(self.bn_in(self.conv_in(grid)))
        out = self.res_layer1(out)
        out = self.res_layer2(out)
        out = self.res_layer3(out)

        # Flatten
        out = out.view(batch_size, -1)

        # 3. MLP
        logits = self.mlp(out)
        output = self.output_head(logits)  # (B, num_hexes)

        # 4. Masking
        filled_mask = (players != 0)
        output = output.masked_fill(filled_mask, -float('inf'))

        probs = torch.softmax(output, dim=1)
        final_values = probs * 100.0

        return final_values


class AlphaBH(nn.Module):
    def __init__(self, layers=_DEFAULT_LAYERS, hidden_dims=[128, 64], start_val=-1000):
        super(AlphaBH, self).__init__()
        self.layers = layers
        self.num_hexes = (layers * (layers + 1)) // 2
        self.tiles_per_player = (self.num_hexes - 1) // 2
        self.start_val = start_val

        self.config = {
            'layers': layers,
            'hidden_dims': hidden_dims,
            'start_val': start_val
        }

        self.map_channels = 1

        # Initial Conv (LxL)
        self.conv_in = nn.Conv2d(self.map_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Downscaling Blocks
        self.layer1 = ResBlock(32, 64, stride=2)
        self.layer2 = ResBlock(64, 128, stride=2)
        self.layer3 = ResBlock(128, 256, stride=2)

        self.flattened_res_size = _compute_flattened_size(layers)

        # Common MLP Body
        mlp_layers = []
        prev_dim = self.flattened_res_size
        for h_dim in hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, h_dim))
            mlp_layers.append(nn.BatchNorm1d(h_dim))
            mlp_layers.append(nn.ReLU())
            prev_dim = h_dim

        self.body = nn.Sequential(*mlp_layers)

        # Policy Head: logits for each board position
        self.policy_head = nn.Linear(prev_dim, self.num_hexes)

        # Value Head: scalar [-1, 1]
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

        # Triangular position map
        self.pos_map = [(r, c) for r in range(layers) for c in range(r + 1)]

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device

        # 1. Transform state to LxLx1
        flat_board_size = self.num_hexes * 2
        board_data = x[:, :flat_board_size].view(batch_size, self.num_hexes, 2)
        players = board_data[:, :, 0]  # (B, num_hexes)
        values  = board_data[:, :, 1]  # (B, num_hexes)

        grid = torch.full((batch_size, 1, self.layers, self.layers), self.start_val, device=device, dtype=torch.float32)

        rows = torch.tensor([r for r, c in self.pos_map], device=device)
        cols = torch.tensor([c for r, c in self.pos_map], device=device)

        # Signed normalised tile weights
        normalized_values = values.float() / float(self.tiles_per_player)
        p1_weights = (players == 1).float() * normalized_values
        p2_weights = (players == 2).float() * -normalized_values
        grid[:, 0, rows, cols] = p1_weights + p2_weights

        # 2. ResNet Encoding
        out = self.relu(self.bn_in(self.conv_in(grid)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Flatten
        out = out.view(batch_size, -1)

        # 3. Body
        features = self.body(out)

        # 4. Heads
        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        return policy_logits, value


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def preprocess_batch(obs, device):
    """Batch preprocessing for VectorEnv. obs['board'] is (B, N, 2)."""
    board = torch.tensor(obs["board"], dtype=torch.float32, device=device)
    board_flat = board.view(board.shape[0], -1)  # (B, N*2)
    current_tile = torch.zeros((board.shape[0], 1), dtype=torch.float32, device=device)
    features = torch.cat([board_flat, current_tile], dim=1)  # (B, N*2 + 1)
    return features


def get_action_mask_batch(obs, device):
    """Batch action mask. Returns (B, N) bool tensor; True = valid."""
    board = torch.tensor(obs["board"], device=device)
    valid = (board[:, :, 0] == 0)  # (B, N)
    return valid


def preprocess_obs(obs, device):
    """Single observation preprocessing."""
    board_flat = torch.tensor(obs["board"].flatten(), dtype=torch.float32, device=device)
    current_tile = torch.tensor([obs.get("current_tile", 0)], dtype=torch.float32, device=device)
    features = torch.cat([board_flat, current_tile])
    return features


def get_action_mask(obs, device):
    """Single env action mask. Returns (N,) bool tensor; True = valid."""
    board = torch.tensor(obs["board"], device=device)
    valid = (board[:, 0] == 0)
    return valid
