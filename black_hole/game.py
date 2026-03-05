import numpy as np
from collections import deque

# Default game configuration (9 layers = 45 hexes, 22 tiles per player)
DEFAULT_LAYERS = 9

class BlackHoleGame:
    def __init__(self, layers=DEFAULT_LAYERS):
        self.layers = layers
        self.num_hexes = (layers * (layers + 1)) // 2  # 45 for L=9
        self.tiles_per_player = (self.num_hexes - 1) // 2  # 22 for L=9
        self.adj = self._build_adjacency()
        self.reset()

    def reset(self):
        # Board state: List of (player_id, value).
        # player_id: 0 (empty), 1 (Player 1), 2 (Player 2)
        # value: 1 to tiles_per_player. 0 if empty.
        self.board = np.zeros((self.num_hexes, 2), dtype=int)
        self.current_player = 1  # Player 1 starts
        self.tiles_placed = 0
        self.history = []

        self.p1_tiles = list(range(1, self.tiles_per_player + 1))
        self.p2_tiles = list(range(1, self.tiles_per_player + 1))

    def _build_adjacency(self):
        """
        Dynamically builds the adjacency list for an L-layer triangular grid.
        Node IDs are assigned row by row, left to right.
        Row r has (r+1) nodes, starting at index r*(r+1)//2.
        """
        N = self.num_hexes
        adj = {i: [] for i in range(N)}

        def node_id(r, c):
            return (r * (r + 1)) // 2 + c

        def add_edge(u, v):
            if 0 <= u < N and 0 <= v < N:
                if v not in adj[u]: adj[u].append(v)
                if u not in adj[v]: adj[v].append(u)

        for r in range(self.layers):
            for c in range(r + 1):
                curr = node_id(r, c)
                # Horizontal neighbor (right)
                if c + 1 <= r:
                    add_edge(curr, node_id(r, c + 1))
                # Down-left neighbor
                if r + 1 < self.layers:
                    add_edge(curr, node_id(r + 1, c))
                # Down-right neighbor
                if r + 1 < self.layers:
                    add_edge(curr, node_id(r + 1, c + 1))

        return adj

    def get_valid_moves(self):
        return [i for i, (pid, val) in enumerate(self.board) if pid == 0]

    def make_move(self, position, value):
        if self.board[position][0] != 0:
            raise ValueError(f"Position {position} is already occupied.")

        self.board[position] = [self.current_player, value]
        self.tiles_placed += 1
        self.history.append((self.current_player, position, value))

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1

        return self.check_game_over()

    def check_game_over(self):
        # Game ends when all but one tile is placed (the Black Hole)
        return self.tiles_placed >= self.num_hexes - 1

    def calculate_score(self):
        # Find black hole (the empty spot)
        black_hole_idx = -1
        for i in range(self.num_hexes):
            if self.board[i][0] == 0:
                black_hole_idx = i
                break

        if black_hole_idx == -1:
            raise RuntimeError("No black hole found, but game over?")

        # Calculate rings using BFS
        distances = {black_hole_idx: 0}
        queue = deque([black_hole_idx])

        rings = {}  # distance -> list of (player, value)
        visited = {black_hole_idx}

        while queue:
            curr = queue.popleft()
            dist = distances[curr]

            if dist > 0:  # Don't count the hole itself
                if dist not in rings: rings[dist] = []
                rings[dist].append(self.board[curr])

            for neighbor in self.adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = dist + 1
                    queue.append(neighbor)

        # Compare rings
        max_dist = max(rings.keys()) if rings else 0

        p1_total = 0
        p2_total = 0

        for d in range(1, max_dist + 1):
            if d not in rings: continue

            r_p1 = sum(tile[1] for tile in rings[d] if tile[0] == 1)
            r_p2 = sum(tile[1] for tile in rings[d] if tile[0] == 2)

            p1_total += r_p1
            p2_total += r_p2

            if r_p1 < r_p2:
                return 1, f"Player 1 wins at Ring {d} ({r_p1} vs {r_p2})"
            elif r_p2 < r_p1:
                return 2, f"Player 2 wins at Ring {d} ({r_p2} vs {r_p1})"

        # If all rings tied, total score check
        if p1_total < p2_total:
            return 1, f"Player 1 wins by Total ({p1_total} vs {p2_total})"
        elif p2_total < p1_total:
            return 2, f"Player 2 wins by Total ({p2_total} vs {p1_total})"

        return 0, "Draw"

    def render_ascii(self):
        chars = []
        for i in range(self.num_hexes):
            p, v = self.board[i]
            if p == 0: s = "{ .}"
            elif p == 1: s = f"{{A{v:02d}}}"
            else: s = f"{{B{v:02d}}}"
            chars.append(s)

        idx = 0
        for r in range(self.layers):
            indent = " " * (self.layers - r)
            row_str = " ".join(chars[idx:idx + r + 1])
            print(indent + row_str)
            idx += r + 1
