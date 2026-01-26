import numpy as np
from collections import deque

class BlackHoleGame:
    def __init__(self):
        self.reset()
        self.adj = self._build_adjacency()

    def reset(self):
        # 21 spaces.
        # Board state: List of (player_id, value). 
        # player_id: 0 (empty), 1 (Player 1), 2 (Player 2)
        # value: 1-10. 0 if empty.
        self.board = np.zeros((21, 2), dtype=int) 
        self.current_player = 1 # Player 1 starts
        self.tiles_placed = 0
        self.history = []
        
        # Tile queues for each player: [1, 2, ..., 10]
        # We don't strictly need to track this if we enforce correct moves from env,
        # but good for validation.
        self.p1_tiles = list(range(1, 11))
        self.p2_tiles = list(range(1, 11))

    def _build_adjacency(self):
        # 21-node triangle structure
        #      0
        #     1 2
        #    3 4 5
        #   6 7 8 9
        # 10 11 12 13 14
        #15 16 17 18 19 20
        
        adj = {i: [] for i in range(21)}
        
        def add_edge(u, v):
            if u < 21 and v < 21:
                if v not in adj[u]: adj[u].append(v)
                if u not in adj[v]: adj[v].append(u)

        # Connect rows
        # Row 0 to 1
        add_edge(0, 1); add_edge(0, 2)
        
        # Row 1 to 2
        add_edge(1, 3); add_edge(1, 4)
        add_edge(2, 4); add_edge(2, 5)
        add_edge(1, 2) # horizontal

        # Row 2 to 3
        add_edge(3, 6); add_edge(3, 7)
        add_edge(4, 7); add_edge(4, 8)
        add_edge(5, 8); add_edge(5, 9)
        add_edge(3, 4); add_edge(4, 5) # horizontal

        # Row 3 to 4
        add_edge(6, 10); add_edge(6, 11)
        add_edge(7, 11); add_edge(7, 12)
        add_edge(8, 12); add_edge(8, 13)
        add_edge(9, 13); add_edge(9, 14)
        add_edge(6, 7); add_edge(7, 8); add_edge(8, 9) # horizontal

        # Row 4 to 5
        add_edge(10, 15); add_edge(10, 16)
        add_edge(11, 16); add_edge(11, 17)
        add_edge(12, 17); add_edge(12, 18)
        add_edge(13, 18); add_edge(13, 19)
        add_edge(14, 19); add_edge(14, 20)
        add_edge(10, 11); add_edge(11, 12); add_edge(12, 13); add_edge(13, 14) # horizontal
        
        # Horizontal for last row
        add_edge(15, 16); add_edge(16, 17); add_edge(17, 18); add_edge(18, 19); add_edge(19, 20)

        return adj

    def get_valid_moves(self):
        # Return indices of empty spaces
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
        return self.tiles_placed >= 20

    def calculate_score(self):
        # Find black hole (the empty spot)
        black_hole_idx = -1
        for i in range(21):
            if self.board[i][0] == 0:
                black_hole_idx = i
                break
        
        if black_hole_idx == -1:
            raise RuntimeError("No black hole found, but game over?")

        # Calculate rings using BFS
        distances = {black_hole_idx: 0}
        queue = deque([black_hole_idx])
        
        rings = {} # distance -> list of (player, value)

        visited = {black_hole_idx}
        
        while queue:
            curr = queue.popleft()
            dist = distances[curr]
            
            if dist > 0: # Don't count the hole itself
                if dist not in rings: rings[dist] = []
                # Add tile info
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
        
        # If all rings tied, total score check (though strictly sums of rings cover this)
        if p1_total < p2_total:
             return 1, f"Player 1 wins by Total ({p1_total} vs {p2_total})"
        elif p2_total < p1_total:
             return 2, f"Player 2 wins by Total ({p2_total} vs {p1_total})"

        return 0, "Draw"
    
    def render_ascii(self):
        # visual helper
        chars = []
        for i in range(21):
            p, v = self.board[i]
            if p == 0: s = "{ .}"
            elif p == 1: s = f"{{A{v}}}"
            else: s = f"{{B{v}}}"
            chars.append(s)
            
        print(f"      {chars[0]}")
        print(f"     {chars[1]} {chars[2]}")
        print(f"    {chars[3]} {chars[4]} {chars[5]}")
        print(f"   {chars[6]} {chars[7]} {chars[8]} {chars[9]}")
        print(f"  {chars[10]} {chars[11]} {chars[12]} {chars[13]} {chars[14]}")
        print(f" {chars[15]} {chars[16]} {chars[17]} {chars[18]} {chars[19]} {chars[20]}")
