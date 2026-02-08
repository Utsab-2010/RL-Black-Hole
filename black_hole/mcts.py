
import math
import random
import copy
from game import BlackHoleGame

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # BlackHoleGame instance
        self.parent = parent
        self.action = action  # Action taken to reach this state
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = state.get_valid_moves()
        self.player_just_moved = 2 if state.current_player == 1 else 1 # Parent's move led here

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.414):
        choices_weights = [
            (child.wins / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

class MCTS:
    def __init__(self, root_state, iterations=1000):
        # We must CLONE the root state so search doesn't mutate actual game
        self.root = MCTSNode(copy.deepcopy(root_state))
        self.iterations = iterations

    def search(self):
        for _ in range(self.iterations):
            node = self.select(self.root)
            reward = self.simulate(node.state)
            self.backpropagate(node, reward)
        
        # Return best move (robust child)
        if not self.root.children:
            return None
        return sorted(self.root.children, key=lambda c: c.visits)[-1].action

    def select(self, node):
        while not node.state.check_game_over():
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = node.best_child()
        return node

    def expand(self, node):
        action = node.untried_actions.pop()
        next_state = copy.deepcopy(node.state)
        
        # Determine value (1-10) for the move
        # We need logic to determine WHAT value is placed.
        # Gym logic: tile = (tiles_placed // 2) + 1
        # game logic: does make_move handle value?
        # make_move(position, value)
        
        tile_val = (next_state.tiles_placed // 2) + 1
        if tile_val > 10: tile_val = 10
        
        next_state.make_move(action, tile_val)
        child_node = MCTSNode(next_state, parent=node, action=action)
        node.children.append(child_node)
        return child_node

    def simulate(self, state):
        current_state = copy.deepcopy(state)
        while not current_state.check_game_over():
            valid_moves = current_state.get_valid_moves()
            if not valid_moves: break # Should not happen if not game over
            
            action = random.choice(valid_moves)
            
            tile_val = (current_state.tiles_placed // 2) + 1
            if tile_val > 10: tile_val = 10
            
            current_state.make_move(action, tile_val)
            
        # Evaluation
        winner, _ = current_state.calculate_score()
        # Return reward for player who JUST moved at root?
        # Standard: Reward is relative to the player who made the move.
        # But MCTS usually backs up reward for Root Player.
        
        # Let's say we optimize for Player 1 (root player).
        # Reward = 1 if P1 wins, 0 if P2 wins (or -1).
        
        # But node store "wins". Usually relative to parent logic.
        # Convention: "wins" stores value for the player who JUST MOVED.
        # So backprop logic:
        # If result is P1 win, and node.player_just_moved == 1 -> wins++
        
        return winner

    def backpropagate(self, node, winner):
        while node is not None:
            node.visits += 1
            if node.player_just_moved == winner:
                node.wins += 1
            elif winner == 0: # Draw
                node.wins += 0.5 
            node = node.parent

if __name__ == "__main__":
    game = BlackHoleGame()
    
    # Simulate a few random moves first to make it interesting
    for _ in range(5):
        valid = game.get_valid_moves()
        if not valid: break
        move = random.choice(valid)
        
        val = (game.tiles_placed // 2) + 1
        if val > 10: val = 10
             
        game.make_move(move, val)
        
    game.render_ascii()
    print(f"Current Player: {game.current_player}")
    
    iters = 1000
    print(f"Running MCTS for {iters} iterations...")
    mcts = MCTS(game, iterations=iters)
    best_move = mcts.search()
    
    print(f"MCTS suggests move: {best_move}")
    
    # Verify move validity
    if best_move in game.get_valid_moves():
        print("Move is valid.")
    else:
        print("ERROR: MCTS suggested invalid move!")
