# C++ Self-Play Engine — Implementation Plan

## Problem

Python's GIL + pure-Python loops make MCTS self-play ~10–50× slower than it should be. The bottlenecks in order:

1. **MCTS tree traversal** — Python dict lookups + loops per simulation
2. **Game state operations** — [make_move](file:///d:/RL-games-kgts/black_hole/game.py#79-91), [get_valid_moves](file:///d:/RL-games-kgts/black_hole/game.py#76-78), [clone](file:///d:/RL-games-kgts/black_hole/game.py#27-43) in pure Python
3. **Node allocation** — thousands of [MCTSNode](file:///d:/RL-games-kgts/black_hole/mcts.py#10-25) Python objects per game

DeepMind's solution: C++ actors, Python NN server, IPC between them. We will mirror this.

---

## Architecture (Single Machine)

```
┌─────────────────────────────────┐
│  Python Process (GPU)           │
│  ┌───────────────────────────┐  │
│  │  AlphaBH Neural Network   │  │
│  │  (PyTorch on CUDA)        │  │
│  └────────────┬──────────────┘  │
│               │ pybind11 call   │
│  ┌────────────▼──────────────┐  │
│  │  C++ MCTS Workers         │  │  ← This is what we build
│  │  (N threads, CPU)         │  │
│  │   • BlackHoleGame         │  │
│  │   • MCTSNode pool         │  │
│  │   • PUCT selection        │  │
│  │   • Leaf batching         │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

**Key design:** Single process, pybind11 callback from C++ into Python for NN eval. All game + tree logic runs in C++ without the GIL.

---

## Component 1: C++ Game Engine (`blackhole_cpp/game.h`)

```cpp
struct BlackHoleGame {
    // Fixed-size arrays: no heap alloc in hot path
    int8_t  board[45][2];   // [player_id, tile_value]
    uint8_t adj[45][6];     // adjacency: up to 6 neighbors per hex
    uint8_t adj_count[45];  // how many valid neighbors
    uint8_t current_player; // 1 or 2
    uint8_t tiles_placed;
    uint8_t num_hexes;
    uint8_t tiles_per_player;

    void   reset();
    void   make_move(int action, int tile_val);
    void   get_valid_moves(int* out, int& count) const; // no allocations
    bool   check_game_over() const;
    int    calculate_winner() const; // -1=P2, 0=draw, 1=P1
    void   clone_into(BlackHoleGame& dst) const;        // memcpy
};
```

**Why this is fast:**
- `board[45][2]` is 90 bytes — fits in a single cache line
- `clone_into` is a plain `memcpy(90 bytes)` vs Python's 5 dict + list copies
- No heap allocation per move

---

## Component 2: C++ MCTS (`blackhole_cpp/mcts.h`)

```cpp
struct MCTSNode {
    float    prior;
    float    total_value;
    int32_t  visits;
    uint8_t  action;
    uint8_t  n_children;
    bool     is_expanded;
    MCTSNode* children[46]; // max 45 moves + 1 null
    MCTSNode* parent;
    BlackHoleGame state;    // embedded, not pointer
};

class MCTSWorker {
public:
    // Callback type: Python provides this, C++ calls it for batch NN eval
    using NNEvalFn = std::function<void(
        const float* boards,    // (B, 2, 9, 9) flat
        int          batch_size,
        float*       out_policy, // (B, 45)
        float*       out_value   // (B, 1)
    )>;

    MCTSWorker(int n_games, int n_sims, NNEvalFn eval_fn);

    // Run one full batch of games, return (states, policies, outcomes)
    SelfPlayResult run_self_play(int temperature_cutoff = 10);

private:
    void     select(MCTSNode* root, std::vector<MCTSNode*>& path);
    void     expand_and_eval(std::vector<MCTSNode*>& leaves); // calls eval_fn in batch
    void     backpropagate(const std::vector<MCTSNode*>& path, float value);
    MCTSNode node_pool[MAX_NODES]; // arena allocator, zero GC pressure
    int      pool_idx;
};
```

**Node arena allocator**: Instead of `new MCTSNode()` per node (expensive), pre-allocate a flat pool and bump a pointer. Zero fragmentation, cache-friendly.

---

## Component 3: pybind11 Bridge (`blackhole_cpp/bindings.cpp`)

```python
# What this looks like from Python after compilation:
import blackhole_cpp

def nn_eval_callback(boards_flat, batch_size, out_policy, out_value):
    """Called by C++ MCTS for leaf evaluation."""
    x = torch.from_numpy(boards_flat).to(device)  # zero-copy
    with torch.no_grad():
        pi, v = agent(x)
    pi = torch.softmax(pi, dim=1).cpu().numpy()
    np.copyto(out_policy, pi)
    np.copyto(out_value, v.cpu().numpy())

worker = blackhole_cpp.MCTSWorker(n_games=40, n_sims=15, eval_fn=nn_eval_callback)
result = worker.run_self_play()
# result.states, result.policies, result.outcomes -> numpy arrays, drop into buffer
```

**The bridge releases the GIL** while C++ is doing tree search, and reacquires it only for NN forward passes. Tree search runs asynchronously on CPU while GPU is idle between batches.

---

## Component 4: Build System (`CMakeLists.txt`)

```cmake
cmake_minimum_required(VERSION 3.18)
project(blackhole_cpp)
find_package(pybind11 REQUIRED)
pybind11_add_module(blackhole_cpp
    game.cpp
    mcts.cpp
    bindings.cpp
)
target_compile_options(blackhole_cpp PRIVATE -O3 -march=native)
```

Build with:
```powershell
pip install pybind11
cmake -B build . && cmake --build build --config Release
```

---

## Integration with [train_alphazero.py](file:///d:/RL-games-kgts/train_alphazero.py)

Replace the entire [self_play()](file:///d:/RL-games-kgts/train_alphazero.py#59-136) function:
```python
import blackhole_cpp  # C++ extension

def self_play(agent, n_games, n_sims, buffer):
    def eval_fn(boards, batch_size, out_pi, out_v):
        x = torch.from_numpy(boards).to(device)
        with torch.no_grad():
            pi, v = agent(x)
        np.copyto(out_pi, torch.softmax(pi, dim=1).cpu().numpy())
        np.copyto(out_v, v.cpu().numpy())

    result = blackhole_cpp.run_self_play(n_games, n_sims, eval_fn)
    for state, policy, outcome in zip(result.states, result.policies, result.outcomes):
        buffer.push(state, policy, outcome)
```

---

## Estimated Speedup

| Operation | Python (ms) | C++ (ms) | Speedup |
|---|---|---|---|
| `game.clone()` | ~0.05 | ~0.001 | **50×** |
| [make_move()](file:///d:/RL-games-kgts/black_hole/game.py#79-91) | ~0.02 | ~0.0002 | **100×** |
| MCTS node alloc | ~0.01 | ~0.0001 | **100×** |
| PUCT loop (45 children) | ~0.05 | ~0.001 | **50×** |
| Full 15-sim MCTS | ~5ms | ~0.1ms | **~50×** |
| 40 games × 44 moves | ~8.8s | ~0.18s | **~50×** |

> [!IMPORTANT]
> The NN forward pass (GPU) stays in Python — it's already the optimized part. Only the CPU-bound tree search moves to C++.

---

## Build Order

1. `blackhole_cpp/game.h + game.cpp` — game logic, compile and unit-test independently
2. `blackhole_cpp/mcts.h + mcts.cpp` — MCTS with a stub eval callback (uniform random)
3. `blackhole_cpp/bindings.cpp` — pybind11 exposure
4. Integration into [train_alphazero.py](file:///d:/RL-games-kgts/train_alphazero.py)
5. Verify self-play outputs match Python reference
