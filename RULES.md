# Black Hole Game Rules

The **Black Hole** is a strategic tile-placement game played on a triangular grid. 

## 1. Components
*   **The Board**: A triangle consisting of 21 spaces (arranged in rows of 1, 2, 3, 4, 5, 6).
*   **The Tiles**: Two sets of tiles numbered **1 to 10**.
    *   **Player 1 (Red)**: Has tiles 1-10.
    *   **Player 2 (Green)**: Has tiles 1-10.

## 2. Objective
The goal is to **minimize** the value of your tiles that end up "sucked in" by the Black Hole. 

Specifically, you want your **Lowest Value** tiles to be closest to the empty space left at the end of the game, and your **Highest Value** tiles to be as far away as possible.

## 3. Gameplay
1.  **Starting**: The board is empty.
2.  **Turns**: Players alternate turns placing one of their available tiles on any empty space.
3.  **The Black Hole**: The game ends when there is exactly **one space left empty**. This empty space becomes the **Black Hole**.
    *   Since there are 21 spaces and 20 total tiles, this happens exactly after both players have played all 10 of their tiles.

## 4. Scoring (The Event Horizon)
Once the Black Hole is established, the board is evaluated in **Rings** (layers of distance from the Black Hole).

**Ring 1**: All tiles directly adjacent to the Black Hole.
**Ring 2**: All tiles adjacent to Ring 1 (distance 2 from Black Hole).
**Ring 3+**: Etc.

The winner is determined by comparing the sum of tile values in each Ring, starting from **Ring 1**:

1.  **Check Ring 1**: Calculate the sum of Player 1's tiles vs. Player 2's tiles in this ring.
    *   If **Player 1 < Player 2**: Player 1 WINS immediately.
    *   If **Player 2 < Player 1**: Player 2 WINS immediately.
2.  **Tie-Breaker**: If the sums in Ring 1 are equal, move to **Ring 2** and repeat the comparison.
3.  **Continue**: Continue outwards until a winner is found.
4.  **Draw**: If all rings are tied, the game is a draw.

## Strategy Tip
*   **Save your 1s and 2s**: Try to place these *last* or near where you think the Black Hole will form.
*   **Dump your 9s and 10s**: Place these early on the edges or corners, far away from the chaotic center.
*   **Control the Hole**: Since the last empty space becomes the Black Hole, the player who moves last (Player 2) has slightly more control over its final location, but Player 1 dictates the board shape leading up to it.
