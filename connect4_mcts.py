"""
Assignment II - Starter Code: Connect 4 in Pygame with MCTS based hint for 2 human players.

This file is meant to be the starter code, it is not a requirement, feel free to ignore it.
In your tasks, you need to implement:
    -  1 human player against an AI agent
    -  2 AI agents playing against each other.

Run this sample codeussing: 
    python connect4_mcts.py

Rememever that you need Pygame installed:
    pip install pygame
"""

import pygame
import sys
import math
import random

# ==========================================
#              GAME CONSTANTS AND COLORS
# ==========================================

ROWS = 6
COLS = 7

EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2

SQUARESIZE = 100
RADIUS = SQUARESIZE // 2 - 5

BOARD_COLOR = (0, 0, 200)
BG_COLOR = (0, 0, 0)
PLAYER1_COLOR = (200, 0, 0)
PLAYER2_COLOR = (230, 230, 0)
TEXT_COLOR = (255, 255, 255)
HINT_COLOR = (0, 200, 0)

WIDTH = COLS * SQUARESIZE
HEIGHT = (ROWS + 2) * SQUARESIZE
SIZE = (WIDTH, HEIGHT)

FPS = 60


# ====================================
#              CONNECT 4 STATE CLASS
# ====================================

class Connect4State:
    def __init__(self, board=None, current_player=PLAYER1):
        if board is None:
            self.board = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]
        else:
            self.board = [row[:] for row in board]
        self.current_player = current_player

    def clone(self):
        return Connect4State(self.board, self.current_player)

    def get_legal_moves(self):
        moves = []
        for c in range(COLS):
            if self.board[0][c] == EMPTY:
                moves.append(c)
        return moves

    def make_move(self, col):
        for r in range(ROWS - 1, -1, -1):
            if self.board[r][col] == EMPTY:
                self.board[r][col] = self.current_player
                self.current_player = PLAYER1 if self.current_player == PLAYER2 else PLAYER2
                return True
        return False

    def check_winner(self):
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                piece = self.board[r][c]
                if piece != EMPTY:
                    if all(self.board[r][c + i] == piece for i in range(4)):
                        return piece

        # Vertical
        for c in range(COLS):
            for r in range(ROWS - 3):
                piece = self.board[r][c]
                if piece != EMPTY:
                    if all(self.board[r + i][c] == piece for i in range(4)):
                        return piece

        # Diagonal (top-left to bottom-right)
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                piece = self.board[r][c]
                if piece != EMPTY:
                    if all(self.board[r + i][c + i] == piece for i in range(4)):
                        return piece

        # Diagonal (bottom-left to top-right)
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                piece = self.board[r][c]
                if piece != EMPTY:
                    if all(self.board[r - i][c + i] == piece for i in range(4)):
                        return piece

        return None

    def is_full(self):
        return all(self.board[0][c] != EMPTY for c in range(COLS))

    def is_terminal(self):
        return self.check_winner() is not None or self.is_full()


# ============================================================
#              IMPROVED MCTS IMPLEMENTATION
# ============================================================

"""
MONTE CARLO TREE SEARCH (MCTS) - DETAILED EXPLANATION

MCTS is a smart search algorithm that learns which moves are good by playing
many random games. It's especially powerful for games like Connect 4 where
it's too expensive to look at every possible future move.

The algorithm has 4 main phases that repeat many times:

1. SELECTION
   - Start at the root (current game position)
   - Navigate down the tree by picking the most promising child at each level
   - We use a formula called UCB1 (Upper Confidence Bound) that balances:
     * EXPLOITATION: Choose moves that have won a lot in the past
     * EXPLORATION: Try moves we haven't tested much yet
   - Keep going down until we reach a node that isn't fully explored yet

2. EXPANSION
   - Once we find a node with untried moves, pick one of them
   - Create a new child node representing the game state after that move
   - Add this child to the tree

3. SIMULATION (also called ROLLOUT or PLAYOUT)
   - From the new child node, play the game randomly until it ends
   - Just pick random legal moves for both players until someone wins or it's a draw
   - This gives us a result: win, loss, or draw

4. BACKPROPAGATION
   - Take the result from the simulation
   - Walk back up the tree from the child to the root
   - Update statistics at every node along the path:
     * Increment visit count (how many times we've been here)
     * Add the reward (1.0 for win, 0.0 for loss, 0.5 for draw)
   - This teaches the tree which moves lead to good outcomes

After running thousands of iterations:
   - Each possible move from the root has statistics
   - We pick the move that was visited the most (not the highest win rate!)
   - This is called "robust child selection" and gives us the best move

Key insight: Random playouts + statistics = surprisingly good moves!
"""

class MCTSNode:
    """
    Node in the MCTS tree.
    
    Each node represents a game state and stores:
    - state: The Connect4State at this point in the game
    - parent: The node that led to this one (None for root)
    - move: The column number that was played to get here from parent
    - children: List of child nodes (one for each move we've tried)
    - visits: How many times we've visited this node during search
    - wins: Total reward accumulated from this node (from current player's perspective)
    - untried_moves: Moves we haven't created children for yet
    
    The key improvement over basic MCTS: we track untried_moves directly,
    making expansion faster and cleaner.
    """
    
    def __init__(self, state, parent=None, move=None):
        """
        Create a new MCTS node.
        
        Arguments:
            state: Connect4State representing the game position at this node
            parent: Parent MCTSNode (None if this is the root)
            move: The column index (0-6) that led from parent to this node
        """
        self.state = state              # Game state at this node
        self.parent = parent            # Parent node in the tree
        self.move = move                # Move that created this node
        self.children = []              # List of child nodes we've expanded
        self.visits = 0                 # Number of times this node was visited
        self.wins = 0.0                 # Sum of rewards from simulations through this node
        
        # Track which moves we haven't tried yet
        # If the game is over, there are no moves to try
        self.untried_moves = state.get_legal_moves() if not state.is_terminal() else []
    
    def is_fully_expanded(self):
        """
        Check if this node has tried all possible moves.
        
        A node is fully expanded when:
        - We've created a child for every legal move from this position, OR
        - The game is over at this node (no moves are possible)
        
        We know this by checking if untried_moves is empty.
        
        Returns:
            True if all moves have been tried, False otherwise
        """
        # If there are no untried moves left, we're fully expanded
        return len(self.untried_moves) == 0
    
    def is_terminal(self):
        """
        Check if this node represents a finished game.
        
        The game is terminal if someone won or the board is full.
        
        Returns:
            True if the game is over, False otherwise
        """
        return self.state.is_terminal()
    
    def select_child_uct(self, exploration_constant=1.414):
        """
        Select the most promising child using the UCB1 formula.
        
        This is the heart of MCTS selection. The UCB1 formula is:
        
            UCB1 = (wins/visits) + c * sqrt(ln(parent_visits) / visits)
                   \___ exploit__/   \_______ explore ___________/
        
        - EXPLOITATION (wins/visits): Prefer children with high win rates
        - EXPLORATION (sqrt term): Prefer children we haven't visited much
        - c (exploration_constant): Controls the balance (typically √2 ≈ 1.414)
        
        The sqrt(ln(parent)/child) term ensures:
        - As parent visits increase, we explore more
        - As child visits increase, exploration bonus decreases
        - This balances trying new things vs exploiting good moves
        
        Special case: If a child has never been visited (visits=0),
        we return it immediately to ensure every child gets tried at least once.
        
        Arguments:
            exploration_constant: Weight for exploration term (default √2)
            
        Returns:
            The child node with the highest UCB1 score
        """
        best_score = float('-inf')  # Start with worst possible score
        best_children = []          # Track all children with the best score
        
        # Evaluate each child using UCB1
        for child in self.children:
            # If child was never visited, give it infinite priority
            # This ensures every move gets tried at least once
            if child.visits == 0:
                return child
            
            # Calculate exploitation: what's the win rate?
            # This is from the parent's perspective, so we just use wins/visits
            exploitation = child.wins / child.visits
            
            # Calculate exploration: how much should we explore this child?
            # Uses the Upper Confidence Bound formula
            exploration = exploration_constant * math.sqrt(
                math.log(self.visits) / child.visits
            )
            
            # Total UCB1 score
            ucb_score = exploitation + exploration
            
            # Keep track of the best scoring children
            if ucb_score > best_score:
                best_score = ucb_score
                best_children = [child]
            elif ucb_score == best_score:
                # If there's a tie, keep all tied children
                best_children.append(child)
        
        # If multiple children tie for best score, pick one randomly
        # This adds diversity to our search
        return random.choice(best_children)
    
    def expand(self):
        """
        Expand the node by creating a child for one untried move.
        
        This is the EXPANSION phase of MCTS. When we reach a node during
        selection that still has untried moves, we:
        1. Pick one untried move randomly
        2. Remove it from untried_moves (so we don't try it again)
        3. Create a new game state by applying this move
        4. Create a new child node with this state
        5. Add the child to our children list
        
        Returns:
            The newly created child node, or None if no moves to expand
        """
        # If there are no untried moves, we can't expand
        if not self.untried_moves:
            return None
        
        # Pick a random untried move
        # We could be smarter here (e.g., prioritize center columns),
        # but random selection works well and keeps the algorithm general
        move = random.choice(self.untried_moves)
        
        # Remove this move from untried list so we don't try it again
        self.untried_moves.remove(move)
        
        # Create a new game state by cloning current state and applying the move
        new_state = self.state.clone()
        new_state.make_move(move)
        
        # Create a new node for this child state
        child_node = MCTSNode(new_state, parent=self, move=move)
        
        # Add to our list of children
        self.children.append(child_node)
        
        # Return the new child so we can simulate from it
        return child_node
    
    def best_child(self, exploration_weight=0):
        """
        Select the best child for final move selection.
        
        After MCTS finishes all iterations, we need to pick which move to actually play.
        There are different strategies:
        
        1. ROBUST CHILD (exploration_weight=0): Pick the most visited child
           - Most visits = most confidence in this move
           - This is the recommended approach and works best in practice
           - Less affected by random variance in win rates
        
        2. MAX CHILD: Pick the child with highest win rate
           - Can be misled by lucky wins in small sample sizes
           - Not recommended for final selection
        
        3. ROBUST-MAX: Mix of visits and win rate
           - Use exploration_weight > 0 to still apply UCT
           - Rarely used for final selection
        
        Arguments:
            exploration_weight: If 0, use pure visit count (robust child)
                               If >0, use UCT with this exploration weight
        
        Returns:
            The best child node, or None if no children exist
        """
        # If we have no children, we can't pick one
        if not self.children:
            return None
        
        if exploration_weight == 0:
            # ROBUST CHILD selection: return the most visited child
            # This is the standard final selection method
            return max(self.children, key=lambda c: c.visits)
        else:
            # Use UCT with custom exploration weight
            # This is rarely used but available for experimentation
            return self.select_child_uct(exploration_weight)
    
    def update(self, result):
        """
        Update this node's statistics after a simulation.
        
        This is called during BACKPROPAGATION. After a simulation finishes,
        we walk back up the tree and update every node on the path.
        
        The result is always from the root player's perspective:
        - 1.0 means root player won
        - 0.0 means root player lost
        - 0.5 means draw
        
        But each node stores wins from ITS current player's perspective.
        So before updating, the calling code flips the result if needed.
        
        Arguments:
            result: The reward value to add (already adjusted for perspective)
        """
        self.visits += 1      # Increment visit counter
        self.wins += result   # Add the reward to our total wins


def evaluate_terminal_state(state, player):
    """
    Evaluate a finished game from a player's perspective.
    
    When a simulation ends, we need to know who won. This function
    checks the winner and converts it to a numeric reward:
    
    - If the given player won: return 1.0 (full reward)
    - If the opponent won: return 0.0 (no reward)
    - If it's a draw: return 0.5 (half reward)
    
    These values are somewhat arbitrary but work well:
    - 1.0 for win is standard
    - 0.0 for loss makes the math clean
    - 0.5 for draw is fair (halfway between win and loss)
    
    Arguments:
        state: A terminal Connect4State (game is over)
        player: The player from whose perspective we evaluate (PLAYER1 or PLAYER2)
        
    Returns:
        1.0 for win, 0.0 for loss, 0.5 for draw
    """
    # Check who won the game
    winner = state.check_winner()
    
    # If there's no winner, it's a draw
    if winner is None:
        return 0.5  # Draw = half reward
    # If the specified player won
    elif winner == player:
        return 1.0  # Win = full reward
    # If the opponent won
    else:
        return 0.0  # Loss = no reward


def simulate_random_playout(state, player):
    """
    Perform a random simulation from the given state until the game ends.
    
    This is the SIMULATION (or ROLLOUT/PLAYOUT) phase of MCTS.
    
    The idea is simple:
    1. Clone the state so we don't modify the actual game
    2. Keep making random moves for both players
    3. Continue until someone wins or the board is full
    4. Return the result from the given player's perspective
    
    Why random moves?
    - It's fast! We can do thousands of simulations quickly
    - It's unbiased - doesn't favor any particular strategy
    - Surprisingly, random playouts + statistics = good moves!
    - For stronger play, you could use "heavy playouts" with heuristics,
      but that's slower and often not worth the cost
    
    Arguments:
        state: Connect4State to simulate from (will be cloned, not modified)
        player: The player from whose perspective we want the result
        
    Returns:
        Float value: 1.0 (player wins), 0.0 (player loses), or 0.5 (draw)
    """
    # Clone the state so we don't affect the actual game tree
    simulation_state = state.clone()
    
    # Play randomly until the game is over
    while not simulation_state.is_terminal():
        # Get all legal moves from current position
        legal_moves = simulation_state.get_legal_moves()
        
        # If no legal moves (shouldn't happen if terminal check works), break
        if not legal_moves:
            break
        
        # Pick a random legal move
        move = random.choice(legal_moves)
        
        # Apply this move to the simulation state
        # This also switches to the other player
        simulation_state.make_move(move)
    
    # Game is over, evaluate the result from the player's perspective
    return evaluate_terminal_state(simulation_state, player)


def mcts_search(root_state, iterations=1000, exploration_constant=1.414):
    """
    Execute Monte Carlo Tree Search to find the best move.
    
    This is the main MCTS algorithm that ties together all four phases:
    SELECTION → EXPANSION → SIMULATION → BACKPROPAGATION
    
    HIGH-LEVEL OVERVIEW:
    We build a search tree by running many iterations. Each iteration:
    1. Starts at the root (current game state)
    2. Walks down the tree using UCB1 to find promising paths
    3. Adds a new node to the tree (expansion)
    4. Simulates a random game from that node
    5. Updates all nodes on the path with the result
    
    After many iterations, the tree has good statistics about which
    moves lead to wins. We pick the most-visited move as our choice.
    
    WHY IT WORKS:
    - Good moves get visited more because UCB1 favors them
    - But we also explore uncertain moves to avoid missing good ones
    - Random playouts are fast enough to do thousands of iterations
    - Law of large numbers: statistics converge to true values
    
    PARAMETERS EXPLAINED:
    - iterations: More = smarter but slower (1000-5000 typical)
    - exploration_constant: Usually √2 ≈ 1.414
      * Higher = explore more (try risky moves)
      * Lower = exploit more (stick to known good moves)
    
    Arguments:
        root_state: Current game state to search from
        iterations: Number of MCTS iterations to perform (default 1000)
        exploration_constant: UCB1 exploration parameter (default √2)
        
    Returns:
        Best move as column index (0-6), or None if no legal moves
    """
    # If the game is already over, there's no move to make
    if root_state.is_terminal():
        return None
    
    # Remember which player is making this decision
    # We'll need this to correctly evaluate simulation results
    root_player = root_state.current_player
    
    # Create the root node of our search tree
    # We clone the state so our search doesn't affect the actual game
    root_node = MCTSNode(root_state.clone())
    
    # Run MCTS for the specified number of iterations
    for _ in range(iterations):
        # Start at the root and pick a path down the tree using UCB1
        node = root_node          # Current node we're examining
        state = root_state.clone()  # Clone state to track where we are
        # === PHASE 1: SELECTION ===
        
        # Keep going down the tree as long as:
        # - The node has been fully expanded (all moves tried)
        # - The game isn't over yet at this node
        while node.is_fully_expanded() and not node.is_terminal():
            # Use UCB1 to select the most promising child
            node = node.select_child_uct(exploration_constant)
            
            # Apply the move that leads to this child
            # This keeps our state in sync with the node
            state.make_move(node.move)
        
        # === PHASE 2: EXPANSION ===
        # If the selected node isn't terminal and has untried moves,
        # create a new child for one of those moves
        
        if not node.is_terminal() and not node.is_fully_expanded():
            # Expand by creating a new child node
            node = node.expand()
            
            # Update our state to match the new child
            # node.state is already the state after the move,
            # so we just clone it
            state = node.state.clone()
        
        # === PHASE 3: SIMULATION ===
        # From this node, play out the game randomly until it ends
        
        # Run a random playout from the current state
        # This returns 1.0 (root player wins), 0.0 (loses), or 0.5 (draw)
        result = simulate_random_playout(state, root_player)
        
        # === PHASE 4: BACKPROPAGATION ===
        # Walk back up the tree and update all nodes with the result
        
        # Start from the node we just simulated from
        # and work our way back to the root
        while node is not None:
            # The result is from root_player's perspective
            # But each node stores wins from ITS player's perspective
            
            # If this node's player is the root player,
            # add the result directly
            # player who made the move into this node
            player_who_moved = PLAYER1 if node.state.current_player == PLAYER2 else PLAYER2

            if player_who_moved == root_player:
                node.update(result)
            else:
                node.update(1.0 - result)

            # Move up to the parent
            node = node.parent
    
    # === FINAL MOVE SELECTION ===
    # After all iterations, pick the best child using robust selection
    # (most visited child, not highest win rate)
    
    best_child = root_node.best_child(exploration_weight=0)
    
    # If we somehow have no children (shouldn't happen), return None
    if best_child is None:
        return None

    print("\n=== MCTS STATS ===")
    print(f"Total root visits: {root_node.visits}")

    print("\nChildren:")
    for child in root_node.children:
        win_rate = child.wins / child.visits if child.visits > 0 else 0
        print(f"  Move {child.move}: visits={child.visits}, wins={child.wins}, win_rate={win_rate:.3f}")

    print("\nUCB scores:")
    for child in root_node.children:
        if child.visits > 0:
            exploitation = child.wins / child.visits
            exploration = exploration_constant * math.sqrt(math.log(root_node.visits) / child.visits)
            ucb = exploitation + exploration
        else:
            ucb = float('inf')
        print(f"  Move {child.move}: UCB={ucb}")


    print("==================\n")
    
    # Return the move (column index) that leads to the best child
    return best_child.move


# ============================================================
#             DRAWING THE GAME WITH PYGAME
# ============================================================

def draw_board(screen, state, font, message=""):
    """
    Draw the entire game screen with Pygame.
    
    This function renders:
    1. Black background
    2. Message text at the top (player turn, winner, etc.)
    3. Blue Connect 4 board with circular holes
    4. Red and yellow pieces for both players
    
    The board is drawn 2 rows down from the top to leave space for messages.
    
    Arguments:
        screen: Pygame surface to draw on
        state: Connect4State representing current game position
        font: Pygame font object for rendering text
        message: String to display at the top (default empty string)
    """
    # Fill entire screen with black background
    screen.fill(BG_COLOR)
    
    # Draw the message text at the top
    text_surface = font.render(message, True, TEXT_COLOR)
    screen.blit(text_surface, (10, 5))  # Position at (10, 5) from top-left
    
    # Draw the blue board with holes
    # We iterate through each column and row to draw squares and circles
    for c in range(COLS):
        for r in range(ROWS):
            # Draw blue square for this cell
            # (r + 2) because we leave 2 rows at top for messages
            pygame.draw.rect(
                screen,
                BOARD_COLOR,
                (c * SQUARESIZE, (r + 2) * SQUARESIZE, SQUARESIZE, SQUARESIZE)
            )
            
            # Draw black circle "hole" in the middle of the square
            # This creates the classic Connect 4 appearance
            pygame.draw.circle(
                screen,
                BG_COLOR,
                (c * SQUARESIZE + SQUARESIZE // 2, (r + 2) * SQUARESIZE + SQUARESIZE // 2),
                RADIUS
            )
    
    # Draw the pieces (red and yellow discs)
    # Iterate through the board and draw a colored circle wherever there's a piece
    for c in range(COLS):
        for r in range(ROWS):
            piece = state.board[r][c]
            
            # Skip empty cells
            if piece != EMPTY:
                # Choose color based on which player's piece
                color = PLAYER1_COLOR if piece == PLAYER1 else PLAYER2_COLOR
                
                # Draw the colored disc
                pygame.draw.circle(
                    screen,
                    color,
                    (c * SQUARESIZE + SQUARESIZE // 2, (r + 2) * SQUARESIZE + SQUARESIZE // 2),
                    RADIUS
                )
    
    # Update the display to show all our drawings
    pygame.display.update()


# ============================================================
#              MAIN GAME LOOP
# ============================================================

def main():
    """
    Main function that runs the Connect 4 game with Pygame.
    
    GAME FLOW:
    1. Initialize Pygame and create the game window
    2. Show mode selection screen (Player vs AI or AI vs AI)
    3. Run the main game loop:
       - Handle player input (mouse clicks)
       - Run AI moves using MCTS
       - Check for win/loss/draw
       - Draw the board
    4. Allow restart with 'R' key
    
    GAME MODES:
    - Mode 1 (Player vs AI): Human plays as red (Player 1), AI as yellow (Player 2)
    - Mode 2 (AI vs AI): Both players are AI, useful for testing or entertainment
    
    CONTROLS:
    - Mouse click: Drop piece in column (Player vs AI mode only)
    - R key: Restart game at any time
    - Close window: Quit the game
    """
    # Initialize Pygame library
    pygame.init()
    
    # Create the game window with specified size
    screen = pygame.display.set_mode(SIZE)
    pygame.display.set_caption("MCTS Connect 4")
    
    # Create clock to control frame rate
    clock = pygame.time.Clock()
    
    # Create font for text rendering
    font = pygame.font.SysFont("arial", 24)
    
    # === MODE SELECTION SCREEN ===
    # Let the player choose between Player vs AI or AI vs AI
    
    mode = None         # Will store 1 (PvAI) or 2 (AI vs AI)
    selecting = True    # Controls the selection loop
    
    while selecting:
        # Clear screen to black
        screen.fill(BG_COLOR)
        
        # Display mode selection options
        t1 = font.render("Press 1 for Player vs AI", True, TEXT_COLOR)
        t2 = font.render("Press 2 for AI vs AI", True, TEXT_COLOR)
        screen.blit(t1, (50, 100))
        screen.blit(t2, (50, 150))
        pygame.display.update()
        
        # Handle events during mode selection
        for event in pygame.event.get():
            # Allow closing window
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # Check for key presses
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    mode = 1            # Player vs AI
                    selecting = False   # Exit selection loop
                elif event.key == pygame.K_2:
                    mode = 2            # AI vs AI
                    selecting = False   # Exit selection loop
    
    # === GAME INITIALIZATION ===
    # Start with a fresh game state
    state = Connect4State()
    game_over = False
    message = "Red turn"
    
    # === MAIN GAME LOOP ===
    running = True
    while running:
        # Limit frame rate to FPS (60 frames per second)
        clock.tick(FPS)
        
        # === EVENT HANDLING ===
        # Process all Pygame events (mouse clicks, key presses, etc.)
        
        for event in pygame.event.get():
            # Handle window close button
            if event.type == pygame.QUIT:
                running = False
            
            # Handle key presses
            if event.type == pygame.KEYDOWN:
                # R key restarts the game
                if event.key == pygame.K_r:
                    state = Connect4State()  # Fresh board
                    game_over = False
                    message = "Red turn"
            
            # === PLAYER INPUT (Mode 1 only) ===
            # In Player vs AI mode, handle mouse clicks for human player
            if mode == 1 and not game_over and state.current_player == PLAYER1:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Get mouse position
                    x, _ = event.pos
                    
                    # Calculate which column was clicked
                    col = x // SQUARESIZE
                    
                    # If this column is legal, make the move
                    if col in state.get_legal_moves():
                        state.make_move(col)
                        
                        # Check if this move ended the game
                        winner = state.check_winner()
                        if winner:
                            # Game over - someone won
                            message = "Red wins! Press R to restart" if winner == PLAYER1 else "Yellow wins! Press R to restart"
                            game_over = True
                        elif state.is_full():
                            # Game over - draw
                            message = "Draw! Press R to restart"
                            game_over = True
                        else:
                            # Game continues - AI's turn
                            message = "AI thinking..."
        
        # === AI MOVES ===
        # If the game isn't over, let the AI play when it's their turn
        
        if not game_over:
            if mode == 1:
                # Player vs AI mode: AI plays as Player 2 (yellow)
                if state.current_player == PLAYER2:
                    # Run MCTS to find the best move
                    # Use 2000 iterations for strong play
                    ai_move = mcts_search(state, iterations=3000)
                    
                    # If MCTS found a move, apply it
                    if ai_move is not None:
                        state.make_move(ai_move)
                        
                        # Check if AI's move ended the game
                        winner = state.check_winner()
                        if winner:
                            message = "Red wins! Press R to restart" if winner == PLAYER1 else "Yellow wins! Press R to restart"
                            game_over = True
                        elif state.is_full():
                            message = "Draw! Press R to restart"
                            game_over = True
                        else:
                            # Back to player's turn
                            message = "Red turn"
            
            elif mode == 2:
                # AI vs AI mode: Both players use MCTS
                
                # Run MCTS for current player
                # Use 1500 iterations (balanced speed vs strength)
                ai_move = mcts_search(state, iterations=1500)
                
                # Apply the move
                if ai_move is not None:
                    state.make_move(ai_move)
                    
                    # Check if move ended the game
                    winner = state.check_winner()
                    if winner:
                        message = "Red wins! Press R to restart" if winner == PLAYER1 else "Yellow wins! Press R to restart"
                        game_over = True
                    elif state.is_full():
                        message = "Draw! Press R to restart"
                        game_over = True
                    else:
                        # Update turn message
                        message = "Red turn" if state.current_player == PLAYER1 else "Yellow turn"
                
                # Add a delay in AI vs AI mode so humans can watch
                pygame.time.wait(500)  # 500ms pause between moves
        
        # === DRAW THE GAME ===
        # Render the current board state to the screen
        draw_board(screen, state, font, message)
    
    # === CLEANUP ===
    # When the game loop exits, quit Pygame properly
    pygame.quit()
    sys.exit()


# This ensures main() only runs when this file is executed directly,
# not when imported as a module
if __name__ == "__main__":
    main()