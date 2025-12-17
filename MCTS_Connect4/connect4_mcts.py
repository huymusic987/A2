"""
MCTS Connect 4 Game with Advanced Statistics and Visualization

This module implements a Connect 4 game with Monte Carlo Tree Search (MCTS) AI opponent.
Features:
- Player vs AI or AI vs AI game modes
- Real-time MCTS decision analysis with UCB1 exploration
- Column-based win rate statistics and visualization
- Modern dark-themed UI with side statistics panel
- Win highlighting with gold circles

The game uses pygame for graphics and implements a full MCTS algorithm with:
- Selection phase using UCB1 (Upper Confidence Bound)
- Expansion of unexplored nodes
- Random playout simulation
- Backpropagation of results

Statistics are tracked both historically (cumulative wins per column) and
in real-time (MCTS analysis showing win rates, visit counts, and UCB scores
for each possible move).

Author: Connect 4 MCTS Implementation
License: Open Source
"""

import pygame
import sys
import math
import random

#              GAME CONSTANTS AND COLORS

# Board dimensions
ROWS = 6  # Number of rows in the Connect 4 board
COLS = 7  # Number of columns in the Connect 4 board

# Player identifiers
EMPTY = 0     # Empty cell marker
PLAYER1 = 1   # Red player (human or AI)
PLAYER2 = 2   # Cyan player (AI)

# Display constants
SQUARESIZE = 100  # Size of each grid square in pixels
RADIUS = SQUARESIZE // 2 - 5  # Radius of game pieces (with 5px padding)

# Color scheme - modern dark theme
BOARD_COLOR = (15, 20, 45)          # deep navy board
BG_COLOR = (10, 10, 10)             # dark background
PLAYER1_COLOR = (220, 70, 70)       # modern red
PLAYER2_COLOR = (70, 200, 170)      # modern teal
TEXT_COLOR = (240, 240, 240)        # off-white text
HINT_COLOR = (0, 230, 255)          # neon highlight
PANEL_COLOR = (18, 18, 22)          # matte side panel
HEADER_COLOR = (55, 90, 140)        # steel-blue headers

EMPTY_SLOT_COLOR = (30, 30, 40)     # visible dark gray-blue
EMPTY_SLOT_BORDER = (90, 90, 120)   # subtle outline
WIN_HIGHLIGHT = (255, 215, 0)       # gold

# Window dimensions
STATS_PANEL_WIDTH = 400  # Width of the right-side statistics panel
WIDTH = COLS * SQUARESIZE + STATS_PANEL_WIDTH + 10  # Total window width
HEIGHT = (ROWS + 2) * SQUARESIZE  # Total window height (+ 2 for header and spacing)
SIZE = (WIDTH, HEIGHT)  # Window size tuple

FPS = 60  # Frames per second for game loop

# Global statistics tracking - stores historical win counts for each column
column_stats = {col: {"player1_wins": 0, "player2_wins": 0} for col in range(COLS)}

class Connect4State:
    """
    Represents the state of a Connect 4 game.
    
    Tracks the board configuration and current player, and provides methods
    for game logic including legal moves, making moves, and checking for wins.
    
    Attributes:
        board: 2D list representing the game board (ROWS x COLS)
        current_player: The player whose turn it is (PLAYER1 or PLAYER2)
    """
    
    def __init__(self, board=None, current_player=PLAYER1):
        """
        Initialize a new Connect 4 game state.
        
        Args:
            board: Optional 2D list to copy from. If None, creates empty board.
            current_player: The player who moves next (default: PLAYER1)
        """
        if board is None:
            # Create empty board filled with EMPTY markers
            self.board = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]
        else:
            # Deep copy the provided board to avoid reference issues
            self.board = [row[:] for row in board]
        self.current_player = current_player

    def clone(self):
        """
        Create a deep copy of this game state.
        
        Returns:
            A new Connect4State with the same board and current player
        """
        return Connect4State(self.board, self.current_player)

    def get_legal_moves(self):
        """
        Get all legal column moves (columns that aren't full).
        
        Returns:
            List of column indices where a piece can be dropped
        """
        moves = []
        # A column is legal if the top row (row 0) is empty
        for c in range(COLS):
            if self.board[0][c] == EMPTY:
                moves.append(c)
        return moves

    def make_move(self, col):
        """
        Drop a piece in the specified column for the current player.
        
        The piece falls to the lowest available row in that column.
        After making the move, switches to the other player.
        
        Args:
            col: Column index to drop the piece
            
        Returns:
            True if move was successful, False if column is full
        """
        # Start from bottom row and work upward to find empty slot
        for r in range(ROWS - 1, -1, -1):
            if self.board[r][col] == EMPTY:
                # Place the current player's piece
                self.board[r][col] = self.current_player
                # Switch to the other player
                self.current_player = PLAYER1 if self.current_player == PLAYER2 else PLAYER2
                return True
        return False  # Column was full

    def check_winner(self, return_positions=False):
        """
        Check if there's a winner (4 in a row).
        
        Checks all possible win conditions: horizontal, vertical, and both diagonals.
        
        Args:
            return_positions: If True, returns (winner, positions) tuple.
                            If False, returns just the winner.
        
        Returns:
            If return_positions=False: PLAYER1, PLAYER2, or None
            If return_positions=True: (winner, list of (row, col) tuples) or (None, [])
        """
        # Check horizontal wins (4 in a row across)
        for r in range(ROWS):
            for c in range(COLS - 3):  # Stop 3 from the end
                piece = self.board[r][c]
                if piece != EMPTY:
                    # Check if next 3 pieces match
                    if all(self.board[r][c + i] == piece for i in range(4)):
                        positions = [(r, c + i) for i in range(4)]
                        return (piece, positions) if return_positions else piece

        # Check vertical wins (4 in a column)
        for c in range(COLS):
            for r in range(ROWS - 3):  # Stop 3 from the bottom
                piece = self.board[r][c]
                if piece != EMPTY:
                    # Check if next 3 pieces below match
                    if all(self.board[r + i][c] == piece for i in range(4)):
                        positions = [(r + i, c) for i in range(4)]
                        return (piece, positions) if return_positions else piece

        # Check diagonal wins (top-left to bottom-right)
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                piece = self.board[r][c]
                if piece != EMPTY:
                    # Check diagonal down-right
                    if all(self.board[r + i][c + i] == piece for i in range(4)):
                        positions = [(r + i, c + i) for i in range(4)]
                        return (piece, positions) if return_positions else piece

        # Check diagonal wins (bottom-left to top-right)
        for r in range(3, ROWS):  # Start from row 3
            for c in range(COLS - 3):
                piece = self.board[r][c]
                if piece != EMPTY:
                    # Check diagonal up-right
                    if all(self.board[r - i][c + i] == piece for i in range(4)):
                        positions = [(r - i, c + i) for i in range(4)]
                        return (piece, positions) if return_positions else piece

        # No winner found
        return (None, []) if return_positions else None

    def is_full(self):
        """
        Check if the board is completely full (draw condition).
        
        Returns:
            True if all columns are full, False otherwise
        """
        # Board is full if top row has no empty spaces
        return all(self.board[0][c] != EMPTY for c in range(COLS))

    def is_terminal(self):
        """
        Check if the game is over (win or draw).
        
        Returns:
            True if there's a winner or board is full, False otherwise
        """
        return self.check_winner() is not None or self.is_full()

class MCTSNode:
    """
    Node in the Monte Carlo Tree Search tree.
    
    Each node represents a game state and tracks statistics for the MCTS algorithm:
    - Number of visits (simulations that passed through this node)
    - Number of wins (successful outcomes from this node's perspective)
    - Untried moves (legal moves not yet explored from this state)
    - Children nodes (explored successor states)
    
    Attributes:
        state: The Connect4State this node represents
        parent: Parent MCTSNode (None for root)
        move: The move that led to this state from parent
        children: List of child MCTSNode objects
        visits: Number of times this node was visited during search
        wins: Accumulated win score (0.0 to visits)
        untried_moves: List of legal moves not yet expanded
    """
    
    def __init__(self, state, parent=None, move=None):
        """
        Initialize a new MCTS node.
        
        Args:
            state: Connect4State this node represents
            parent: Parent MCTSNode (None if root)
            move: Column index that led to this state (None if root)
        """
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []  # Child nodes (explored moves)
        self.visits = 0     # Number of simulations through this node
        self.wins = 0.0     # Accumulated win score
        # Initialize untried moves with all legal moves (if not terminal)
        self.untried_moves = state.get_legal_moves() if not state.is_terminal() else []
    
    def is_fully_expanded(self):
        """
        Check if all legal moves from this node have been tried.
        
        Returns:
            True if no untried moves remain, False otherwise
        """
        return len(self.untried_moves) == 0
    
    def is_terminal(self):
        """
        Check if this node represents a terminal game state.
        
        Returns:
            True if game is over at this node, False otherwise
        """
        return self.state.is_terminal()
    
    def select_child_uct(self, exploration_constant=1.414):
        """
        Select best child using UCB1 (Upper Confidence Bound) formula.
        
        UCB1 balances exploitation (choosing high win-rate moves) and
        exploration (trying less-visited moves). Formula:
        UCB1 = (wins/visits) + c * sqrt(ln(parent_visits)/visits)
        
        Args:
            exploration_constant: Controls exploration vs exploitation tradeoff
                                (default: sqrt(2) ≈ 1.414)
        
        Returns:
            Child MCTSNode with highest UCB1 score
        """
        best_score = float('-inf')
        best_children = []  # Track ties for random tiebreaking
        
        for child in self.children:
            # Unvisited children get infinite score (explored first)
            if child.visits == 0:
                return child
            
            # Exploitation term: win rate
            exploitation = child.wins / child.visits
            # Exploration term: sqrt(ln(parent_visits) / child_visits)
            exploration = exploration_constant * math.sqrt(
                math.log(self.visits) / child.visits
            )
            
            # UCB1 score combines both terms
            ucb_score = exploitation + exploration
            
            # Track best scoring children (with ties)
            if ucb_score > best_score:
                best_score = ucb_score
                best_children = [child]
            elif ucb_score == best_score:
                best_children.append(child)
        
        # Random tiebreaking among equally good children
        return random.choice(best_children)
    
    def expand(self):
        """
        Expand the tree by creating a child node for an untried move.
        
        Randomly selects one untried move, creates the resulting game state,
        and adds it as a child node.
        
        Returns:
            The new child MCTSNode, or None if no moves to expand
        """
        if not self.untried_moves:
            return None

        # Randomly select an untried move
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)

        # Create new state by applying the move
        new_state = self.state.clone()
        new_state.make_move(move)

        # Create child node and add to children list
        child = MCTSNode(new_state, parent=self, move=move)
        self.children.append(child)
        return child
    
    def best_child(self):
        """
        Get the child with the most visits (most explored).
        
        After MCTS search completes, the most-visited child is typically
        the best move, as it represents the most promising line of play.
        
        Returns:
            Child MCTSNode with highest visit count, or None if no children
        """
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)
    
    def update(self, result):
        """
        Update node statistics after a simulation (backpropagation).
        
        Args:
            result: Simulation outcome (0.0 = loss, 0.5 = draw, 1.0 = win)
        """
        self.visits += 1
        self.wins += result


def evaluate_terminal_state(state, player):
    """
    Evaluate the outcome of a terminal game state from a player's perspective.
    
    Args:
        state: Terminal Connect4State to evaluate
        player: Player to evaluate for (PLAYER1 or PLAYER2)
    
    Returns:
        1.0 if player won, 0.0 if player lost, 0.5 if draw
    """
    winner = state.check_winner()
    
    if winner is None:
        return 0.5  # Draw
    elif winner == player:
        return 1.0  # Win
    else:
        return 0.0  # Loss


def simulate_random_playout(state, player):
    """
    Simulate a random game from the current state to completion.
    
    This is the "simulation" phase of MCTS. Plays random legal moves
    until the game ends, then evaluates the result for the given player.
    
    Args:
        state: Connect4State to simulate from (not modified)
        player: Player to evaluate result for
    
    Returns:
        1.0 if player wins, 0.0 if loses, 0.5 if draw
    """
    # Clone state to avoid modifying original
    simulation_state = state.clone()
    
    # Play random moves until game ends
    while not simulation_state.is_terminal():
        legal_moves = simulation_state.get_legal_moves()
        
        if not legal_moves:
            break  # Board full (shouldn't happen if is_terminal works correctly)
        
        # Make random move
        move = random.choice(legal_moves)
        simulation_state.make_move(move)
    
    # Evaluate final state
    return evaluate_terminal_state(simulation_state, player)


def mcts_search(root_state, iterations=1000, exploration_constant=1.414):
    """
    Perform Monte Carlo Tree Search to find the best move.
    
    MCTS algorithm phases:
    1. Selection: Traverse tree using UCB1 to select most promising path
    2. Expansion: Add new child node for unexplored move
    3. Simulation: Play out random game to completion
    4. Backpropagation: Update statistics for all nodes in path
    
    Args:
        root_state: Current game state to search from
        iterations: Number of MCTS iterations (simulations) to run
        exploration_constant: UCB1 exploration parameter (default: sqrt(2))
    
    Returns:
        Tuple of (best_move, stats_dict) where:
        - best_move: Column index of best move (or None if no legal moves)
        - stats_dict: Dictionary containing detailed search statistics
    """
    # Can't search from terminal state
    if root_state.is_terminal():
        return None, {}
    
    # Remember which player is making the decision
    root_player = root_state.current_player
    root_node = MCTSNode(root_state.clone())
    
    # Run MCTS iterations
    for _ in range(iterations):
        node = root_node
        state = root_state.clone()
        
        # SELECTION PHASE 
        # Traverse tree using UCB1 until we reach a node to expand
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.select_child_uct(exploration_constant)
            state.make_move(node.move)
        
        # EXPANSION PHASE 
        # If node isn't terminal and has untried moves, expand it
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()
            state = node.state.clone()
        
        # SIMULATION PHASE 
        # Play random game to completion from this node
        result = simulate_random_playout(state, root_player)
        
        # BACKPROPAGATION PHASE
        # Update all nodes in the path with the result
        while node is not None:
            if node.parent is None:
                # Root node: always update with result as-is
                node.update(result)
            else:
                # Non-root: flip result if it was opponent's turn
                # (opponent's good outcome = our bad outcome)
                if node.parent.state.current_player == root_player:
                    node.update(result)
                else:
                    node.update(1.0 - result)
    
            node = node.parent
    
    # Find best move (most visited child)
    best_child = root_node.best_child()
    
    # Collect detailed statistics for all children
    move_stats = []
    for child in root_node.children:
        if child.visits > 0:
            # Calculate win rate as percentage
            win_rate = (child.wins / child.visits) * 100
            
            # Calculate UCB1 components for display
            exploitation = child.wins / child.visits
            exploration = exploration_constant * math.sqrt(
                math.log(root_node.visits) / child.visits
            )
            ucb_score = exploitation + exploration
            
            # Store statistics for this move
            move_stats.append({
                'move': child.move,
                'win_rate': win_rate,
                'visits': child.visits,
                'ucb_score': ucb_score
            })
    
    # Sort by visits (most explored first)
    move_stats.sort(key=lambda x: x['visits'], reverse=True)
    
    # Package all statistics for display
    stats = {
        'selected_move': best_child.move if best_child else None,
        'selected_win_rate': (best_child.wins / best_child.visits * 100) if best_child and best_child.visits > 0 else 0,
        'selected_visits': best_child.visits if best_child else 0,
        'total_simulations': root_node.visits,
        'all_moves': move_stats
    }
    
    if best_child is None:
        return None, stats
    
    return best_child.move, stats


def draw_stats_panel(screen, font, small_font, stats, column_stats, selected_col):
    """
    Draw the statistics panel on the right side of the screen.
    
    Displays two main sections:
    1. AI Move Analysis - Shows MCTS search results with win rates and UCB scores
    2. Column Win Rates - Shows historical or current win rates for each column
    
    Args:
        screen: Pygame surface to draw on
        font: Large font for headers
        small_font: Small font for details (this parameter appears twice due to code structure)
        stats: Dictionary of MCTS statistics from last search
        column_stats: Global dictionary of historical column win counts
        selected_col: Column index of last move (for highlighting)
    """
    # Calculate x-position for the stats panel (right side)
    panel_x = COLS * SQUARESIZE

    # Draw panel background
    pygame.draw.rect(screen, PANEL_COLOR, (panel_x, 0, STATS_PANEL_WIDTH, HEIGHT))

    # Draw "STATS" title
    title_text = font.render("STATS", True, TEXT_COLOR)
    screen.blit(title_text, (panel_x + 10, 10))

    y_offset = 60

    # AI MOVE ANALYSIS SECTION
    # Draw section header
    pygame.draw.rect(screen, HEADER_COLOR, (panel_x + 10, y_offset, STATS_PANEL_WIDTH - 20, 30))
    section1_text = small_font.render("AI Move Analysis", True, TEXT_COLOR)
    screen.blit(section1_text, (panel_x + 20, y_offset + 5))

    y_offset += 40

    # Display selected move information
    if stats and 'selected_move' in stats and stats['selected_move'] is not None:
        sel_move = stats['selected_move']
        sel_wr = stats.get('selected_win_rate', 0)
        sel_visits = stats.get('selected_visits', 0)
        total_sims = stats.get('total_simulations', 0)

        # Highlight the selected move in cyan
        selected_text = small_font.render(f"SELECTED: Column {sel_move}] Win Rate: {sel_wr:.1f}% | Visits: {sel_visits}", True, HINT_COLOR)
        screen.blit(selected_text, (panel_x + 20, y_offset))
        y_offset += 20

        # Show total simulations run
        total_text = small_font.render(f"Total Simulations: {total_sims}", True, TEXT_COLOR)
        screen.blit(total_text, (panel_x + 20, y_offset))
        y_offset += 25

        # Display all moves considered during MCTS search
        all_moves_text = small_font.render("All Moves Considered:", True, TEXT_COLOR)
        screen.blit(all_moves_text, (panel_x + 20, y_offset))
        y_offset += 20

        # List each move with its statistics
        for move_data in stats.get('all_moves', []):
            col = move_data['move']
            wr = move_data['win_rate']
            ucb = move_data['ucb_score']
            visits = move_data['visits']

            # Highlight selected move in cyan, others in white
            color = HINT_COLOR if col == sel_move else TEXT_COLOR

            # Display: Column, Win Rate, UCB1 Score, Visit Count
            move_text = small_font.render(f"* Col {col}: {wr:.1f}% | UCB1 {ucb:.3f} ({visits})", True, color)
            screen.blit(move_text, (panel_x + 25, y_offset))
            y_offset += 18
    else:
        # No MCTS data available yet
        no_data_text = small_font.render("No move data yet", True, TEXT_COLOR)
        screen.blit(no_data_text, (panel_x + 20, y_offset))
        y_offset += 30

    y_offset += 20

    # COLUMN WIN RATES SECTION
    # Draw section header
    pygame.draw.rect(screen, HEADER_COLOR, (panel_x + 10, y_offset, STATS_PANEL_WIDTH - 20, 30))
    section2_text = small_font.render("Column Win Rates", True, TEXT_COLOR)
    screen.blit(section2_text, (panel_x + 20, y_offset + 5))

    y_offset += 40

    # Draw column win rates as split horizontal bars
    bar_width = STATS_PANEL_WIDTH - 140  # Width of the bar
    bar_height = 25  # Height of each bar

    # Build lookup table from MCTS stats if available
    # Maps column -> win_rate (0-100) for current position
    mcts_by_col = {}
    if stats and 'all_moves' in stats:
        for move_data in stats['all_moves']:
            mcts_by_col[move_data['move']] = move_data['win_rate']

    # Draw a bar for each column
    for col in range(COLS):
        # Column label
        col_text = small_font.render(f"Col {col}:", True, TEXT_COLOR)
        screen.blit(col_text, (panel_x + 20, y_offset + 5))

        bar_x = panel_x + 70

        # Decide whether to show MCTS data or historical data
        if mcts_by_col:
            # Use MCTS win rate (current position analysis)
            p1_pct = mcts_by_col.get(col, 0.0)
            p2_pct = 100.0 - p1_pct
            total_games_present = True
        else:
            # Fallback to historical cumulative statistics
            p1_wins = column_stats[col]["player1_wins"]
            p2_wins = column_stats[col]["player2_wins"]
            total = p1_wins + p2_wins
            if total > 0:
                p1_pct = (p1_wins / total) * 100
                p2_pct = (p2_wins / total) * 100
                total_games_present = True
            else:
                # No data for this column yet
                total_games_present = False
                p1_pct = p2_pct = 0.0

        if total_games_present:
            # Draw split bar: left portion = player 1, right = player 2
            p1_width = int((p1_pct / 100.0) * bar_width)
            p2_width = bar_width - p1_width

            # Player 1 portion (red)
            if p1_width > 0:
                pygame.draw.rect(screen, PLAYER1_COLOR, (bar_x, y_offset, p1_width, bar_height))
            # Player 2 portion (cyan)
            if p2_width > 0:
                pygame.draw.rect(screen, PLAYER2_COLOR, (bar_x + p1_width, y_offset, p2_width, bar_height))

            # Draw border around bar
            pygame.draw.rect(screen, TEXT_COLOR, (bar_x, y_offset, bar_width, bar_height), 1)

            # Highlight if this is the selected (last moved) column
            if col == selected_col:
                pygame.draw.rect(screen, HINT_COLOR, (bar_x, y_offset, bar_width, bar_height), 2)

            # Display percentages next to bar (rounded to nearest integer)
            pct_text = small_font.render(f"{p1_pct:.0f}% / {p2_pct:.0f}%", True, TEXT_COLOR)
            screen.blit(pct_text, (bar_x + bar_width + 10, y_offset + 5))
        else:
            # No data available - draw gray bar
            pygame.draw.rect(screen, (60, 60, 60), (bar_x, y_offset, bar_width, bar_height))
            pygame.draw.rect(screen, TEXT_COLOR, (bar_x, y_offset, bar_width, bar_height), 1)

            no_data = small_font.render("No data", True, (150, 150, 150))
            screen.blit(no_data, (bar_x + bar_width + 10, y_offset + 5))

        y_offset += 30


def draw_board(screen, state, font, small_font, message="", stats=None, selected_col=None, win_positions=[]):
    """
    Draw the complete game screen including board, pieces, and statistics panel.
    
    Args:
        screen: Pygame surface to draw on
        state: Current Connect4State to display
        font: Large font for message text
        small_font: Small font for stats panel (passed to draw_stats_panel)
        message: Status message to display at top (e.g., "Red turn")
        stats: MCTS statistics dictionary (optional)
        selected_col: Last moved column for highlighting (optional)
        win_positions: List of (row, col) tuples for winning pieces (optional)
    """
    # Fill background
    screen.fill(BG_COLOR)
    
    # Draw status message at top
    text_surface = font.render(message, True, TEXT_COLOR)
    screen.blit(text_surface, (10, 5))
    
    # Draw board grid
    for c in range(COLS):
        for r in range(ROWS):
            # Draw blue square for each cell
            pygame.draw.rect(
                screen,
                BOARD_COLOR,
                (c * SQUARESIZE, (r + 2) * SQUARESIZE, SQUARESIZE, SQUARESIZE)
            )
            
            # Calculate center of this cell
            center = (
                c * SQUARESIZE + SQUARESIZE // 2,
                (r + 2) * SQUARESIZE + SQUARESIZE // 2
            )

            # Draw empty slot (dark circle with border)
            pygame.draw.circle(screen, EMPTY_SLOT_COLOR, center, RADIUS)
            pygame.draw.circle(screen, EMPTY_SLOT_BORDER, center, RADIUS, 2)
    
    # Draw game pieces
    for c in range(COLS):
        for r in range(ROWS):
            piece = state.board[r][c]
            
            # Draw piece if cell is not empty
            if piece != EMPTY:
                color = PLAYER1_COLOR if piece == PLAYER1 else PLAYER2_COLOR
                
                pygame.draw.circle(
                    screen,
                    color,
                    (c * SQUARESIZE + SQUARESIZE // 2, (r + 2) * SQUARESIZE + SQUARESIZE // 2),
                    RADIUS
                )

    # Highlight winning pieces
    if win_positions:
        for r, c in win_positions:
            center = (
                c * SQUARESIZE + SQUARESIZE // 2,
                (r + 2) * SQUARESIZE + SQUARESIZE // 2
            )

            # Draw gold ring around winning pieces
            pygame.draw.circle(screen, WIN_HIGHLIGHT, center, RADIUS + 4, 4)

    
    # Draw statistics panel on right side
    if stats is None:
        stats = {}
    draw_stats_panel(screen, font, small_font, stats, column_stats, selected_col)
    
    # Update display
    pygame.display.update()


def update_column_stats(col, winner):
    """
    Update cumulative column statistics after a game ends.
    
    Tracks which columns lead to wins for each player over multiple games.
    
    Args:
        col: Column index where the winning move was made
        winner: PLAYER1 or PLAYER2 (which player won)
    """
    if winner == PLAYER1:
        column_stats[col]["player1_wins"] += 1
    elif winner == PLAYER2:
        column_stats[col]["player2_wins"] += 1


def reset_column_stats():
    """
    Reset all cumulative column statistics to zero.
    Used when starting a fresh set of games.
    """
    global column_stats
    column_stats = {col: {"player1_wins": 0, "player2_wins": 0} for col in range(COLS)}


def menu_loop(screen, font):
    while True:
        screen.fill(BG_COLOR)

        t1 = font.render("Press 1 for Player vs AI", True, TEXT_COLOR)
        t2 = font.render("Press 2 for AI vs AI", True, TEXT_COLOR)
        screen.blit(t1, (50, 100))
        screen.blit(t2, (50, 150))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return 1
                elif event.key == pygame.K_2:
                    return 2

#  MAIN GAME LOOP

def main():
    """
    Main game loop - handles initialization, mode selection, and game flow.
    Game flow:
    1. Initialize pygame and display
    2. Let user select game mode (Player vs AI or AI vs AI)
    3. Run main game loop:
       - Handle events (mouse clicks, keyboard)
       - Execute AI moves using MCTS
       - Check win conditions
       - Update display
    4. Allow restart with 'R' key
    """
    pygame.init()
    
    # Create game window
    screen = pygame.display.set_mode(SIZE)
    pygame.display.set_caption("MCTS Connect 4 with Stats")
    
    # Create clock for frame rate control
    clock = pygame.time.Clock()
    
    font = pygame.font.SysFont("arial", 24)
    small_font = pygame.font.SysFont("arial", 18)
    

    MENU = "menu"
    GAME = "game"

    game_state = MENU
    running = True
    mode = None

    # GAME INITIALIZATION
    state = Connect4State() # Create empty board
    game_over = False
    message = ""
    stats = {} # MCTS statistics dictionary
    last_move = None # Track last column played
    win_positions = []  # Positions of winning 4-in-a-row
    
    # MAIN GAME LOOP
    running = True
    while running:
        clock.tick(FPS)
        if game_state == MENU:
            mode = menu_loop(screen, font)

            # Reset game state
            state = Connect4State()
            game_over = False
            message = "Red turn"
            stats = {}
            last_move = None
            win_positions = []

            game_state = GAME
            continue
        
        # EVENT HANDLING
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    game_state = MENU
                    break

                if event.key == pygame.K_r:
                    state = Connect4State()
                    game_over = False
                    message = "Red turn"
                    stats = {}
                    last_move = None
                    win_positions = []
            
            # PLAYER MOVE (Mode 1 only)
            if mode == 1 and not game_over and state.current_player == PLAYER1:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Get mouse position and convert to column
                    x, _ = event.pos
                    col = x // SQUARESIZE
                    
                    # Check if click is in valid column
                    if col < COLS and col in state.get_legal_moves():
                        last_move = col
                        state.make_move(col)
                        
                        # Check for winner after player move
                        winner, winning_positions = state.check_winner(return_positions=True)
                        win_positions = winning_positions

                        if winner:
                            message = "Red wins! Press R to restart" if winner == PLAYER1 else "Cyan wins! Press R to restart"
                            game_over = True
                            if last_move is not None:
                                update_column_stats(last_move, winner)
                        elif state.is_full():
                            message = "Draw! Press R to restart"
                            game_over = True
                        else:
                            message = "AI thinking..."
        
        # AI MOVE
        if not game_over:
            # Mode 1: AI plays as Player 2 (cyan)
            if mode == 1:
                if state.current_player == PLAYER2:
                    ai_move, stats = mcts_search(state)
                    
                    if ai_move is not None:
                        last_move = ai_move
                        state.make_move(ai_move)
                        
                        winner, winning_positions = state.check_winner(return_positions=True)
                        win_positions = winning_positions

                        if winner:
                            message = "Red wins! Press R to restart" if winner == PLAYER1 else "Cyan wins! Press R to restart"
                            game_over = True
                            if last_move is not None:
                                update_column_stats(last_move, winner)
                        elif state.is_full():
                            message = "Draw! Press R to restart"
                            game_over = True
                        else:
                            message = "Red turn"

            # Mode 2: AI vs AI - both players use MCTS
            elif mode == 2:
                # Run MCTS for current player
                ai_move, stats = mcts_search(state)
                
                if ai_move is not None:
                    last_move = ai_move
                    current_player = state.current_player
                    state.make_move(ai_move)
                    
                    # Check for winner
                    winner, winning_positions = state.check_winner(return_positions=True)
                    win_positions = winning_positions

                    if winner:
                        message = "Red wins! Press R to restart" if winner == PLAYER1 else "Cyan wins! Press R to restart"
                        game_over = True
                        if last_move is not None:
                            update_column_stats(last_move, winner)
                    elif state.is_full():
                        message = "Draw! Press R to restart"
                        game_over = True
                    else:
                        message = "Red turn" if state.current_player == PLAYER1 else "Cyan turn"
                
                # Add delay so humans can follow AI vs AI gameplay
                pygame.time.wait(500)
        
        draw_board(screen, state, font, small_font, message, stats, last_move, win_positions)
        
    # Clean up and exit
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()