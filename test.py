from connect4_mcts import *

"""
Test suite to verify MCTS backpropagation is working correctly.

Add this to your connect4_mcts.py file or run it separately after importing the necessary classes.
"""

def create_forced_win_position():
    """
    Create a position where PLAYER1 has a forced win in column 3.
    
    Board layout (X = PLAYER1/Red, O = PLAYER2/Yellow):
    
    Col:  0 1 2 3 4 5 6
    Row 0: . . . . . . .
    Row 1: . . . . . . .
    Row 2: . . . . . . .
    Row 3: . . . X . . .
    Row 4: . . . X . . .
    Row 5: . . . X . . .
    
    PLAYER1 can win immediately by playing column 3 (completing vertical).
    """
    state = Connect4State()
    
    # Place three PLAYER1 pieces in column 3 (rows 3, 4, 5)
    state.board[5][3] = PLAYER1
    state.board[4][3] = PLAYER1
    state.board[3][3] = PLAYER1
    
    # Make sure it's PLAYER1's turn
    state.current_player = PLAYER1
    
    return state


def create_forced_loss_position():
    """
    Create a position where PLAYER1 will lose if they don't block.
    
    Board layout:
    
    Col:  0 1 2 3 4 5 6
    Row 0: . . . . . . .
    Row 1: . . . . . . .
    Row 2: . . . . . . .
    Row 3: . . . . . . .
    Row 4: . . . . . . .
    Row 5: O O O . . . .
    
    PLAYER2 has three in a row. PLAYER1 MUST play column 3 to block.
    """
    state = Connect4State()
    
    # Place three PLAYER2 pieces in a row
    state.board[5][0] = PLAYER2
    state.board[5][1] = PLAYER2
    state.board[5][2] = PLAYER2
    
    # It's PLAYER1's turn
    state.current_player = PLAYER1
    
    return state


def test_forced_win():
    """
    Test that MCTS correctly identifies a forced winning move.
    
    Expected behavior:
    - Column 3 should have the highest number of visits
    - Column 3 should have a win rate close to 1.0 (100%)
    - Other columns should have lower win rates
    """
    print("=" * 60)
    print("TEST 1: Forced Win Detection")
    print("=" * 60)
    print("\nSetup: PLAYER1 has three in a column, can win by playing column 3")
    print("\nExpected: Column 3 should dominate visits and have ~100% win rate\n")
    
    state = create_forced_win_position()
    
    # Print the board
    print("Current board:")
    for r in range(ROWS):
        row_str = ""
        for c in range(COLS):
            if state.board[r][c] == EMPTY:
                row_str += ". "
            elif state.board[r][c] == PLAYER1:
                row_str += "X "
            else:
                row_str += "O "
        print(f"Row {r}: {row_str}")
    print("Columns: 0 1 2 3 4 5 6\n")
    
    # Run MCTS
    print("Running MCTS with 5000 iterations...\n")
    best_move = mcts_search(state, iterations=5000)
    
    print(f"\n✓ MCTS selected move: {best_move}")
    
    if best_move == 3:
        print("✓ PASS: Correctly identified winning move (column 3)")
    else:
        print("✗ FAIL: Did not select winning move (expected column 3)")
    
    print("\n" + "=" * 60 + "\n")


def test_forced_block():
    """
    Test that MCTS correctly identifies a forced blocking move.
    
    Expected behavior:
    - Column 3 should have the highest number of visits
    - Column 3 should have a high win rate (it's the only non-losing move)
    - Other columns should have very low win rates (they lose immediately)
    """
    print("=" * 60)
    print("TEST 2: Forced Block Detection")
    print("=" * 60)
    print("\nSetup: PLAYER2 has three in a row, PLAYER1 must block at column 3")
    print("\nExpected: Column 3 should dominate visits and have highest win rate\n")
    
    state = create_forced_loss_position()
    
    # Print the board
    print("Current board:")
    for r in range(ROWS):
        row_str = ""
        for c in range(COLS):
            if state.board[r][c] == EMPTY:
                row_str += ". "
            elif state.board[r][c] == PLAYER1:
                row_str += "X "
            else:
                row_str += "O "
        print(f"Row {r}: {row_str}")
    print("Columns: 0 1 2 3 4 5 6\n")
    
    # Run MCTS
    print("Running MCTS with 5000 iterations...\n")
    best_move = mcts_search(state, iterations=5000)
    
    print(f"\n✓ MCTS selected move: {best_move}")
    
    if best_move == 3:
        print("✓ PASS: Correctly identified blocking move (column 3)")
    else:
        print("✗ FAIL: Did not select blocking move (expected column 3)")
    
    print("\n" + "=" * 60 + "\n")


def test_detailed_statistics():
    """
    Run a more detailed statistical analysis showing all moves.
    This helps visualize if backpropagation is working correctly.
    """
    print("=" * 60)
    print("TEST 3: Detailed Statistics Analysis")
    print("=" * 60)
    print("\nRunning forced win scenario with detailed output...\n")
    
    state = create_forced_win_position()
    
    # Import what we need for detailed analysis
    root_player = state.current_player
    root_node = MCTSNode(state.clone())
    
    # Run MCTS iterations manually to get access to root node
    iterations = 5000
    exploration_constant = 1.414
    
    for i in range(iterations):
        node = root_node
        sim_state = state.clone()
        
        # Selection
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.select_child_uct(exploration_constant)
            sim_state.make_move(node.move)
        
        # Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()
            sim_state = node.state.clone()
        
        # Simulation
        result = simulate_random_playout(sim_state, root_player)
        
        # Backpropagation (using the CORRECTED version)
        current_result = result
        while node is not None:
            node.update(current_result)
            current_result = 1.0 - current_result
            node = node.parent
    
    # Print detailed statistics
    print(f"Total iterations: {iterations}")
    print(f"Root visits: {root_node.visits}\n")
    
    print("Detailed move statistics:")
    print("-" * 80)
    print(f"{'Move':<6} {'Visits':<8} {'Wins':<10} {'Win Rate':<10} {'Visit %':<10} {'UCB Score':<10}")
    print("-" * 80)
    
    total_visits = sum(child.visits for child in root_node.children)
    
    for child in sorted(root_node.children, key=lambda c: c.visits, reverse=True):
        if child.visits > 0:
            win_rate = child.wins / child.visits
            visit_pct = (child.visits / total_visits) * 100
            
            exploitation = child.wins / child.visits
            exploration = exploration_constant * math.sqrt(
                math.log(root_node.visits) / child.visits
            )
            ucb = exploitation + exploration
            
            print(f"{child.move:<6} {child.visits:<8} {child.wins:<10.1f} {win_rate:<10.3f} {visit_pct:<10.1f}% {ucb:<10.3f}")
        else:
            print(f"{child.move:<6} {child.visits:<8} {'N/A':<10} {'N/A':<10} {'0.0%':<10} {'inf':<10}")
    
    print("-" * 80)
    
    # Analysis
    best_child = max(root_node.children, key=lambda c: c.visits)
    best_move = best_child.move
    best_winrate = best_child.wins / best_child.visits if best_child.visits > 0 else 0
    
    print(f"\nBest move by visits: {best_move}")
    print(f"Win rate of best move: {best_winrate:.3f}")
    
    if best_move == 3 and best_winrate > 0.95:
        print("\n✓ PASS: Column 3 has highest visits AND win rate > 95%")
    elif best_move == 3:
        print("\n⚠ PARTIAL: Column 3 selected but win rate is lower than expected")
        print("  This might indicate backpropagation issues")
    else:
        print("\n✗ FAIL: Column 3 not selected as best move")
        print("  This indicates serious backpropagation issues")
    
    print("\n" + "=" * 60 + "\n")


def run_all_tests():
    """
    Run all MCTS tests in sequence.
    """
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "MCTS BACKPROPAGATION TEST SUITE" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    test_forced_win()
    test_forced_block()
    test_detailed_statistics()
    
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 20 + "TESTS COMPLETE" + " " * 24 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    print("How to interpret results:")
    print("- If PASS: Your backpropagation is working correctly")
    print("- If FAIL: You have backpropagation issues (check the logic)")
    print("- Look for column 3 having 80%+ of visits in forced win scenario")
    print("- Win rates should be close to 1.0 for winning moves")
    print("\n")

run_all_tests()