import pytest
import chess
import jax.numpy as jnp
from liquid_chess.models.lrt.feature_extraction import (
    ChessFeatureExtractor,
    board_to_enhanced_input
)


def test_attack_maps_starting_position():
    """Test attack map extraction from starting position"""
    board = chess.Board()
    extractor = ChessFeatureExtractor()
    
    white_attacks, black_attacks = extractor.extract_attack_maps(board)
    
    # Both sides should have some attacks
    assert white_attacks.sum() > 0
    assert black_attacks.sum() > 0
    
    # Attacks should be symmetric in starting position
    assert abs(white_attacks.sum() - black_attacks.sum()) < 5


def test_pin_detection():
    """Test pin detection"""
    # Create a position with a clear pin: Rook pins Knight to King on same file
    # Position: White King e1, Black King e8, Black Knight e5, White Rook e3
    # The knight on e5 is pinned by the rook on e3 to the king on e8
    board = chess.Board()
    board.clear()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.E5, chess.Piece(chess.KNIGHT, chess.BLACK))
    board.set_piece_at(chess.E3, chess.Piece(chess.ROOK, chess.WHITE))
    
    extractor = ChessFeatureExtractor()
    
    pins = extractor.extract_pin_detection(board)
    
    # Debug: Check if position actually has pins
    print(f"Board position: {board.fen()}")
    print(f"Pins detected: {pins.sum()}")
    
    # Check manually if there are any pins
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        print(f"{color} king at {king_sq}")
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                is_pinned = board.is_pinned(color, square)
                if is_pinned:
                    print(f"  {piece} at {square} is pinned")
    
    # The knight on e5 should be pinned
    assert pins.sum() == 1.0, f"Expected 1 pin, got {pins.sum()}"
    assert pins[chess.E5] == 1.0, "Knight on e5 should be pinned"


def test_king_safety_zones():
    """Test king safety zone extraction"""
    board = chess.Board()
    extractor = ChessFeatureExtractor()
    
    white_safety, black_safety = extractor.extract_king_safety(board)
    
    # Each king should have a 3x3 safety zone (9 squares), but kings on back rank have 6 squares
    # White king on e1 (back rank) -> 6 squares
    # Black king on e8 (back rank) -> 6 squares
    assert white_safety.sum() == 6
    assert black_safety.sum() == 6


def test_passed_pawns():
    """Test passed pawn detection"""
    # Position with white passed pawn on d5
    board = chess.Board("8/8/8/3P4/8/8/8/8 w - - 0 1")
    extractor = ChessFeatureExtractor()
    
    pawn_structure = extractor.extract_pawn_structure(board)
    
    # Should detect the passed pawn
    assert pawn_structure['passed_pawns'].sum() == 1.0


def test_isolated_pawns():
    """Test isolated pawn detection"""
    # Position with isolated pawns on a and h files
    board = chess.Board("8/8/8/P6P/8/8/8/8 w - - 0 1")
    extractor = ChessFeatureExtractor()
    
    pawn_structure = extractor.extract_pawn_structure(board)
    
    # Both pawns should be isolated
    assert pawn_structure['isolated_pawns'].sum() == 2.0


def test_doubled_pawns():
    """Test doubled pawn detection"""
    # Position with doubled pawns on the a-file
    board = chess.Board("8/8/P7/P7/8/8/8/8 w - - 0 1")
    extractor = ChessFeatureExtractor()
    
    pawn_structure = extractor.extract_pawn_structure(board)
    
    # Both pawns should be marked as doubled
    assert pawn_structure['doubled_pawns'].sum() == 2.0


def test_material_imbalance():
    """Test material imbalance calculation"""
    extractor = ChessFeatureExtractor()
    
    # Equal material
    board = chess.Board()
    material = extractor.extract_material_imbalance(board)
    assert abs(material[0]) < 0.01  # Should be near zero
    
    # White up a queen (remove black queen from starting position)
    board = chess.Board("rnbk1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    material = extractor.extract_material_imbalance(board)
    assert material[0] > 0.2  # Positive for white advantage


def test_board_to_enhanced_input():
    """Test full board conversion"""
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    
    features = board_to_enhanced_input(board)
    
    # Check all required keys are present
    required_keys = [
        'pieces', 'turn', 'castling', 'ep_square',
        'white_attacks', 'black_attacks', 'pins',
        'king_safety_white', 'king_safety_black',
        'passed_pawns', 'isolated_pawns', 'doubled_pawns',
        'material_imbalance'
    ]
    
    for key in required_keys:
        assert key in features, f"Missing key: {key}"
    
    # Check shapes
    assert features['pieces'].shape == (8, 8)
    assert features['white_attacks'].shape == (64,)
    assert features['material_imbalance'].shape == (64,)


def test_feature_extraction_performance():
    """Test that feature extraction is reasonably fast"""
    import time
    
    board = chess.Board()
    
    start = time.time()
    for _ in range(20):  
        _ = board_to_enhanced_input(board)
    elapsed = time.time() - start
    
    # Should process 20 positions in under 3 seconds (very relaxed)
    assert elapsed < 3.0, f"Feature extraction too slow: {elapsed:.2f}s for 20 positions"