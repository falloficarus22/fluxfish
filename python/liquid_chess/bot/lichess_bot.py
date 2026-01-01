import time
import numpy as np
import chess
import chess.engine
from liquid_chess.models.lrt import UltraFastLRT
from liquid_chess.bridge.cpp_bridge import JAXLRTBridge
import jax
import jax.numpy as jnp

class PerformanceBenchmark:
    """Benchmark suite for Liquid Chess Engine"""
    
    def __init__(self):
        self.results = {}
        
    def benchmark_move_generation(self, num_positions=1000):
        """Benchmark move generation speed"""
        print("Benchmarking move generation...")
        
        # Create random positions
        positions = []
        board = chess.Board()
        for _ in range(num_positions):
            # Make some random moves
            for _ in range(np.random.randint(0, 20)):
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    board.push(np.random.choice(legal_moves))
            positions.append(board.copy())
        
        # Time move generation
        start = time.time()
        total_moves = 0
        
        for board in positions:
            moves = list(board.legal_moves)
            total_moves += len(moves)
        
        elapsed = time.time() - start
        moves_per_second = total_moves / elapsed
        
        self.results['move_generation'] = {
            'positions': num_positions,
            'total_moves': total_moves,
            'time_s': elapsed,
            'moves_per_second': moves_per_second,
            'moves_per_position': total_moves / num_positions
        }
        
        print(f"  Generated {total_moves:,} moves in {elapsed:.2f}s")
        print(f"  Speed: {moves_per_second:,.0f} moves/second")
    
    def benchmark_lrt_evaluation(self, num_positions=100):
        """Benchmark LRT evaluation speed"""
        print("\nBenchmarking LRT evaluation...")
        
        # Initialize model
        model = UltraFastLRT()
        
        # Create random board states
        board_states = []
        for _ in range(num_positions):
            board = chess.Board()
            for _ in range(np.random.randint(0, 30)):
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    board.push(np.random.choice(legal_moves))
            
            # Convert to model input
            board_state = self._board_to_input(board)
            board_states.append(board_state)
        
        # Warmup
        print("  Warming up...")
        for _ in range(10):
            _ = model.evaluate_fast(board_states[0])
        
        # Benchmark
        print("  Benchmarking...")
        start = time.time()
        
        evaluations = []
        steps_used = []
        
        for board_state in board_states:
            value, metadata = model.evaluate_fast(board_state)
            evaluations.append(value)
            steps_used.append(metadata['reasoning_steps'])
        
        elapsed = time.time() - start
        evals_per_second = num_positions / elapsed
        
        self.results['lrt_evaluation'] = {
            'positions': num_positions,
            'time_s': elapsed,
            'evals_per_second': evals_per_second,
            'avg_steps': np.mean(steps_used),
            'avg_time_per_eval': elapsed / num_positions * 1000  # ms
        }
        
        print(f"  Evaluated {num_positions} positions in {elapsed:.2f}s")
        print(f"  Speed: {evals_per_second:.1f} positions/second")
        print(f"  Average steps: {np.mean(steps_used):.1f}")
        print(f"  Average time: {elapsed / num_positions * 1000:.1f}ms per position")
    
    def benchmark_search(self, depth=10, num_positions=10):
        """Benchmark search speed"""
        print("\nBenchmarking search...")
        
        # Initialize engine
        engine = chess.engine.SimpleEngine.popen_uci("./liquid_chess")
        
        # Create test positions
        positions = []
        board = chess.Board()
        for _ in range(num_positions):
            for _ in range(np.random.randint(10, 30)):
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    board.push(np.random.choice(legal_moves))
            positions.append(board.copy())
        
        # Benchmark
        nodes_total = 0
        time_total = 0
        
        for i, board in enumerate(positions):
            print(f"  Position {i+1}/{num_positions}...")
            
            start = time.time()
            result = engine.play(
                board,
                chess.engine.Limit(depth=depth),
                info=chess.engine.INFO_ALL
            )
            elapsed = time.time() - start
            
            # Get node count from info
            if hasattr(result, 'info') and 'nodes' in result.info:
                nodes = result.info['nodes']
            else:
                nodes = 0
            
            nodes_total += nodes
            time_total += elapsed
            
            print(f"    Nodes: {nodes:,}, Time: {elapsed:.2f}s")
        
        engine.quit()
        
        nodes_per_second = nodes_total / time_total
        
        self.results['search'] = {
            'depth': depth,
            'positions': num_positions,
            'total_nodes': nodes_total,
            'total_time_s': time_total,
            'nodes_per_second': nodes_per_second,
            'avg_time_per_position': time_total / num_positions
        }
        
        print(f"\n  Total nodes: {nodes_total:,}")
        print(f"  Total time: {time_total:.2f}s")
        print(f"  Speed: {nodes_per_second:,.0f} nodes/second")
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage"""
        print("\nBenchmarking memory usage...")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure baseline
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Initialize components
        print("  Initializing engine components...")
        
        model = UltraFastLRT()
        bridge = JAXLRTBridge()
        
        # Memory after initialization
        after_init = process.memory_info().rss / 1024 / 1024
        
        # Memory after processing
        board = chess.Board()
        board_state = self._board_to_input(board)
        
        for _ in range(100):
            _ = model.evaluate_fast(board_state)
        
        after_processing = process.memory_info().rss / 1024 / 1024
        
        self.results['memory'] = {
            'baseline_mb': baseline_memory,
            'after_init_mb': after_init,
            'after_processing_mb': after_processing,
            'engine_memory_mb': after_init - baseline_memory,
            'cache_memory_mb': after_processing - after_init
        }
        
        print(f"  Baseline: {baseline_memory:.1f} MB")
        print(f"  After init: {after_init:.1f} MB")
        print(f"  After processing: {after_processing:.1f} MB")
        print(f"  Engine memory: {after_init - baseline_memory:.1f} MB")
        print(f"  Cache memory: {after_processing - after_init:.1f} MB")
    
    def _board_to_input(self, board):
        """Convert chess board to model input"""
        # Simplified conversion
        return {
            'pieces': np.zeros((8, 8), dtype=np.int8),
            'turn': np.array(board.turn, dtype=np.bool_),
            'castling': np.array([True, True, True, True], dtype=np.bool_),
            'ep_square': np.array(-1, dtype=np.int8)
        }
    
    def run_all(self):
        """Run all benchmarks"""
        print("=" * 60)
        print("LIQUID CHESS ENGINE - PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        self.benchmark_move_generation()
        self.benchmark_lrt_evaluation()
        self.benchmark_search(depth=8, num_positions=5)
        self.benchmark_memory_usage()
        
        self.print_summary()
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        for test_name, result in self.results.items():
            print(f"\n{test_name.upper().replace('_', ' ')}:")
            for key, value in result.items():
                if isinstance(value, float):
                    if key.endswith('_second'):
                        print(f"  {key}: {value:,.0f}")
                    elif key.endswith('_mb'):
                        print(f"  {key}: {value:.1f}")
                    else:
                        print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
        
        # Compare with Stockfish
        print("\n" + "=" * 60)
        print("COMPARISON WITH STOCKFISH (ESTIMATED)")
        print("=" * 60)
        
        # Stockfish reference values (approximate)
        stockfish_ref = {
            'move_generation': 100_000_000,  # moves/second
            'nodes_per_second': 10_000_000,  # nodes/second at depth 8
            'memory_usage': 200,  # MB typical
        }
        
        our_values = {
            'move_generation': self.results['move_generation']['moves_per_second'],
            'nodes_per_second': self.results['search']['nodes_per_second'],
            'memory_usage': self.results['memory']['engine_memory_mb']
        }
        
        for metric, our_value in our_values.items():
            sf_value = stockfish_ref.get(metric, 0)
            if sf_value > 0:
                percentage = (our_value / sf_value) * 100
                print(f"{metric}: {our_value:,.0f} ({percentage:.1f}% of Stockfish)")

if __name__ == '__main__':
    benchmark = PerformanceBenchmark()
    benchmark.run_all()