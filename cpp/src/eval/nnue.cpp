#include "liquid_chess/evaluation.hpp"
#include <immintrin.h>

namespace LiquidChess {

class NNUEAccumulator {
private:
    // Network architecture: 768->256->32->1
    alignas(64) int16_t weights1[768 * 256];
    alignas(64) int16_t weights2[256 * 32];
    alignas(64) int16_t weights3[32];
    alignas(64) int16_t biases1[256];
    alignas(64) int16_t biases2[32];
    int32_t bias3;
    
    // Accumulator for incremental updates
    struct Accumulator {
        alignas(64) int16_t values[256];
        uint64_t dirty; // Bitmask of dirty features
    };
    
    Accumulator accumulators[COLOR_NB];
    
public:
    NNUEAccumulator() {
        load_network("data/nnue.bin");
        reset();
    }
    
    // Incremental update after a move
    void update(const Position& pos, Move move) {
        Square from = from_sq(move);
        Square to = to_sq(move);
        Piece pc = pos.piece_on(from);
        
        // Remove old piece features
        if (pc != NO_PIECE) {
            int idx = feature_index(pc, from);
            subtract_feature(pos.side_to_move(), idx);
        }
        
        // Add new piece features
        if (type_of(move) == PROMOTION) {
            Piece promo = make_piece(color_of(pc), promotion_type(move));
            int idx = feature_index(promo, to);
            add_feature(pos.side_to_move(), idx);
        } else if (type_of(move) != CASTLING) {
            int idx = feature_index(pc, to);
            add_feature(pos.side_to_move(), idx);
        }
        
        // Handle captures
        if (pos.capture(move)) {
            Piece captured = pos.captured_piece(move);
            int idx = feature_index(captured, to);
            subtract_feature(~pos.side_to_move(), idx);
        }
        
        // Handle castling
        if (type_of(move) == CASTLING) {
            // Update rook position
            Square rookFrom, rookTo;
            get_castling_squares(pos, move, rookFrom, rookTo);
            
            Piece rook = make_piece(color_of(pc), ROOK);
            subtract_feature(pos.side_to_move(), feature_index(rook, rookFrom));
            add_feature(pos.side_to_move(), feature_index(rook, rookTo));
        }
        
        // Update king position accumulator
        if (type_of(pc) == KING) {
            // King move requires full refresh of accumulator
            refresh(pos.side_to_move(), pos);
        }
    }
    
    // Fast evaluation using SIMD
    int evaluate(Color side) const {
        const Accumulator& acc = accumulators[side];
        const Accumulator& opp = accumulators[~side];
        
        // Clipped ReLU activation
        alignas(64) int16_t activated[256];
        
        // Process with AVX2
        for (int i = 0; i < 256; i += 16) {
            __m256i val1 = _mm256_load_si256((__m256i*)&acc.values[i]);
            __m256i val2 = _mm256_load_si256((__m256i*)&opp.values[i]);
            
            // Average of both perspectives
            __m256i avg = _mm256_avg_epi16(val1, val2);
            
            // Clipped ReLU: max(0, min(127, x))
            __m256i zero = _mm256_setzero_si256();
            __m256i max127 = _mm256_set1_epi16(127);
            avg = _mm256_max_epi16(avg, zero);
            avg = _mm256_min_epi16(avg, max127);
            
            _mm256_store_si256((__m256i*)&activated[i], avg);
        }
        
        // First layer: 256->32
        alignas(64) int32_t layer2[32] = {0};
        
        for (int i = 0; i < 256; ++i) {
            if (activated[i]) {
                // Sparse accumulation
                __m256i weight = _mm256_load_si256(
                    (__m256i*)&weights1[i * 32]);
                __m256i activation = _mm256_set1_epi16(activated[i]);
                
                // Multiply and accumulate
                __m256i prod = _mm256_madd_epi16(weight, activation);
                __m256i sum = _mm256_load_si256((__m256i*)&layer2[0]);
                sum = _mm256_add_epi32(sum, prod);
                _mm256_store_si256((__m256i*)&layer2[0], sum);
            }
        }
        
        // Add bias and clip
        for (int i = 0; i < 32; ++i) {
            layer2[i] += biases1[i];
            layer2[i] = std::max(0, std::min(127, layer2[i] >> 8));
        }
        
        // Output layer: 32->1
        int32_t output = bias3;
        for (int i = 0; i < 32; ++i) {
            output += layer2[i] * weights3[i];
        }
        
        return output / (16 * 387);
    }
    
private:
    void add_feature(Color side, int idx) {
        if (idx < 0) return;
        
        const int16_t* weight = &weights1[idx * 256];
        int16_t* acc = accumulators[side].values;
        
        // AVX2-accelerated addition
        for (int i = 0; i < 256; i += 16) {
            __m256i acc_vec = _mm256_load_si256((__m256i*)&acc[i]);
            __m256i weight_vec = _mm256_load_si256((__m256i*)&weight[i]);
            __m256i sum = _mm256_add_epi16(acc_vec, weight_vec);
            _mm256_store_si256((__m256i*)&acc[i], sum);
        }
        
        accumulators[side].dirty |= 1ULL << (idx % 64);
    }
    
    void subtract_feature(Color side, int idx) {
        if (idx < 0) return;
        
        const int16_t* weight = &weights1[idx * 256];
        int16_t* acc = accumulators[side].values;
        
        // AVX2-accelerated subtraction
        for (int i = 0; i < 256; i += 16) {
            __m256i acc_vec = _mm256_load_si256((__m256i*)&acc[i]);
            __m256i weight_vec = _mm256_load_si256((__m256i*)&weight[i]);
            __m256i diff = _mm256_sub_epi16(acc_vec, weight_vec);
            _mm256_store_si256((__m256i*)&acc[i], diff);
        }
        
        accumulators[side].dirty |= 1ULL << (idx % 64);
    }
};

}