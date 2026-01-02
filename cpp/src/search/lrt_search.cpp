#pragma once
#include "bitboard.hpp"
#include "evaluation.hpp"
#include <memory>
#include <vector>

namespace LiquidChess {

class LRTManager; // Forward declaration

struct SearchStack {
    Move* pv;
    int ply;
    Move currentMove;
    Move excludedMove;
    Move killers[2];
    int staticEval;
    int moveCount;
    bool skipEarlyPruning;
    bool inCheck;
    int lrtComplexity;
};

class LRTGuidedSearch {
private:
    Position rootPos;
    SearchLimits limits;
    TranspositionTable& tt;
    LRTManager& lrt;
    TimeManager& tm;
    
    // Search statistics
    uint64_t nodes;
    uint64_t tbHits;
    int selDepth;
    std::vector<Move> rootPV;
    
    // Adaptive search parameters
    int lmrDepth;
    int lmrMoveCount;
    int futilityMargin[16];
    int razorMargin[4];
    
    // Thread synchronization
    std::mutex ioMutex;
    std::atomic<bool> stop;
    
public:
    LRTGuidedSearch(Position& pos, TranspositionTable& tt, 
                   LRTManager& lrt, TimeManager& tm)
        : rootPos(pos), tt(tt), lrt(lrt), tm(tm), nodes(0), stop(false) {
        init_search_tables();
    }
    
    // Main search function with LRT integration
    template<NodeType nodeType>
    int search(int depth, int alpha, int beta, 
               SearchStack* ss, bool cutNode);
    
    // LRT-enhanced null move pruning
    bool lrt_null_move_pruning(int depth, int beta, 
                              SearchStack* ss, int eval);
    
    // Adaptive forward pruning based on LRT complexity
    bool forward_pruning_allowed(int depth, int moveCount, 
                                SearchStack* ss, bool improving);
    
    // Multi-variant search for unclear positions
    void multi_variant_search(int depth, std::vector<Move>& moves,
                             std::vector<int>& scores);
};

// LRT complexity estimation for search tuning
class LRTManager {
private:
    struct LRTState {
        float complexity;      // 0-1, higher = more complex
        int recommendedDepth;  // LRT-suggested search depth
        float confidence;      // How sure LRT is
        int reasoningSteps;    // Steps LRT actually took
    };
    
    std::unordered_map<uint64_t, LRTState> cache;
    py::object pythonLRT;      // Python LRT model
    std::mutex cacheMutex;
    
public:
    LRTManager(const std::string& modelPath);
    
    // Get LRT evaluation with complexity estimate
    std::pair<float, LRTState> evaluate_position(const Position& pos);
    
    // Should we use LRT for this position?
    bool should_use_lrt(const Position& pos, int depth) const;
    
    // Get adaptive search parameters
    SearchParams get_search_params(const Position& pos, int depth);
    
    // Clear cache
    void clear();
    
private:
    LRTState estimate_complexity_fast(const Position& pos);
};

}