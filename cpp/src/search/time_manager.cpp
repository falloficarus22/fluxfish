#include "liquid_chess/search.hpp"
#include <algorithm>
#include <cmath>

namespace LiquidChess {

class AdaptiveTimeManager {
private:
    struct TimeControl {
        int time;           // Remaining time
        int inc;            // Increment per move
        int movesToGo;      // Moves to next time control
        int moveOverhead;   // Network/UI overhead
    };
    
    TimeControl tc[COLOR_NB];
    Color us;
    int64_t startTime;
    int64_t optimumTime;
    int64_t maximumTime;
    
    // LRT complexity tracking
    float lrtComplexity;
    float gamePhaseComplexity[100]; // Last 100 moves
    int complexityIndex;
    
public:
    AdaptiveTimeManager(const SearchLimits& limits, Color us) : us(us) {
        tc[WHITE] = {limits.wtime, limits.winc, limits.movestogo, 10};
        tc[BLACK] = {limits.btime, limits.binc, limits.movestogo, 10};
        
        startTime = now();
        lrtComplexity = 0.5f; // Default medium complexity
        
        calculate_times();
    }
    
    void update_complexity(float complexity, Move move) {
        lrtComplexity = complexity;
        gamePhaseComplexity[complexityIndex % 100] = complexity;
        complexityIndex++;
        
        // Recalculate times based on new complexity
        calculate_times();
    }
    
    bool timeout() const {
        return now() - startTime >= maximumTime;
    }
    
    bool optimum_reached() const {
        return now() - startTime >= optimumTime;
    }
    
    int64_t elapsed() const {
        return now() - startTime;
    }
    
private:
    void calculate_times() {
        int myTime = tc[us].time;
        int myInc = tc[us].inc;
        int movesToGo = tc[us].movesToGo;
        
        if (movesToGo == 0) {
            // Sudden death
            movesToGo = estimate_moves_to_go();
        }
        
        // Base time calculation (similar to Stockfish)
        int64_t baseTime = myTime / movesToGo + myInc;
        
        // Adjust based on LRT complexity
        // More complex positions get more time
        float complexityFactor = 0.5f + lrtComplexity;
        
        // Adjust for game phase
        float gamePhaseFactor = 1.0f;
        if (complexityIndex > 10) {
            float avgComplexity = 0.0f;
            int count = std::min(10, complexityIndex);
            for (int i = 0; i < count; ++i) {
                avgComplexity += gamePhaseComplexity[(complexityIndex - i - 1) % 100];
            }
            avgComplexity /= count;
            
            // If recent positions were complex, allocate more time
            if (avgComplexity > 0.7f) {
                gamePhaseFactor = 1.2f;
            }
        }
        
        // Calculate optimum and maximum times
        optimumTime = baseTime * complexityFactor * gamePhaseFactor * 0.7;
        maximumTime = std::min(optimumTime * 2, int64_t(myTime * 0.95));
        
        // Ensure minimum thinking time
        optimumTime = std::max(optimumTime, int64_t(50));
        maximumTime = std::max(maximumTime, optimumTime + 100);
        
        // Don't exceed remaining time
        optimumTime = std::min(optimumTime, int64_t(myTime - tc[us].moveOverhead));
        maximumTime = std::min(maximumTime, int64_t(myTime - tc[us].moveOverhead));
    }
    
    int estimate_moves_to_go() const {
        // Estimate based on remaining time and game phase
        int remainingTime = tc[us].time;
        
        if (remainingTime < 30000) { // Less than 30 seconds
            return 20 + remainingTime / 1000; // More moves in blitz
        } else if (remainingTime < 60000) { // Less than 1 minute
            return 30;
        } else if (remainingTime < 180000) { // Less than 3 minutes
            return 40;
        } else {
            return 50;
        }
    }
    
    static int64_t now() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
};

}