#pragma once
#include "liquid_chess/uci.hpp"
#include <map>
#include <vector>
#include <memory>
#include <cmath>

namespace LiquidChess {

struct MCTSNode {
    Position pos;
    MCTSNode* parent;
    Move move;
    std::map<Move, std::unique_ptr<MCTSNode>> children;
    
    int visits = 0;
    float value_sum = 0.0f;
    float prior = 0.0f;
    bool expanded = false;

    MCTSNode(const Position& p, MCTSNode* prnt = nullptr, Move m = MOVE_NONE, float pr = 0.0f)
        : pos(p), parent(prnt), move(m), prior(pr) {}

    float value() const {
        return visits == 0 ? 0.0f : value_sum / visits;
    }

    float ucb_score(float c_puct) const {
        if (!parent) return 0.0f;
        float exploration = c_puct * prior * std::sqrt(parent->visits) / (1 + visits);
        float exploitation = (visits > 0) ? -value() : 0.0f; // Negamax
        return exploitation + exploration;
    }
};

class MCTS {
public:
    MCTS(int simulations = 800) : num_simulations(simulations) {}

    Move search(const Position& pos) {
        root = std::make_unique<MCTSNode>(pos);
        
        for (int i = 0; i < num_simulations; ++i) {
            simulate();
        }

        Move best_move = MOVE_NONE;
        int max_visits = -1;
        for (auto& [move, child] : root->children) {
            if (child->visits > max_visits) {
                max_visits = child->visits;
                best_move = move;
            }
        }
        return best_move;
    }

private:
    int num_simulations;
    std::unique_ptr<MCTSNode> root;

    void simulate() {
        MCTSNode* current = root.get();
        std::vector<MCTSNode*> path = {current};

        // Selection
        while (current->expanded && !current->children.empty()) {
            current = select_child(current);
            path.push_back(current);
        }

        // Expansion
        if (!current->expanded) {
            expand(current);
        }

        // Evaluation (Random for now, until JAX bridge is active)
        float v = evaluate(current->pos);

        // Backprop
        for (int i = path.size() - 1; i >= 0; --i) {
            float sign = (i % 2 == 0) ? 1.0f : -1.0f;
            path[i]->visits++;
            path[i]->value_sum += v * sign;
        }
    }

    void expand(MCTSNode* node) {
        auto moves = node->pos.legal_moves();
        if (moves.empty()) return;

        float p = 1.0f / moves.size();
        for (auto m : moves) {
            Position next_pos = node->pos;
            next_pos.do_move(m);
            node->children[m] = std::make_unique<MCTSNode>(next_pos, node, m, p);
        }
        node->expanded = true;
    }

    MCTSNode* select_child(MCTSNode* node) {
        float best_score = -1e9f;
        MCTSNode* best_child = nullptr;
        for (auto& [move, child] : node->children) {
            float score = child->ucb_score(1.5f);
            if (score > best_score) {
                best_score = score;
                best_child = child.get();
            }
        }
        return best_child;
    }

    float evaluate(const Position& pos) {
        // Simple Material + Position Evaluation (Fast for MCTS)
        int score = 0;
        
        // Piece values
        const int pawn_val = 100, knight_val = 320, bishop_val = 330, rook_val = 500, queen_val = 900;
        
        // Piece-Square Tables (Simplified)
        static const int pawn_psqt[64] = {
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
             5,  5, 10, 25, 25, 10,  5,  5,
             0,  0,  0, 20, 20,  0,  0,  0,
             5, -5,-10,  0,  0,-10, -5,  5,
             5, 10, 10,-20,-20, 10, 10,  5,
             0,  0,  0,  0,  0,  0,  0,  0
        };

        for (int sq = 0; sq < 64; ++sq) {
            Piece p = pos.piece_at(sq);
            if (p == NO_PIECE) continue;
            
            int val = 0;
            switch (type_of(p)) {
                case PAWN: val = pawn_val + pawn_psqt[color_of(p) == WHITE ? sq : 63 - sq]; break;
                case KNIGHT: val = knight_val; break;
                case BISHOP: val = bishop_val; break;
                case ROOK: val = rook_val; break;
                case QUEEN: val = queen_val; break;
                default: break;
            }
            score += (color_of(p) == WHITE) ? val : -val;
        }

        // Return normalized value between -1.0 and 1.0 (relative to side to move)
        float v = (float)score / 2000.0f;
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        return (pos.side_to_move() == WHITE) ? v : -v;
    }
};

}
