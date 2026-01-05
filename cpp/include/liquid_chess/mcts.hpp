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
        // TODO: Call JAX model here
        return 0.0f; 
    }
};

}
