#include "liquid_chess/uci.hpp"

#include <iostream>
#include <sstream>
#include <string>

namespace LiquidChess {

class UCIHandler {
public:
    UCIHandler() = default;

    void loop() {
        std::string line;
        std::string token;

        while (std::getline(std::cin, line)) {
            std::istringstream iss(line);
            iss >> token;

            if (token == "uci") {
                identify();
            } else if (token == "isready") {
                std::cout << "readyok" << std::endl;
            } else if (token == "ucinewgame") {
                new_game();
            } else if (token == "position") {
                position(iss);
            } else if (token == "go") {
                go(iss);
            } else if (token == "perft") {
                perft(iss);
            } else if (token == "dumplegal") {
                dumplegal();
            } else if (token == "quit") {
                break;
            }
        }
    }

private:
    Position pos;

    void identify() {
        std::cout << "id name LiquidChess" << std::endl;
        std::cout << "id author fluxfish" << std::endl;
        std::cout << "uciok" << std::endl;
    }

    void new_game() {
        pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    void position(std::istringstream& iss) {
        std::string token;
        iss >> token;

        if (token == "startpos") {
            pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            iss >> token;
        } else if (token == "fen") {
            std::string fen;
            while (iss >> token && token != "moves") {
                fen += token;
                fen += ' ';
            }
            pos.set(fen);
        }

        while (iss >> token) {
            Move m = move_from_string(pos, token);
            if (!(m == MOVE_NONE) && pos.is_legal(m)) {
                pos.do_move(m);
            }
        }
    }

    void go(std::istringstream& iss) {
        int depth = 1;
        std::string token;
        while (iss >> token) {
            if (token == "depth") iss >> depth;
        }

        (void)depth;

        auto moves = pos.legal_moves();
        if (moves.empty()) {
            std::cout << "bestmove 0000" << std::endl;
            return;
        }
        std::cout << "bestmove " << move_to_string(moves[0]) << std::endl;
    }

    void perft(std::istringstream& iss) {
        int depth = 1;
        iss >> depth;
        uint64_t nodes = pos.perft(depth);
        std::cout << "perft " << depth << ": " << nodes << std::endl;
    }

    void dumplegal() {
        auto moves = pos.legal_moves();
        for (const auto& m : moves) {
            std::cout << move_to_string(m) << std::endl;
        }
    }
};

void uci_loop() {
    UCIHandler uci;
    uci.loop();
}

}