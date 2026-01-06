#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace LiquidChess {

using Square = int;

enum Color : uint8_t {
    WHITE = 0,
    BLACK = 1,
};

inline constexpr Color operator~(Color c) { return c == WHITE ? BLACK : WHITE; }

enum PieceType : uint8_t {
    PAWN = 1,
    KNIGHT = 2,
    BISHOP = 3,
    ROOK = 4,
    QUEEN = 5,
    KING = 6,
};

enum Piece : uint8_t {
    NO_PIECE = 0,

    W_PAWN = 1,
    W_KNIGHT = 2,
    W_BISHOP = 3,
    W_ROOK = 4,
    W_QUEEN = 5,
    W_KING = 6,

    B_PAWN = 7,
    B_KNIGHT = 8,
    B_BISHOP = 9,
    B_ROOK = 10,
    B_QUEEN = 11,
    B_KING = 12,
};

inline constexpr Color color_of(Piece p) { return p >= B_PAWN ? BLACK : WHITE; }
inline constexpr PieceType type_of(Piece p) {
    if (p == NO_PIECE) return static_cast<PieceType>(0);
    uint8_t v = static_cast<uint8_t>(p);
    return static_cast<PieceType>(v <= 6 ? v : v - 6);
}

struct Move {
    uint16_t from;
    uint16_t to;
    uint8_t promo;
    uint8_t flags;
};

inline constexpr bool operator==(const Move& a, const Move& b) {
    return a.from == b.from && a.to == b.to && a.promo == b.promo && a.flags == b.flags;
}

inline bool operator<(const Move& a, const Move& b) {
    if (a.from != b.from) return a.from < b.from;
    if (a.to != b.to) return a.to < b.to;
    if (a.promo != b.promo) return a.promo < b.promo;
    return a.flags < b.flags;
}

inline constexpr Move MOVE_NONE{0, 0, 0, 0};

struct Undo {
    Piece captured;
    uint8_t castling;
    int8_t ep_square;
    uint16_t halfmove_clock;
    Move move;
};

class Position {
public:
    Position();

    void set(const std::string& fen);

    Color side_to_move() const;
    Piece piece_at(Square sq) const { return board_[sq]; }

    std::vector<Move> legal_moves() const;

    uint64_t perft(int depth) const;

    void do_move(const Move& m);
    void undo_move();

    bool is_legal(const Move& m) const;

private:
    std::array<Piece, 64> board_;
    Color stm_;
    uint8_t castling_rights_;
    int8_t ep_square_;
    uint16_t halfmove_clock_;
    uint16_t fullmove_number_;
    std::vector<Undo> history_;

    Square king_square(Color c) const;
    bool square_attacked(Square sq, Color by) const;
    bool in_check(Color c) const;

    std::vector<Move> pseudo_moves() const;

    void add_pawn_moves(std::vector<Move>& out) const;
    void add_knight_moves(std::vector<Move>& out) const;
    void add_slider_moves(std::vector<Move>& out, PieceType pt) const;
    void add_king_moves(std::vector<Move>& out) const;

    void make_move_internal(const Move& m, Undo& u);
};

Move move_from_string(const Position& pos, const std::string& uci);
std::string move_to_string(const Move& m);

void uci_loop();

}
