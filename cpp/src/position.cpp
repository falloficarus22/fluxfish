#include "liquid_chess/uci.hpp"

#include <array>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace LiquidChess {

static inline int file_of(Square s) { return s & 7; }
static inline int rank_of(Square s) { return s >> 3; }
static inline Square make_square(int file, int rank) { return rank * 8 + file; }

static inline bool on_board(int file, int rank) { return file >= 0 && file < 8 && rank >= 0 && rank < 8; }

static inline bool same_color(Piece a, Piece b) {
    if (a == NO_PIECE || b == NO_PIECE) return false;
    return color_of(a) == color_of(b);
}

static inline bool is_enemy(Piece a, Piece b) {
    if (a == NO_PIECE || b == NO_PIECE) return false;
    return color_of(a) != color_of(b);
}

static inline Piece make_piece(Color c, PieceType pt) {
    if (pt == PAWN) return c == WHITE ? W_PAWN : B_PAWN;
    if (pt == KNIGHT) return c == WHITE ? W_KNIGHT : B_KNIGHT;
    if (pt == BISHOP) return c == WHITE ? W_BISHOP : B_BISHOP;
    if (pt == ROOK) return c == WHITE ? W_ROOK : B_ROOK;
    if (pt == QUEEN) return c == WHITE ? W_QUEEN : B_QUEEN;
    if (pt == KING) return c == WHITE ? W_KING : B_KING;
    return NO_PIECE;
}

static inline bool is_slider(Piece p, PieceType pt) {
    if (p == NO_PIECE) return false;
    PieceType t = type_of(p);
    if (pt == BISHOP) return t == BISHOP || t == QUEEN;
    if (pt == ROOK) return t == ROOK || t == QUEEN;
    return false;
}

static inline PieceType promo_from_char(char c) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (c == 'q') return QUEEN;
    if (c == 'r') return ROOK;
    if (c == 'b') return BISHOP;
    if (c == 'n') return KNIGHT;
    return static_cast<PieceType>(0);
}

static inline char promo_to_char(PieceType pt) {
    if (pt == QUEEN) return 'q';
    if (pt == ROOK) return 'r';
    if (pt == BISHOP) return 'b';
    if (pt == KNIGHT) return 'n';
    return '\0';
}

static inline int square_from_uci(const std::string& s, size_t off) {
    if (off + 1 >= s.size()) return -1;
    char f = s[off];
    char r = s[off + 1];
    if (f < 'a' || f > 'h') return -1;
    if (r < '1' || r > '8') return -1;
    int file = f - 'a';
    int rank = r - '1';
    return make_square(file, rank);
}

Position::Position() {
    for (auto& p : board_) p = NO_PIECE;
    stm_ = WHITE;
    castling_rights_ = 0;
    ep_square_ = -1;
    halfmove_clock_ = 0;
    fullmove_number_ = 1;
}

Color Position::side_to_move() const { return stm_; }

Square Position::king_square(Color c) const {
    Piece k = c == WHITE ? W_KING : B_KING;
    for (int i = 0; i < 64; ++i) {
        if (board_[i] == k) return i;
    }
    return -1;
}

bool Position::square_attacked(Square sq, Color by) const {
    int f = file_of(sq);
    int r = rank_of(sq);

    if (by == WHITE) {
        if (r > 0) {
            if (f > 0) {
                Square s = make_square(f - 1, r - 1);
                if (board_[s] == W_PAWN) return true;
            }
            if (f < 7) {
                Square s = make_square(f + 1, r - 1);
                if (board_[s] == W_PAWN) return true;
            }
        }
    } else {
        if (r < 7) {
            if (f > 0) {
                Square s = make_square(f - 1, r + 1);
                if (board_[s] == B_PAWN) return true;
            }
            if (f < 7) {
                Square s = make_square(f + 1, r + 1);
                if (board_[s] == B_PAWN) return true;
            }
        }
    }

    static constexpr std::array<std::pair<int, int>, 8> kKnight = {{{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}}};
    Piece attackerN = by == WHITE ? W_KNIGHT : B_KNIGHT;
    for (auto [df, dr] : kKnight) {
        int nf = f + df;
        int nr = r + dr;
        if (on_board(nf, nr)) {
            if (board_[make_square(nf, nr)] == attackerN) return true;
        }
    }

    static constexpr std::array<std::pair<int, int>, 8> kKing = {{{1, 1}, {1, 0}, {1, -1}, {0, 1}, {0, -1}, {-1, 1}, {-1, 0}, {-1, -1}}};
    Piece attackerK = by == WHITE ? W_KING : B_KING;
    for (auto [df, dr] : kKing) {
        int nf = f + df;
        int nr = r + dr;
        if (on_board(nf, nr)) {
            if (board_[make_square(nf, nr)] == attackerK) return true;
        }
    }

    auto ray_attacked = [&](int df, int dr, PieceType slider) {
        int nf = f + df;
        int nr = r + dr;
        while (on_board(nf, nr)) {
            Square ns = make_square(nf, nr);
            Piece p = board_[ns];
            if (p != NO_PIECE) {
                if (color_of(p) == by && is_slider(p, slider)) return true;
                return false;
            }
            nf += df;
            nr += dr;
        }
        return false;
    };

    if (ray_attacked(1, 1, BISHOP)) return true;
    if (ray_attacked(1, -1, BISHOP)) return true;
    if (ray_attacked(-1, 1, BISHOP)) return true;
    if (ray_attacked(-1, -1, BISHOP)) return true;

    if (ray_attacked(1, 0, ROOK)) return true;
    if (ray_attacked(-1, 0, ROOK)) return true;
    if (ray_attacked(0, 1, ROOK)) return true;
    if (ray_attacked(0, -1, ROOK)) return true;

    return false;
}

bool Position::in_check(Color c) const {
    Square ksq = king_square(c);
    if (ksq < 0) return false;
    return square_attacked(ksq, ~c);
}

void Position::set(const std::string& fen) {
    for (auto& p : board_) p = NO_PIECE;
    castling_rights_ = 0;
    ep_square_ = -1;
    halfmove_clock_ = 0;
    fullmove_number_ = 1;
    history_.clear();

    std::istringstream iss(fen);
    std::string boardPart;
    std::string stmPart;
    std::string castlingPart;
    std::string epPart;
    std::string halfmovePart;
    std::string fullmovePart;

    iss >> boardPart >> stmPart >> castlingPart >> epPart >> halfmovePart >> fullmovePart;
    if (boardPart.empty() || stmPart.empty()) {
        throw std::runtime_error("invalid fen");
    }

    int rank = 7;
    int file = 0;
    for (char c : boardPart) {
        if (c == '/') {
            rank--;
            file = 0;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            file += c - '0';
            continue;
        }
        if (!on_board(file, rank)) throw std::runtime_error("invalid fen");

        Piece pc = NO_PIECE;
        switch (c) {
            case 'P': pc = W_PAWN; break;
            case 'N': pc = W_KNIGHT; break;
            case 'B': pc = W_BISHOP; break;
            case 'R': pc = W_ROOK; break;
            case 'Q': pc = W_QUEEN; break;
            case 'K': pc = W_KING; break;
            case 'p': pc = B_PAWN; break;
            case 'n': pc = B_KNIGHT; break;
            case 'b': pc = B_BISHOP; break;
            case 'r': pc = B_ROOK; break;
            case 'q': pc = B_QUEEN; break;
            case 'k': pc = B_KING; break;
            default: throw std::runtime_error("invalid fen");
        }
        board_[make_square(file, rank)] = pc;
        file++;
    }

    stm_ = (stmPart == "w") ? WHITE : BLACK;

    if (!castlingPart.empty() && castlingPart != "-") {
        for (char c : castlingPart) {
            if (c == 'K') castling_rights_ |= 1;
            else if (c == 'Q') castling_rights_ |= 2;
            else if (c == 'k') castling_rights_ |= 4;
            else if (c == 'q') castling_rights_ |= 8;
        }
    }

    if (!epPart.empty() && epPart != "-") {
        int sq = square_from_uci(epPart, 0);
        ep_square_ = sq;
    }

    if (!halfmovePart.empty()) halfmove_clock_ = std::atoi(halfmovePart.c_str());
    if (!fullmovePart.empty()) fullmove_number_ = std::atoi(fullmovePart.c_str());
}

void Position::add_pawn_moves(std::vector<Move>& out) const {
    Color us = stm_;
    Color them = ~us;

    for (int sq = 0; sq < 64; ++sq) {
        Piece p = board_[sq];
        if (p == NO_PIECE) continue;
        if (color_of(p) != us) continue;
        if (type_of(p) != PAWN) continue;

        int f = file_of(sq);
        int r = rank_of(sq);

        int dir = (us == WHITE) ? 1 : -1;
        int startRank = (us == WHITE) ? 1 : 6;
        int promoRank = (us == WHITE) ? 6 : 1;
        int epRank = (us == WHITE) ? 4 : 3;

        int nr = r + dir;
        if (nr >= 0 && nr < 8) {
            Square to = make_square(f, nr);
            if (board_[to] == NO_PIECE) {
                if (r == promoRank) {
                    for (PieceType pt : {QUEEN, ROOK, BISHOP, KNIGHT}) {
                        out.push_back(Move{static_cast<uint16_t>(sq), static_cast<uint16_t>(to), static_cast<uint8_t>(pt), 0});
                    }
                } else {
                    out.push_back(Move{static_cast<uint16_t>(sq), static_cast<uint16_t>(to), 0, 0});
                    if (r == startRank) {
                        Square to2 = make_square(f, r + 2 * dir);
                        if (board_[to2] == NO_PIECE) {
                            out.push_back(Move{static_cast<uint16_t>(sq), static_cast<uint16_t>(to2), 0, 8});
                        }
                    }
                }
            }
        }

        for (int df : {-1, 1}) {
            int nf = f + df;
            int nr2 = r + dir;
            if (!on_board(nf, nr2)) continue;
            Square to = make_square(nf, nr2);
            Piece cap = board_[to];
            if (cap != NO_PIECE && color_of(cap) == them) {
                if (r == promoRank) {
                    for (PieceType pt : {QUEEN, ROOK, BISHOP, KNIGHT}) {
                        out.push_back(Move{static_cast<uint16_t>(sq), static_cast<uint16_t>(to), static_cast<uint8_t>(pt), 4});
                    }
                } else {
                    out.push_back(Move{static_cast<uint16_t>(sq), static_cast<uint16_t>(to), 0, 4});
                }
            }
        }

        if (r == epRank && ep_square_ >= 0) {
            int epf = file_of(ep_square_);
            int epr = rank_of(ep_square_);
            if (epr == r + dir && std::abs(epf - f) == 1) {
                out.push_back(Move{static_cast<uint16_t>(sq), static_cast<uint16_t>(ep_square_), 0, 1});
            }
        }
    }
}

void Position::add_knight_moves(std::vector<Move>& out) const {
    Color us = stm_;

    static constexpr std::array<std::pair<int, int>, 8> kKnight = {{{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}}};

    for (int sq = 0; sq < 64; ++sq) {
        Piece p = board_[sq];
        if (p == NO_PIECE || color_of(p) != us || type_of(p) != KNIGHT) continue;
        int f = file_of(sq);
        int r = rank_of(sq);
        for (auto [df, dr] : kKnight) {
            int nf = f + df;
            int nr = r + dr;
            if (!on_board(nf, nr)) continue;
            Square to = make_square(nf, nr);
            Piece dst = board_[to];
            if (dst == NO_PIECE || is_enemy(dst, p)) {
                uint8_t flags = (dst != NO_PIECE) ? 4 : 0;
                out.push_back(Move{static_cast<uint16_t>(sq), static_cast<uint16_t>(to), 0, flags});
            }
        }
    }
}

void Position::add_slider_moves(std::vector<Move>& out, PieceType pt) const {
    Color us = stm_;

    std::array<std::pair<int, int>, 4> dirs;
    if (pt == BISHOP) {
        dirs = {{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}};
    } else {
        dirs = {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
    }

    for (int sq = 0; sq < 64; ++sq) {
        Piece p = board_[sq];
        if (p == NO_PIECE || color_of(p) != us) continue;
        PieceType t = type_of(p);
        if (pt == BISHOP && !(t == BISHOP || t == QUEEN)) continue;
        if (pt == ROOK && !(t == ROOK || t == QUEEN)) continue;

        int f = file_of(sq);
        int r = rank_of(sq);

        for (auto [df, dr] : dirs) {
            int nf = f + df;
            int nr = r + dr;
            while (on_board(nf, nr)) {
                Square to = make_square(nf, nr);
                Piece dst = board_[to];
                if (dst == NO_PIECE) {
                    out.push_back(Move{static_cast<uint16_t>(sq), static_cast<uint16_t>(to), 0, 0});
                } else {
                    if (is_enemy(dst, p)) {
                        out.push_back(Move{static_cast<uint16_t>(sq), static_cast<uint16_t>(to), 0, 4});
                    }
                    break;
                }
                nf += df;
                nr += dr;
            }
        }
    }
}

void Position::add_king_moves(std::vector<Move>& out) const {
    Color us = stm_;
    Color them = ~us;

    static constexpr std::array<std::pair<int, int>, 8> kKing = {{{1, 1}, {1, 0}, {1, -1}, {0, 1}, {0, -1}, {-1, 1}, {-1, 0}, {-1, -1}}};

    for (int sq = 0; sq < 64; ++sq) {
        Piece p = board_[sq];
        if (p == NO_PIECE || color_of(p) != us || type_of(p) != KING) continue;

        int f = file_of(sq);
        int r = rank_of(sq);

        for (auto [df, dr] : kKing) {
            int nf = f + df;
            int nr = r + dr;
            if (!on_board(nf, nr)) continue;
            Square to = make_square(nf, nr);
            Piece dst = board_[to];
            if (dst == NO_PIECE || is_enemy(dst, p)) {
                uint8_t flags = (dst != NO_PIECE) ? 4 : 0;
                out.push_back(Move{static_cast<uint16_t>(sq), static_cast<uint16_t>(to), 0, flags});
            }
        }

        if (in_check(us)) continue;

        if (us == WHITE) {
            if ((castling_rights_ & 1) != 0) {
                if (board_[make_square(5, 0)] == NO_PIECE && board_[make_square(6, 0)] == NO_PIECE) {
                    if (!square_attacked(make_square(4, 0), them) && !square_attacked(make_square(5, 0), them) && !square_attacked(make_square(6, 0), them)) {
                        out.push_back(Move{static_cast<uint16_t>(sq), static_cast<uint16_t>(make_square(6, 0)), 0, 2});
                    }
                }
            }
            if ((castling_rights_ & 2) != 0) {
                if (board_[make_square(1, 0)] == NO_PIECE && board_[make_square(2, 0)] == NO_PIECE && board_[make_square(3, 0)] == NO_PIECE) {
                    if (!square_attacked(make_square(4, 0), them) && !square_attacked(make_square(3, 0), them) && !square_attacked(make_square(2, 0), them)) {
                        out.push_back(Move{static_cast<uint16_t>(sq), static_cast<uint16_t>(make_square(2, 0)), 0, 2});
                    }
                }
            }
        } else {
            if ((castling_rights_ & 4) != 0) {
                if (board_[make_square(5, 7)] == NO_PIECE && board_[make_square(6, 7)] == NO_PIECE) {
                    if (!square_attacked(make_square(4, 7), them) && !square_attacked(make_square(5, 7), them) && !square_attacked(make_square(6, 7), them)) {
                        out.push_back(Move{static_cast<uint16_t>(sq), static_cast<uint16_t>(make_square(6, 7)), 0, 2});
                    }
                }
            }
            if ((castling_rights_ & 8) != 0) {
                if (board_[make_square(1, 7)] == NO_PIECE && board_[make_square(2, 7)] == NO_PIECE && board_[make_square(3, 7)] == NO_PIECE) {
                    if (!square_attacked(make_square(4, 7), them) && !square_attacked(make_square(3, 7), them) && !square_attacked(make_square(2, 7), them)) {
                        out.push_back(Move{static_cast<uint16_t>(sq), static_cast<uint16_t>(make_square(2, 7)), 0, 2});
                    }
                }
            }
        }
    }
}

std::vector<Move> Position::pseudo_moves() const {
    std::vector<Move> out;
    out.reserve(64);
    add_pawn_moves(out);
    add_knight_moves(out);
    add_slider_moves(out, BISHOP);
    add_slider_moves(out, ROOK);
    add_king_moves(out);
    return out;
}

void Position::make_move_internal(const Move& m, Undo& u) {
    u.captured = NO_PIECE;
    u.castling = castling_rights_;
    u.ep_square = ep_square_;
    u.halfmove_clock = halfmove_clock_;
    u.move = m;

    Piece moving = board_[m.from];
    Piece captured = board_[m.to];

    ep_square_ = -1;

    if (moving == NO_PIECE) return;

    if (type_of(moving) == PAWN || captured != NO_PIECE || (m.flags & 1) != 0) {
        halfmove_clock_ = 0;
    } else {
        halfmove_clock_++;
    }

    if ((m.flags & 1) != 0) {
        int dir = (stm_ == WHITE) ? -1 : 1;
        Square capSq = static_cast<int>(m.to) + 8 * dir;
        captured = board_[capSq];
        board_[capSq] = NO_PIECE;
    }

    u.captured = captured;

    board_[m.to] = moving;
    board_[m.from] = NO_PIECE;

    if ((m.flags & 2) != 0 && type_of(moving) == KING) {
        if (stm_ == WHITE) {
            if (m.to == make_square(6, 0)) {
                board_[make_square(5, 0)] = board_[make_square(7, 0)];
                board_[make_square(7, 0)] = NO_PIECE;
            } else if (m.to == make_square(2, 0)) {
                board_[make_square(3, 0)] = board_[make_square(0, 0)];
                board_[make_square(0, 0)] = NO_PIECE;
            }
        } else {
            if (m.to == make_square(6, 7)) {
                board_[make_square(5, 7)] = board_[make_square(7, 7)];
                board_[make_square(7, 7)] = NO_PIECE;
            } else if (m.to == make_square(2, 7)) {
                board_[make_square(3, 7)] = board_[make_square(0, 7)];
                board_[make_square(0, 7)] = NO_PIECE;
            }
        }
    }

    if (type_of(moving) == PAWN && (m.flags & 8) != 0) {
        int dir = (stm_ == WHITE) ? 1 : -1;
        int fromRank = rank_of(m.from);
        int toRank = rank_of(m.to);
        if (std::abs(toRank - fromRank) == 2) {
            ep_square_ = make_square(file_of(m.from), fromRank + dir);
        }
    }

    if (type_of(moving) == KING) {
        if (stm_ == WHITE) castling_rights_ &= ~(1 | 2);
        else castling_rights_ &= ~(4 | 8);
    }

    if (type_of(moving) == ROOK) {
        if (stm_ == WHITE) {
            if (m.from == make_square(0, 0)) castling_rights_ &= ~2;
            if (m.from == make_square(7, 0)) castling_rights_ &= ~1;
        } else {
            if (m.from == make_square(0, 7)) castling_rights_ &= ~8;
            if (m.from == make_square(7, 7)) castling_rights_ &= ~4;
        }
    }

    if (captured != NO_PIECE && type_of(captured) == ROOK) {
        if (color_of(captured) == WHITE) {
            if (m.to == make_square(0, 0)) castling_rights_ &= ~2;
            if (m.to == make_square(7, 0)) castling_rights_ &= ~1;
        } else {
            if (m.to == make_square(0, 7)) castling_rights_ &= ~8;
            if (m.to == make_square(7, 7)) castling_rights_ &= ~4;
        }
    }

    if (type_of(moving) == PAWN) {
        int promoRank = (stm_ == WHITE) ? 7 : 0;
        if (rank_of(m.to) == promoRank && m.promo != 0) {
            board_[m.to] = make_piece(stm_, static_cast<PieceType>(m.promo));
        }
    }
}

void Position::do_move(const Move& m) {
    Undo u;
    make_move_internal(m, u);
    history_.push_back(u);

    stm_ = ~stm_;
    if (stm_ == WHITE) fullmove_number_++;
}

void Position::undo_move() {
    if (history_.empty()) return;

    Undo u = history_.back();
    history_.pop_back();

    stm_ = ~stm_;
    if (stm_ == BLACK) fullmove_number_--;

    castling_rights_ = u.castling;
    ep_square_ = u.ep_square;
    halfmove_clock_ = u.halfmove_clock;

    Move m = u.move;
    Piece moving = board_[m.to];

    if ((m.flags & 2) != 0 && type_of(moving) == KING) {
        if (stm_ == WHITE) {
            if (m.to == make_square(6, 0)) {
                board_[make_square(7, 0)] = board_[make_square(5, 0)];
                board_[make_square(5, 0)] = NO_PIECE;
            } else if (m.to == make_square(2, 0)) {
                board_[make_square(0, 0)] = board_[make_square(3, 0)];
                board_[make_square(3, 0)] = NO_PIECE;
            }
        } else {
            if (m.to == make_square(6, 7)) {
                board_[make_square(7, 7)] = board_[make_square(5, 7)];
                board_[make_square(5, 7)] = NO_PIECE;
            } else if (m.to == make_square(2, 7)) {
                board_[make_square(0, 7)] = board_[make_square(3, 7)];
                board_[make_square(3, 7)] = NO_PIECE;
            }
        }
    }

    Piece restored = moving;
    if (m.promo != 0 && type_of(restored) != PAWN) {
        restored = stm_ == WHITE ? W_PAWN : B_PAWN;
    }

    board_[m.from] = restored;
    board_[m.to] = u.captured;

    if ((m.flags & 1) != 0) {
        int dir = (stm_ == WHITE) ? -1 : 1;
        Square capSq = static_cast<int>(m.to) + 8 * dir;
        board_[capSq] = stm_ == WHITE ? B_PAWN : W_PAWN;
        board_[m.to] = NO_PIECE;
    }
}

uint64_t Position::perft(int depth) const {
    if (depth == 0) return 1;
    if (depth == 1) return legal_moves().size();

    uint64_t nodes = 0;
    Position tmp = *this;
    auto moves = legal_moves();
    for (const auto& m : moves) {
        tmp.do_move(m);
        nodes += tmp.perft(depth - 1);
        tmp.undo_move();
    }
    return nodes;
}

bool Position::is_legal(const Move& m) const {
    auto moves = legal_moves();
    for (const auto& lm : moves) {
        if (lm == m) return true;
    }
    return false;
}

std::vector<Move> Position::legal_moves() const {
    std::vector<Move> out;
    auto pseudos = pseudo_moves();
    out.reserve(pseudos.size());

    Position tmp = *this;
    Color us = stm_;

    for (const auto& m : pseudos) {
        tmp.do_move(m);
        bool ok = !tmp.in_check(us);
        tmp.undo_move();
        if (ok) out.push_back(m);
    }

    return out;
}

Move move_from_string(const Position& pos, const std::string& uci) {
    if (uci.size() < 4) return MOVE_NONE;
    int from = square_from_uci(uci, 0);
    int to = square_from_uci(uci, 2);
    if (from < 0 || to < 0) return MOVE_NONE;

    uint8_t promo = 0;
    if (uci.size() >= 5) {
        PieceType pt = promo_from_char(uci[4]);
        if (pt != 0) promo = static_cast<uint8_t>(pt);
    }

    Move m{static_cast<uint16_t>(from), static_cast<uint16_t>(to), promo, 0};

    auto moves = pos.legal_moves();
    for (const auto& lm : moves) {
        if (lm.from == m.from && lm.to == m.to) {
            if (lm.promo == 0 || lm.promo == m.promo) return lm;
        }
    }

    return MOVE_NONE;
}

std::string move_to_string(const Move& m) {
    if (m == MOVE_NONE) return "0000";

    auto sq_to_str = [](int sq) {
        std::string s;
        s.push_back(static_cast<char>('a' + file_of(sq)));
        s.push_back(static_cast<char>('1' + rank_of(sq)));
        return s;
    };

    std::string out = sq_to_str(m.from) + sq_to_str(m.to);
    if (m.promo != 0) {
        char c = promo_to_char(static_cast<PieceType>(m.promo));
        if (c != '\0') out.push_back(c);
    }
    return out;
}

}
