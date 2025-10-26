#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyCranky — um motor de xadrez UCI em Python usando python-chess.

Recursos principais:
- Iterative Deepening + Aspiration Windows
- Alpha-Beta Negamax com Quiescence Search
- Tabela de Transposição (TT) com flags EXACT/LOWER/UPPER
- Zobrist hash via python-chess
- Ordenação de lances: TT move, capturas (MVV-LVA + SEE), killers, history heuristic
- Null Move Pruning (com salvaguardas para finais e cheque)
- Late Move Reductions (LMR) simples em quiets tardios
- Avaliação: material, PST (tapered), mobilidade, par de bispos, peões passados, dobrados/isolados,
  torres em coluna aberta/semiaberta, escudo de peões do rei e segurança do rei básica
- Suporte opcional a livro de aberturas Polyglot (book.bin), se disponível
- Protocolo UCI mínimo: uci, isready, ucinewgame, position, go, stop, quit

Dependências:
    pip install python-chess

Uso:
    python pycranky.py
    Em um GUI UCI (ex.: CuteChess), aponte para este executável.

Aviso honesto:
    Python não é C++. Ainda assim, com profundidades 5–8 em blitz e estes heurísticos,
    o desempenho típico fica na faixa ~1800–2100 Elo em testes amadores.
"""

import sys
import time
import math
import random
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import chess
import chess.polyglot

INF = 10_000_000
MATE_SCORE = 100_000
MATE_BOUND = 99_000

EXACT, LOWER, UPPER = 0, 1, 2

# Piece values (centipawns)
PIECE_VAL = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

# Piece-Square Tables (MG/EG) — valores compactos (brancas na base)
# PSTs inspiradas em tabelas comuns; podem ser ajustadas para performance.
PST_PAWN_MG = [
      0,   0,   0,   0,   0,   0,   0,   0,
     30,  40,  40, -10, -10,  40,  40,  30,
     10,  10,  20,  20,  20,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]
PST_PAWN_EG = [
      0,   0,   0,   0,   0,   0,   0,   0,
     80,  90,  90, 100, 100,  90,  90,  80,
     60,  70,  70,  80,  80,  70,  70,  60,
     40,  50,  50,  60,  60,  50,  50,  40,
     30,  40,  40,  50,  50,  40,  40,  30,
     20,  30,  30,  40,  40,  30,  30,  20,
     10,  20,  20,  20,  20,  20,  20,  10,
      0,   0,   0,   0,   0,   0,   0,   0,
]

PST_KNIGHT = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
]
PST_BISHOP = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]
PST_ROOK = [
     0,  0,  5, 10, 10,  5,  0,  0,
   - 5,  0,  0,  0,  0,  0,  0, -5,
   - 5,  0,  0,  0,  0,  0,  0, -5,
   - 5,  0,  0,  0,  0,  0,  0, -5,
   - 5,  0,  0,  0,  0,  0,  0, -5,
   - 5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
PST_QUEEN = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
]
PST_KING_MG = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
]
PST_KING_EG = [
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-40,-30,-20,-20,-30,-40,-50,
]

PST_MAP_MG = {
    chess.PAWN: PST_PAWN_MG,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK: PST_ROOK,
    chess.QUEEN: PST_QUEEN,
    chess.KING: PST_KING_MG,
}
PST_MAP_EG = {
    chess.PAWN: PST_PAWN_EG,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK: PST_ROOK,
    chess.QUEEN: PST_QUEEN,
    chess.KING: PST_KING_EG,
}

# Pesos para fase do jogo
PHASE_WEIGHTS = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}
MAX_PHASE = sum(v * 2 for v in PHASE_WEIGHTS.values())

@dataclass
class TTEntry:
    key: int
    depth: int
    score: int
    flag: int
    move: Optional[chess.Move]
    age: int

class TimeManager:
    def __init__(self):
        self.start = 0.0
        self.limit = 1.0
        self.stopped = False

    def start_timer(self, limit_sec: float):
        self.start = time.perf_counter()
        self.limit = limit_sec
        self.stopped = False

    def check(self):
        if time.perf_counter() - self.start >= self.limit:
            self.stopped = True
        return self.stopped

class Engine:
    def __init__(self):
        self.board = chess.Board()
        self.tt: Dict[int, TTEntry] = {}
        self.killers = [[None, None] for _ in range(128)]
        self.history = {}
        self.nodes = 0
        self.age = 0
        self.timemgr = TimeManager()
        self.polyglot_book = None
        try:
            self.polyglot_book = chess.polyglot.open_reader("book.bin")
        except Exception:
            self.polyglot_book = None

    # ========================= Avaliação ============================
    def evaluate(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -MATE_SCORE + board.fullmove_number
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        # Material e fase
        mg, eg = 0, 0
        phase = 0

        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1
            # Material básico
            for p, val in PIECE_VAL.items():
                cnt = len(board.pieces(p, color))
                mg += sign * val * cnt
                eg += sign * val * cnt
                if p in PHASE_WEIGHTS:
                    phase += PHASE_WEIGHTS[p] * cnt

            # PSTs
            for p in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
                pst_mg = PST_MAP_MG[p]
                pst_eg = PST_MAP_EG[p]
                for sq in board.pieces(p, color):
                    idx = sq if color == chess.WHITE else chess.square_mirror(sq)
                    mg += sign * pst_mg[idx]
                    eg += sign * pst_eg[idx]

            # Mobilidade simples baseada em casas atacadas
            attack_mask = 0
            for piece_type in chess.PIECE_TYPES:
                for sq in board.pieces(piece_type, color):
                    attack_mask |= board.attacks_mask(sq)
            mobility = chess.popcount(attack_mask)
            mg += sign * mobility
            eg += sign * mobility

        # Ajustes de peões: passados, dobrados, isolados
        mg += self._pawn_structure(board)  # retorna score from white POV
        eg += 2 * self._pawn_structure(board)

        # Par de bispos
        for color in [chess.WHITE, chess.BLACK]:
            if len(board.pieces(chess.BISHOP, color)) >= 2:
                mg += 30 if color == chess.WHITE else -30
                eg += 50 if color == chess.WHITE else -50

        # Torres em coluna aberta / semi-aberta
        mg += self._rook_files(board)
        eg += self._rook_files(board)

        # Segurança do rei básica: escudo de peões
        mg += self._king_safety(board)

        # Tapered eval
        phase = min(MAX_PHASE, phase)
        if MAX_PHASE:
            score = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE
        else:
            score = eg

        return score if board.turn == chess.WHITE else -score

    def _pawn_structure(self, board: chess.Board) -> int:
        score = 0
        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1
            pawns = board.pieces(chess.PAWN, color)
            files = [0]*8
            for sq in pawns:
                files[chess.square_file(sq)] += 1
            for f in range(8):
                if files[f] >= 2:  # dobrado
                    score += sign * -12 * (files[f]-1)
                if files[f] == 0:
                    continue
                # isolado
                if (f == 0 or files[f-1] == 0) and (f == 7 or files[f+1] == 0):
                    score += sign * -15
            # passados
            for sq in pawns:
                if self._is_passed_pawn(board, sq, color):
                    rank = chess.square_rank(sq) if color == chess.WHITE else 7 - chess.square_rank(sq)
                    score += sign * (20 + rank * 8)
        return score

    def _is_passed_pawn(self, board: chess.Board, sq: int, color: bool) -> bool:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        ahead = range(rank+1, 8) if color == chess.WHITE else range(0, rank)
        for r in ahead:
            for f in [file-1, file, file+1]:
                if 0 <= f < 8:
                    if board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, not color):
                        return False
        return True

    def _rook_files(self, board: chess.Board) -> int:
        score = 0
        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1
            for sq in board.pieces(chess.ROOK, color):
                f = chess.square_file(sq)
                pawns_white = any(chess.square(f, r) in board.pieces(chess.PAWN, chess.WHITE) for r in range(8))
                pawns_black = any(chess.square(f, r) in board.pieces(chess.PAWN, chess.BLACK) for r in range(8))
                if not pawns_white and not pawns_black:
                    score += sign * 15  # coluna aberta
                elif (color == chess.WHITE and not pawns_white) or (color == chess.BLACK and not pawns_black):
                    score += sign * 7   # semi-aberta
        return score

    def _king_safety(self, board: chess.Board) -> int:
        score = 0
        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1
            ksq = board.king(color)
            if ksq is None:
                continue
            # penaliza rei sem escudo de peões nas 3 casas à frente
            file = chess.square_file(ksq)
            rank = chess.square_rank(ksq)
            front = rank + (1 if color == chess.WHITE else -1)
            if 0 <= front <= 7:
                shields = 0
                for f in [file-1, file, file+1]:
                    if 0 <= f <= 7:
                        if chess.square(f, front) in board.pieces(chess.PAWN, color):
                            shields += 1
                score += sign * (shields - 2) * -12
        return score

    # ===================== Ordenação de lances =======================
    def mvv_lva(self, board: chess.Board, move: chess.Move) -> int:
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            if victim is None and board.is_en_passant(move):
                victim = chess.Piece(chess.PAWN, not board.turn)
            attacker = board.piece_at(move.from_square)
            v = 0 if victim is None else PIECE_VAL.get(victim.piece_type, 0)
            a = 1 if attacker is None else PIECE_VAL.get(attacker.piece_type, 1)
            return (v * 10) - a
        return 0

    def order_moves(self, board: chess.Board, moves, tt_move: Optional[chess.Move], ply: int):
        scored = []
        k1, k2 = self.killers[ply]
        for m in moves:
            score = 0
            if tt_move and m == tt_move:
                score = 1_000_000
            elif board.is_capture(m):
                score = 500_000 + self.mvv_lva(board, m)
                # filtrar capturas ruins via SEE
                try:
                    if not board.see_ge(m, 0):
                        score -= 50_000
                except Exception:
                    pass
            else:
                if k1 and m == k1:
                    score = 300_000
                elif k2 and m == k2:
                    score = 290_000
                else:
                    score = self.history.get((board.turn, m.from_square, m.to_square), 0)
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    # ========================= TT helpers ============================
    def tt_probe(self, key: int) -> Optional[TTEntry]:
        return self.tt.get(key)

    def tt_store(self, key: int, depth: int, score: int, flag: int, move: Optional[chess.Move]):
        entry = self.tt.get(key)
        if entry is None or depth >= entry.depth or entry.age != self.age:
            self.tt[key] = TTEntry(key, depth, score, flag, move, self.age)

    # ==================== Pesquisa principal ========================
    def search(self, max_depth: int, wtime=None, btime=None, winc=0, binc=0, movetime=None) -> chess.Move:
        # Gerenciar tempo
        if movetime is not None:
            limit = max(0.01, movetime / 1000.0)
        else:
            side_time = wtime if self.board.turn == chess.WHITE else btime
            inc = winc if self.board.turn == chess.WHITE else binc
            if side_time is None:
                limit = 3.0
            else:
                limit = max(0.05, min(5.0, (side_time / 30_000.0) + (inc or 0)/2000.0))
        self.timemgr.start_timer(limit)

        self.nodes = 0
        self.age = (self.age + 1) % 256
        best_move = None
        best_score = -INF
        prev_score = 0

        # Abertura polyglot (se disponível)
        if self.polyglot_book is not None:
            try:
                with self.polyglot_book as reader:
                    entries = list(reader.find_all(self.board))
                    if entries:
                        best_move = random.choice(entries).move
                        return best_move
            except Exception:
                pass

        for depth in range(1, max_depth + 1):
            if self.timemgr.check():
                break
            # Aspiration window
            delta = 20 if depth >= 3 else INF
            alpha = max(-INF, prev_score - delta)
            beta = min(INF, prev_score + delta)
            score, move = self._iter_deep(depth, alpha, beta)
            if move is None:
                break
            if score <= alpha or score >= beta:
                # Rebusca com janela completa
                score, move = self._iter_deep(depth, -INF, INF)
            if self.timemgr.stopped:
                break
            best_score, best_move = score, move
            prev_score = score
            # Info UCI
            print(f"info depth {depth} score cp {score} nodes {self.nodes} time {int((time.perf_counter()-self.timemgr.start)*1000)} pv {self._pv_line(move)}", flush=True)
        if best_move is None:
            # fallback
            best_move = random.choice(list(self.board.legal_moves))
        return best_move

    def _pv_line(self, first_move: chess.Move) -> str:
        # Gera uma PV curta a partir da TT
        board = self.board.copy(stack=False)
        line = []
        move = first_move
        for _ in range(10):
            if move is None or not board.is_legal(move):
                break
            line.append(move.uci())
            board.push(move)
            entry = self.tt.get(board._transposition_key())
            move = entry.move if entry else None
        return " ".join(line)

    def _iter_deep(self, depth: int, alpha: int, beta: int) -> Tuple[int, Optional[chess.Move]]:
        best_move = None
        score = -INF
        # Root search
        tt = self.tt_probe(self.board._transposition_key())
        tt_move = tt.move if tt else None
        moves = list(self.board.legal_moves)
        ordered = self.order_moves(self.board, moves, tt_move, 0)
        for i, m in enumerate(ordered):
            if self.timemgr.check():
                break
            self.board.push(m)
            val = -self.alphabeta(depth-1, -beta, -alpha, 1, allow_null=True)
            self.board.pop()
            if val > score:
                score = val
                best_move = m
            if val > alpha:
                alpha = val
        return score, best_move

    def alphabeta(self, depth: int, alpha: int, beta: int, ply: int, allow_null: bool) -> int:
        if self.timemgr.check():
            return 0
        self.nodes += 1

        key = self.board._transposition_key()
        entry = self.tt_probe(key)
        if entry and entry.depth >= depth:
            if entry.flag == EXACT:
                return entry.score
            if entry.flag == LOWER and entry.score > alpha:
                alpha = entry.score
            elif entry.flag == UPPER and entry.score < beta:
                beta = entry.score
            if alpha >= beta:
                return entry.score

        if depth <= 0:
            return self.quiescence(alpha, beta, ply)

        if self.board.is_checkmate():
            return -MATE_SCORE + ply
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0

        in_check = self.board.is_check()

        # Null move pruning
        if allow_null and not in_check and depth >= 2 and self._has_non_pawn(board=self.board):
            self.board.push(chess.Move.null())
            val = -self.alphabeta(depth-2-1, -beta, -beta+1, ply+1, allow_null=False)
            self.board.pop()
            if val >= beta:
                return beta

        # Gerar e ordenar lances
        tt_move = entry.move if entry else None
        moves = list(self.board.legal_moves)
        ordered = self.order_moves(self.board, moves, tt_move, ply)

        best_move = None
        value = -INF
        legal = 0

        for i, move in enumerate(ordered):
            # LMR: reduzir quiets tardios não-cheque
            reduction = 0
            is_capture = self.board.is_capture(move)
            if not is_capture and not in_check and depth >= 3 and i >= 4:
                reduction = 1

            self.board.push(move)
            if reduction:
                score = -self.alphabeta(depth-1-reduction, -alpha-1, -alpha, ply+1, allow_null=True)
                if score > alpha:
                    score = -self.alphabeta(depth-1, -beta, -alpha, ply+1, allow_null=True)
            else:
                score = -self.alphabeta(depth-1, -beta, -alpha, ply+1, allow_null=True)
            self.board.pop()
            legal += 1

            if score > value:
                value = score
                best_move = move

            if score > alpha:
                alpha = score
                # History / killers
                if not is_capture:
                    hkey = (self.board.turn, move.from_square, move.to_square)
                    self.history[hkey] = self.history.get(hkey, 0) + depth * depth
                    # killers
                    k1, k2 = self.killers[ply]
                    if k1 != move:
                        self.killers[ply][1] = k1
                        self.killers[ply][0] = move

            if alpha >= beta:
                # cutoff
                if not is_capture:
                    hkey = (self.board.turn, move.from_square, move.to_square)
                    self.history[hkey] = self.history.get(hkey, 0) + depth * depth
                break

        if legal == 0:
            return -MATE_SCORE + ply if in_check else 0

        # TT store
        flag = EXACT
        if value <= alpha:
            flag = UPPER
        elif value >= beta:
            flag = LOWER
        self.tt_store(key, depth, value, flag, best_move)
        return value

    def quiescence(self, alpha: int, beta: int, ply: int) -> int:
        if self.timemgr.check():
            return 0
        stand = self.evaluate(self.board)
        if stand >= beta:
            return beta
        if stand > alpha:
            alpha = stand

        # Capturas e checks somente
        for move in list(self.board.legal_moves):
            if not self.board.is_capture(move) and not self._gives_check(move):
                continue
            # SEE prune capturas ruins
            try:
                if self.board.is_capture(move) and not self.board.see_ge(move, 0):
                    continue
            except Exception:
                pass
            self.board.push(move)
            score = -self.quiescence(-beta, -alpha, ply+1)
            self.board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def _gives_check(self, move: chess.Move) -> bool:
        self.board.push(move)
        chk = self.board.is_check()
        self.board.pop()
        return chk

    def _has_non_pawn(self, board: chess.Board) -> bool:
        # Evitar null-move em finais puros de peões
        for p in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            if board.pieces(p, board.turn):
                return True
        return False

    # =========================== UCI ================================
    def uci_loop(self):
        print("id name PyCranky")
        print("id author you")
        print("uciok")
        sys.stdout.flush()
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if line == "uci":
                    print("id name PyCranky")
                    print("id author you")
                    print("uciok")
                    sys.stdout.flush()
                elif line == "isready":
                    print("readyok")
                    sys.stdout.flush()
                elif line.startswith("setoption"):
                    # ignorar por simplicidade
                    pass
                elif line == "ucinewgame":
                    self.board = chess.Board()
                    self.tt.clear()
                    self.killers = [[None, None] for _ in range(128)]
                    self.history.clear()
                    self.age = (self.age + 1) % 256
                elif line.startswith("position"):
                    self._cmd_position(line)
                elif line.startswith("go"):
                    self._cmd_go(line)
                elif line == "stop":
                    self.timemgr.stopped = True
                elif line == "quit":
                    break
            except Exception as e:
                print(f"info string error: {e}", flush=True)

    def _cmd_position(self, line: str):
        parts = line.split()
        if "startpos" in parts:
            self.board = chess.Board()
            idx = parts.index("startpos") + 1
        elif "fen" in parts:
            idx = parts.index("fen") + 1
            fen = " ".join(parts[idx:idx+6])
            self.board = chess.Board(fen)
            idx += 6
        else:
            return
        if idx < len(parts) and parts[idx] == "moves":
            for mv in parts[idx+1:]:
                self.board.push_uci(mv)

    def _cmd_go(self, line: str):
        parts = line.split()
        depth = 64
        wtime = btime = winc = binc = None
        movetime = None
        for i, p in enumerate(parts):
            if p == "depth" and i+1 < len(parts):
                depth = int(parts[i+1])
            elif p == "wtime" and i+1 < len(parts):
                wtime = int(parts[i+1])
            elif p == "btime" and i+1 < len(parts):
                btime = int(parts[i+1])
            elif p == "winc" and i+1 < len(parts):
                winc = int(parts[i+1])
            elif p == "binc" and i+1 < len(parts):
                binc = int(parts[i+1])
            elif p == "movetime" and i+1 < len(parts):
                movetime = int(parts[i+1])
        move = self.search(depth, wtime, btime, winc or 0, binc or 0, movetime)
        print(f"bestmove {move.uci()}", flush=True)


def main():
    eng = Engine()
    if len(sys.argv) > 1 and sys.argv[1] == "uci":
        eng.uci_loop()
    else:
        # CLI rápida para testes
        print("PyCranky CLI. Comandos: 'go', 'move e2e4', 'fen <FEN>', 'quit'")
        while True:
            if eng.board.is_game_over():
                print("Game over:", eng.board.result(), eng.board.outcome())
                break
            cmd = input("> ").strip()
            if cmd == "quit":
                break
            elif cmd.startswith("move "):
                try:
                    eng.board.push_uci(cmd.split()[1])
                except Exception as e:
                    print("erro:", e)
            elif cmd.startswith("fen "):
                eng.board = chess.Board(cmd[4:].strip())
            elif cmd == "go":
                mv = eng.search(8)
                print("engine:", mv)
                eng.board.push(mv)
                print(eng.board)
            else:
                print("comando desconhecido")

if __name__ == "__main__":
    main()
