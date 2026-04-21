import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from game_logic import HUMAN, ROBOT, TicTacToe
from ros_publisher import BlockPosePublisher, GridFrame, USING_REAL_ROS


WINDOW = 600
MARGIN = 40
CELL_PX = (WINDOW - 2 * MARGIN) // 3
STATUS_H = 80
LINE_W = 6

BG = (245, 245, 240)
GRID_COLOR = (40, 40, 40)
X_COLOR = (30, 90, 200)
O_COLOR = (200, 60, 60)

ROBOT_DELAY_MS = 350


def _load_pygame():
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("pygame is required to run the tic-tac-toe UI") from exc
    return pygame


def draw_board(screen, game, status, font, pygame_mod=None):
    pygame_mod = pygame_mod or _load_pygame()
    screen.fill(BG)
    screen.blit(font.render(status, True, (20, 20, 20)), (MARGIN, 10))

    top = STATUS_H
    for i in range(4):
        x = MARGIN + i * CELL_PX
        pygame_mod.draw.line(screen, GRID_COLOR, (x, top), (x, top + 3 * CELL_PX), LINE_W)
        y = top + i * CELL_PX
        pygame_mod.draw.line(screen, GRID_COLOR, (MARGIN, y), (MARGIN + 3 * CELL_PX, y), LINE_W)

    pad = CELL_PX // 5
    for r in range(3):
        for c in range(3):
            cell = game.board[r][c]
            cx = MARGIN + c * CELL_PX + CELL_PX // 2
            cy = top + r * CELL_PX + CELL_PX // 2
            color = O_COLOR if cell == HUMAN else X_COLOR
            if cell == "X":
                a = CELL_PX // 2 - pad
                pygame_mod.draw.line(screen, color, (cx - a, cy - a), (cx + a, cy + a), 10)
                pygame_mod.draw.line(screen, color, (cx + a, cy - a), (cx - a, cy + a), 10)
            elif cell == "O":
                pygame_mod.draw.circle(screen, color, (cx, cy), CELL_PX // 2 - pad, 10)


def cell_from_pixel(mx, my):
    if not (MARGIN <= mx < MARGIN + 3 * CELL_PX):
        return None
    if not (STATUS_H <= my < STATUS_H + 3 * CELL_PX):
        return None
    return (my - STATUS_H) // CELL_PX, (mx - MARGIN) // CELL_PX


def status_text(game):
    w = game.winner()
    if w == HUMAN:
        return "You win! Press R to restart."
    if w == ROBOT:
        return "Robot wins. Press R to restart."
    if game.is_full():
        return "Draw. Press R to restart."
    if game.turn == HUMAN:
        return f"Your turn ({HUMAN}). Click a cell."
    return "Robot thinking..."


def run(grid, headless_script=False, scripted_moves=None, max_frames=None):
    if headless_script:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame = _load_pygame()

    pygame.init()
    pygame.display.set_caption("Tic-Tac-Toe / ROS block-placement publisher")
    screen = pygame.display.set_mode((WINDOW, STATUS_H + 3 * CELL_PX + MARGIN))
    font = pygame.font.SysFont("sans", 22)

    game = TicTacToe()
    publisher = BlockPosePublisher(grid=grid)
    print(f"[tic_tac_toe] ROS backend: {'real rospy' if USING_REAL_ROS else 'mock rospy'}")
    print(f"[tic_tac_toe] Grid origin=({grid.origin_x}, {grid.origin_y}) "
          f"cell={grid.cell_size} frame={grid.frame_id}")

    clock = pygame.time.Clock()
    robot_started = None
    scripted = iter(scripted_moves or [])
    frames = 0
    running = True

    while running:
        frames += 1
        if max_frames is not None and frames > max_frames:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()
                    robot_started = None
                elif event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if game.turn == HUMAN and not game.is_over():
                    cell = cell_from_pixel(*event.pos)
                    if cell and game.place(*cell, HUMAN):
                        publisher.publish_move(cell[0], cell[1], HUMAN)
                        robot_started = pygame.time.get_ticks()

        if headless_script and game.turn == HUMAN and not game.is_over():
            nxt = next(scripted, None)
            if nxt is not None and game.place(*nxt, HUMAN):
                publisher.publish_move(nxt[0], nxt[1], HUMAN)
                robot_started = pygame.time.get_ticks()

        if game.turn == ROBOT and not game.is_over():
            if robot_started is None:
                robot_started = pygame.time.get_ticks()
            delay = 0 if headless_script else ROBOT_DELAY_MS
            if pygame.time.get_ticks() - robot_started >= delay:
                move = game.best_move(ROBOT)
                if move is not None:
                    game.place(*move, ROBOT)
                    publisher.publish_move(move[0], move[1], ROBOT)
                robot_started = None

        draw_board(screen, game, status_text(game), font, pygame)
        pygame.display.flip()
        clock.tick(60)

        if headless_script and game.is_over():
            draw_board(screen, game, status_text(game), font, pygame)
            pygame.display.flip()
            running = False

    pygame.quit()
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--origin-x", type=float, default=0.40)
    p.add_argument("--origin-y", type=float, default=0.00)
    p.add_argument("--cell", type=float, default=0.06)
    p.add_argument("--place-z", type=float, default=0.05)
    p.add_argument("--frame-id", default="base_link")
    p.add_argument("--headless-script", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    grid = GridFrame(args.origin_x, args.origin_y, args.cell, args.place_z, args.frame_id)
    scripted = None
    if args.headless_script:
        scripted = [(1, 1), (0, 0), (2, 2), (0, 2), (2, 0), (2, 1), (0, 1)]
    return run(grid, args.headless_script, scripted, 600 if args.headless_script else None)


if __name__ == "__main__":
    sys.exit(main())
