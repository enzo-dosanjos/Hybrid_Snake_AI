"""
**********************************************************************
Viewer - Live competition games and portable replay recording.
-------------------
copyright            : (C) 2026 by Enzo DOS ANJOS
**********************************************************************
"""

import argparse
import asyncio
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import sys
import threading
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import snake

DIRECTIONS = {"left": (-1, 0), "up": (0, -1), "right": (1, 0), "down": (0, 1)}
ROOT = Path(__file__).resolve().parent


def metrics(board, player):
    """Measure legal neighbors and reachable empty cells with the current bodies held fixed."""
    head = board.bodies[player][0]
    legal = []
    for name, (dx, dy) in DIRECTIONS.items():
        x, y = head[0] + dx, head[1] + dy
        if 0 <= x < board.w and 0 <= y < board.h and board.grid[y][x] == 0:
            legal.append(name)

    visited = {head}
    queue = deque([head])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS.values():
            cell = (x + dx, y + dy)
            nx, ny = cell
            if (0 <= nx < board.w and 0 <= ny < board.h and
                    board.grid[ny][nx] == 0 and cell not in visited):
                visited.add(cell)
                queue.append(cell)

    return legal, len(visited) - 1


class Recorder:
    """Own serializable frames and synchronize live browser input with the game loop."""

    def __init__(self, width, height, growth, names):
        """Initialize the versioned replay and the live input state."""
        self.lock = threading.Lock()
        self.data = {"version": 1, "width": width, "height": height, "growth": growth,
                     "names": names, "frames": [], "finished": False, "error": None}
        self.pending = None
        self.answer = None

    def observe(self, board, players, event):
        """Copy a post-turn snapshot and attach the previous frame's legal alternatives."""
        states = []
        for index, player in enumerate(players):
            legal, space = metrics(board, index) if player.alive else ([], 0)
            states.append({"id": index + 1, "alive": player.alive,
                           "body": list(board.bodies[index]), "legal": legal, "space": space})

        with self.lock:
            if event is not None and self.data["frames"]:
                event = dict(event)
                previous = self.data["frames"][-1]["players"][event["player"] - 1]
                event["legalBefore"] = previous["legal"]
                if event["outcome"] == "user interrupt" and not previous["legal"]:
                    event["outcome"] = "No legal moves"
                event["lengthChange"] = len(states[event["player"] - 1]["body"]) - len(previous["body"])
            self.data["frames"].append({"turn": board.turn,
                                         "round": board.turn // len(players),
                                         "players": states, "event": event})

    def finish(self, error=None):
        """Mark the session complete and release any pending browser controls."""
        with self.lock:
            self.data["finished"] = True
            self.data["error"] = error
            self.pending = None

    def save(self, path):
        """Write the same versioned JSON consumed by the browser and animation exporter."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            path.write_text(json.dumps(self.data, separators=(",", ":")))


class BrowserPlayer(snake.Human):
    def __init__(self, index, recorder, growth):
        """Connect a human player to browser input without a console prompt."""
        super().__init__(index, name=f"Human {index + 1}", growth_rate=growth)
        self.recorder = recorder

    async def ask_move(self, *args, **kwargs):
        """Wait for one legal move tagged with this exact turn to reject stale clicks."""
        with self.recorder.lock:
            frame = self.recorder.data["frames"][-1]
            legal = frame["players"][self.no]["legal"]
            if not legal:
                return None, "user interrupt"
            self.recorder.pending = {"player": self.no + 1, "turn": frame["turn"], "legal": legal}
            self.recorder.answer = None
        while True:
            with self.recorder.lock:
                if self.recorder.answer is not None:
                    answer = self.recorder.answer
                    self.recorder.answer = None
                    self.recorder.pending = None
                    return answer, None
            await asyncio.sleep(0.03)


class DemoPlayer(snake.Human):
    def __init__(self, index, generator, growth):
        """Create a seeded random-legal baseline for demonstrating the viewer."""
        super().__init__(index, name=f"Random {index + 1}", growth_rate=growth)
        self.generator = generator
        self.recorder = None

    async def ask_move(self, *args, **kwargs):
        """Choose from recorded legal moves without pretending to use the hybrid agents."""
        legal = self.recorder.data["frames"][-1]["players"][self.no]["legal"]
        return (self.generator.choice(legal), None) if legal else (None, "user interrupt")


def handler(recorder):
    """Build a localhost handler for viewer assets, incremental snapshots and human moves."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            """Keep polling requests out of the competition console output."""
            pass

        def reply(self, status, body, content_type="application/json"):
            """Send an uncached response with an explicit length."""
            payload = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            """Serve only known assets or the frames following the browser's cursor."""
            request = urlparse(self.path)
            if request.path == "/api/state":
                try:
                    after = max(0, int(parse_qs(request.query).get("after", ["0"])[0]))
                except ValueError:
                    self.reply(400, {"error": "Invalid frame cursor"})
                    return
                with recorder.lock:
                    data = dict(recorder.data)
                    data["frames"] = recorder.data["frames"][after:]
                    data["pending"] = recorder.pending
                self.reply(200, data)
            elif request.path in ("/", "/index.html", "/Viewer.js", "/Style.css"):
                name = "index.html" if request.path == "/" else request.path[1:]
                content_type = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}
                self.reply(200, (ROOT / name).read_bytes(), content_type[Path(name).suffix])
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            """Accept a same-origin JSON move only while its human turn is pending."""
            if self.path != "/api/move" or self.headers.get("Content-Type") != "application/json":
                self.reply(400, {"error": "Expected a JSON move"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024:
                    raise ValueError("Invalid length")
                move = json.loads(self.rfile.read(length))
                with recorder.lock:
                    pending = recorder.pending
                    valid = (isinstance(move, dict) and pending is not None and
                             recorder.answer is None and move.get("turn") == pending["turn"] and
                             move.get("player") == pending["player"] and move.get("action") in pending["legal"])
                    if valid:
                        recorder.answer = move["action"]
                self.reply(200 if valid else 409, {"accepted": valid})
            except (ValueError, TypeError):
                self.reply(400, {"error": "Invalid move"})
    return Handler


async def play(args, recorder, origins):
    """Run the existing competition loop and always stop its bot processes on exit."""
    players = []
    for index, program in enumerate(args.players):
        if program == "human":
            player = BrowserPlayer(index, recorder, args.growth)
        elif program == "random":
            player = DemoPlayer(index, random.Random(args.seed + index), args.growth)
            player.recorder = recorder
        else:
            player = snake.AI(index, program, False, growth_rate=args.growth)
        players.append(player)
    try:
        await snake.game(players, origins, *args.grid, args.growth, False,
                         observer=recorder.observe, turn_delay=args.delay)
        recorder.finish()
    except asyncio.CancelledError:
        recorder.finish("Session stopped by user")
    except Exception as error:
        recorder.finish(str(error))
        print(f"Game stopped: {error}", file=sys.stderr)
    finally:
        for player in players:
            if isinstance(player, snake.AI) and hasattr(player, "prog"):
                await player.stop_game()
        if args.save_replay:
            recorder.save(args.output)
            print(f"Replay saved: {args.output}")


def main():
    """Start a local live viewer or record a headless match using competition rules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("players", nargs="*", default=["random", "random"],
                        help="Bot paths, human (browser), or random (demo)")
    parser.add_argument("--grid", type=int, nargs=2, default=[12, 10])
    parser.add_argument("--growth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--delay", type=float, default=0.18)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=Path("replays/latest.json"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--games", type=int, default=1, help="Number of headless matches")
    parser.add_argument("--record-every", type=int, default=1, help="Save every Nth headless match")
    args = parser.parse_args()
    width, height = args.grid
    if not 2 <= width <= 200 or not 2 <= height <= 200 or args.growth < 1 or not 2 <= len(args.players) <= 4 or args.delay < 0:
        parser.error("Use dimensions >= 2, positive growth, nonnegative delay and 2–4 players")
    if args.headless and "human" in args.players:
        parser.error("A human needs the live browser")
    if args.games < 1 or args.record_every < 1 or (not args.headless and (args.games != 1 or args.record_every != 1)):
        parser.error("Batch options require headless mode and positive counts")
    names = [f"{Path(name).name} · P{index + 1}" for index, name in enumerate(args.players)]
    recorder = Recorder(width, height, args.growth, names)
    origins = [(0, 0), (width - 1, height - 1), (0, height - 1), (width - 1, 0)][:len(args.players)]
    args.save_replay = True
    if args.headless:
        output = args.output
        seed = args.seed
        for game in range(1, args.games + 1):
            recorder = Recorder(width, height, args.growth, names)
            args.seed = seed + (game - 1) * len(args.players)
            args.save_replay = game % args.record_every == 0
            args.output = output if args.games == 1 else output.with_name(f"{output.stem}-{game:06d}{output.suffix}")
            asyncio.run(play(args, recorder, origins))
            if recorder.data["error"]:
                return 1
        return 0
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(recorder))
    loop = asyncio.new_event_loop()
    task = loop.create_task(play(args, recorder, origins))
    worker = threading.Thread(target=lambda: loop.run_until_complete(task), daemon=True)
    worker.start()
    print(f"Viewer: http://127.0.0.1:{args.port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if not task.done():
            loop.call_soon_threadsafe(task.cancel)
        worker.join(timeout=15)
    finally:
        server.server_close()
        if not worker.is_alive():
            loop.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
