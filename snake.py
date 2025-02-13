#!/usr/bin/env python3

from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, Any
from io import StringIO, BytesIO
from pathlib import Path
import argparse
import asyncio
import contextlib
import os
import platform
import re
import sys

# Default Timeouts :
TIMEOUT_LENGTH = 1  # sec
DISCORD_TIMEOUT = 60  # sec

WIDTH = 10
HEIGHT = 10

CELL_SIZE = 50 # pixels
SIDE_OFFSET = 20

# Usefull emojis :
EMOJI_NUMBERS = ("0️⃣", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣")
EMOJI_COLORS = ("🟡", "🔴", "🔵", "🟢", "🟠", "🟣", "🟤", "⚪️", "⚫️")
CODE_COLORS = ("fcd53f", "f8312f", "0074ba", "00d26a", "ff6723", "8d65c5", "6d4534", "ffffff", "000000")
CODE_COLORS = dict(zip(EMOJI_COLORS, (f'#{col}' for col in CODE_COLORS)))

# what is the type of a valid move or a valid input (when it has a specific format) to use in typing
ValidMove = Any
ValidInput = str

# input and output functions types
InputFunction = Callable[
    ..., str
]  # function asking a discord player to make a move, returns the discord answer
OutputFunction = Callable[
    [str], None
]  # function called when an AI wants to "talk" to discord, the argument being the message


class Player(ABC):

    ofunc = None

    def __init__(self, no: int, name: str, growth_rate: int, **kwargs):
        """
        The abstract Player constructor

        Args:
            no (int): player number/id
            name (str, optional): The player name. Defaults to None.
        """
        # You can add any number of kwargs you want that will be passed in the discord command for your game

        self.no = no

        self.icon = self.no
        self.name = name
        self.rendered_name = None
        self.growth_rate = growth_rate

    @abstractmethod
    async def start_game(self, no, w, h, p_pos):
        self.alive = True
        self.no = no
        self.w = w
        self.h = h
        self.p_pos = p_pos

        await Player.print(f"{self} is ready!")

    @abstractmethod
    async def lose_game(self):
        await Player.print(f"{self} is eliminated")

    async def ask_move(
        self, *args, **kwargs
    ) -> tuple[ValidMove, None] | tuple[None | str]:
        pass

    @abstractmethod
    async def tell_move(self, move: ValidInput):
        pass

    async def tell_other_players(self, players: list[Player], move: ValidInput):
        for other_player in players:
            if self != other_player and other_player.alive:
                await other_player.tell_move(move)

    @staticmethod
    async def sanithize(
        userInput: str, **kwargs
    ) -> tuple[ValidMove, None] | tuple[None | str]:
        """
        Parses raw user input text into an error message or a valid move

        Args:
            userInput (`str`): the raw user input text

        Returns:
            `tuple[ValidMove, None] | tuple[None | str]`
        """

        if userInput == "stop":
            # When a human player (or an AI, who knows) wants to abandon.
            return None, "user interrupt"

        # process user inputs
        moves = ("up", "down", "left", "right", "ready")
        if userInput not in moves:
            await Player.print(f"invalid move: {userInput}")
            return None, "invalid move"

        processed_input: ValidMove = userInput

        return processed_input, None

    @staticmethod
    async def print(output: StringIO | str, send_discord=True, end="\n"):
        if isinstance(output, StringIO):
            text = output.getvalue()
            output.close()
        else:
            text = output + end
        print(text, end="")
        if Player.ofunc and send_discord:
            if len(text) < 300:
                await Player.ofunc(text)
            else:
                fp = Player.board_as_img(text)
                await Player.ofunc(file=fp)

    @staticmethod
    def board_as_img(text: str) -> BytesIO:
        # Avoid importing external libraries when it's not needed
        # This function is only called on discord
        from PIL import Image, ImageDraw


        # Create a blank image with white background
        img = Image.new("RGB", (WIDTH * CELL_SIZE + 2*SIDE_OFFSET, HEIGHT * CELL_SIZE + 2*SIDE_OFFSET), color="#313338")
        draw = ImageDraw.Draw(img)

        radius = CELL_SIZE//2*0.7

        # Draw the grid
        for y, line in enumerate(text.split("\n")):
            for x, char in enumerate(line):
                draw.circle(((x+0.5)*CELL_SIZE + SIDE_OFFSET, (y+0.5)*CELL_SIZE + SIDE_OFFSET), radius, CODE_COLORS[char])

        fp = BytesIO()
        img.save(fp, format="PNG")
        fp.seek(0)
        

    def __str__(self):
        return self.rendered_name


class Human(Player):

    def __init__(
        self, no: int, name: str = None, ifunc: InputFunction = None, **kwargs
    ):
        """
        The human player constructor
        Let ifunc be None to get terminal input (for a local game)

        Args:
            no (`int`): player number/id
            name (`str`, optional): The player name. Defaults to None.
            ifunc (`InputFunction`, optional): The input function. Defaults to None.
        """
        super().__init__(no, name, **kwargs)
        self.ifunc = ifunc

        # Personnalize human players name specifically
        self.rendered_name = (
            f"{self.name} {self.icon}" if name else f"Player {self.icon}"
        )

    async def start_game(self, *args, **kwargs):
        await super().start_game(*args, **kwargs)

    async def lose_game(self):
        await super().lose_game()

    async def ask_move(self, *args, **kwargs):
        await super().ask_move(*args, **kwargs)

        await Player.print(f"Awaiting {self}'s move : ", end="")
        try:
            user_input = await self.input()
        except asyncio.TimeoutError:
            await Player.print(
                f"User did not respond in time (over {DISCORD_TIMEOUT}s)"
            )
            return None, "timeout"
    
        return await Human.sanithize(user_input, **kwargs)

    async def tell_move(self, move: ValidInput):
        return await super().tell_move(move)

    async def input(self):
        if not self.ifunc:
            return input()

        user_input = await asyncio.wait_for(
            self.ifunc(self.name), timeout=DISCORD_TIMEOUT
        )
        await Player.print(user_input, send_discord=False)
        return user_input


class AI(Player):

    @staticmethod
    def prepare_command(prog_path: str | Path):
        """
        Prepares the command to start the AI

        Args:
            progPath (`str` | `Path`): the path to the program

        Raises:
            Exception: File not found error

        Returns:
            `str`: the command to start the AI
        """
        path = Path(prog_path).resolve(strict=True)

        match path.suffix:
            case ".py":
                if platform.system() == "Windows":
                    cmd = f"python {path}"
                else:
                    cmd = f"python3 {path}"
            case ".js":
                cmd = f"node {path}"
            case ".class":
                cmd = f"java -cp {path.parent} {path.stem}"
            case _:
                cmd = f"{path}"

        use_firejail = os.environ.get("FIREJAIL_AVAILABLE") == "1"
        if use_firejail:
            cmd = f"firejail --net=none --read-only=/ --whitelist={path.parent} {cmd}"

        return cmd

    def __init__(self, no: int, prog_path: str, discord: bool, **kwargs):
        """
        The AI player constructor

        Args:
            no (int): player number/id
            prog_path (str): AI's program path
            discord (bool): if it is instantiated through discord to associate the user tag
        """
        super().__init__(no, Path(prog_path).stem, **kwargs)
        self.prog_path = prog_path

        if discord:
            # if it's through discord, self.name should be the discord user's ID
            if self.name.startswith("ai_"):
                self.name = self.name[3:]
            self.rendered_name = f"<@{self.name}>'s AI {self.icon}"
        else:
            self.rendered_name = f"AI {self.icon} ({self.name})"


    async def start_game(self, *args, **kwargs):
        # Specify here what parameters are required to start a game for an AI player.
        await super().start_game(*args, **kwargs)
        cmd = AI.prepare_command(self.prog_path)
        self.prog = await asyncio.create_subprocess_shell(
            cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        if self.prog.stdin:
            # Here, write the NORMALIZED message you'll send to the AIs for them to start the game.
            # This is what this method's kwargs are for, the AI will need
            self.prog.stdin.write(f"{self.w} {self.h}\n".encode()) # Grid size
            self.prog.stdin.write(f"{self.growth_rate}\n".encode()) # Growth rate
            self.prog.stdin.write(f"{len(self.p_pos)} {self.no+1}\n".encode()) # Number of players and player number

            for pos in self.p_pos:
                self.prog.stdin.write(f"{pos[0]} {pos[1]}\n".encode()) # Initial positions of each player

    async def lose_game(self):
        await super().lose_game()

    async def ask_move(
        self, debug: bool = True, **kwargs
    ) -> tuple[tuple[int, int] | None, str | None]:
        # sourcery skip: remove-unnecessary-else, swap-if-else-branches
        await super().ask_move(**kwargs)
        try:
            while True:
                if not self.prog.stdout:
                    return None, "communication failed"
                progInput = await asyncio.wait_for(
                    self.prog.stdout.readuntil(), TIMEOUT_LENGTH
                )

                if not isinstance(progInput, bytes):
                    continue
                progInput = progInput.decode().strip()

                if progInput.startswith("Traceback"):
                    output = StringIO()
                    if debug:
                        print(file=output)
                        print(progInput, file=output)
                        progInput = self.prog.stdout.read()
                        if isinstance(progInput, bytes):
                            print(progInput.decode(), file=output)
                        await Player.print(output)
                    return None, "error"

                if progInput.startswith(">"):
                    # Any bot can write lines starting with ">" to debug in local.
                    # It is recommended to remove any debug before playing
                    # against other players to avoid reverse engineering!
                    if debug:
                        await Player.print(f"{self} {progInput}")
                else:
                    break

            await Player.print(f"{self}'s move : {progInput}")

        except (asyncio.TimeoutError, asyncio.exceptions.IncompleteReadError) as e:
            await Player.print(f"AI did not respond in time (over {TIMEOUT_LENGTH}s)")
            return None, "timeout"

        return await AI.sanithize(progInput, **kwargs)

    async def tell_move(self, move: ValidInput):
        if self.prog.stdin:
            # The AIs should keep track of who's playing themselves.
            self.prog.stdin.write(f"{move}\n".encode())

    async def stop_game(self):
        with contextlib.suppress(ProcessLookupError):
            self.prog.terminate()
            try:
                await asyncio.wait_for(self.prog.wait(), timeout=10)
            except asyncio.TimeoutError:
                self.prog.kill()
                await self.prog.wait()


# Game functions

class MoveError(Exception):
    pass


class Board:
    def __init__(self, w: int, h: int, p_pos: list[int], growth_rate: int):
        self.w = w
        self.h = h
        self.turn = 0
        self.growth_rate = growth_rate

        self.bodies = [deque([pos]) for pos in p_pos]
        self.grid = [[0 for _ in range(w)] for _ in range(h)]

        for i, pos in enumerate(p_pos):
            x, y = pos
            self.grid[y][x] = i + 1

    def get_delta(self, move: str) -> tuple[int, int]:
        moves = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        if out := moves.get(move):
            return out
        else:
            raise ValueError("Invalid move")

    def move(self, i: int, move: ValidMove):
        try:
            dx, dy = self.get_delta(move)
        except ValueError as e:
            raise ValueError("Invalid move") from e

        if not (0 <= i < len(self.bodies)):
            raise ValueError("Invalid player")

        body = self.bodies[i]
        x, y = body[0]

        nx, ny = x + dx, y + dy
        if not (0 <= nx < self.w and 0 <= ny < self.h):
            raise MoveError("Out of bounds")

        if self.grid[ny][nx] != 0:
            raise MoveError("Collision")

        self.grid[ny][nx] = i + 1
        body.appendleft((nx, ny))

        if (self.turn // len(self.bodies)) % self.growth_rate != 0:
            tail = body.pop()
            self.grid[tail[1]][tail[0]] = 0

    def display(self):
        out = StringIO()
        for y in range(self.h):
            for x in range(self.w):
                out.write(EMOJI_COLORS[self.grid[y][x]])
            out.write("\n")
        out.seek(0)
        return out


async def game(
    players: list[Human | AI], p_pos: list[int], w: int, h: int, growth_rate: int, debug: bool, **kwargs
) -> tuple[list[Human | AI], Human | AI | None, dict]:
    """
    The function handling all the game logic.
    Once again, you can add as many kwargs as you need.
    Note that you can return anything you need that will be treated in `main()` after the specified args.

    Args:
        players (`list[Human | AI]`): The list of players
        debug (`bool`): _description_

    Returns:
        `tuple[list[Human | AI], Human | AI | None, dict, ...]`: A whole bunch of game data to help display and judge the result
    """

    nb_players = len(players)
    alive_players = nb_players
    errors = {}
    starters = (
        player.start_game(turn, w, h, p_pos) for turn, player in enumerate(players)
    )
    await asyncio.gather(*starters)
    turn = 0
    winner = None

    board = Board(w, h, p_pos, growth_rate)

    # game loop
    while alive_players >= 2:
        i = turn % nb_players
        player = players[i]

        if not player.alive:
            await player.tell_other_players(players, f"death {i}")

        else:
            await Player.print(f"**Turn {turn} **: player {player}")
            await Player.print(board.display())  # Render the grid for the player here

            # player input
            user_input, error = None, None
            while not user_input:
                user_input, error = await player.ask_move(debug, **kwargs)
                if isinstance(player, AI) or error in ("user interrupt", "timeout"):
                    break

            # saving eventual error
            if not user_input:
                await player.lose_game()
                errors[player] = error
                player.alive = False
                alive_players -= 1
                await player.tell_other_players(players, f"death {i}")

            else:
                # Apply the user_input to the game here, it already went through sanithization so it is a ValidMove
                # You'll also need to convert to a ValidInput to notify all the AIs of the played move
                try:
                    board.move(i, user_input)
                except MoveError as e:
                    await player.lose_game()
                    errors[player] = error
                    player.alive = False
                    alive_players -= 1
                    await player.tell_other_players(players, f"death {i}")
                except ValueError as e:
                    raise  # Should not happen

                await player.tell_other_players(players, f"move {i} {user_input}")

                # Check for wins or draw here.
                # Any end must break the `while alive_players >= 2`.
                # Do this step early to avoid an infinite loop!

                # Nothing to do here for snake

        turn += 1
        board.turn = turn

    if alive_players == 1:
        winner = next(player for player in players if player.alive)

    enders = (player.stop_game() for player in players if isinstance(player, AI))
    await asyncio.gather(*enders)

    return players, winner, errors


async def main(
    raw_args: str = None,
    ifunc: InputFunction = None,
    ofunc: OutputFunction = None,
    discord=False,
):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prog", nargs="*", help="AI program to play the game ('user' to play yourself)"
    )
    parser.add_argument(
        "-g",
        "--grid",
        type=int,
        nargs=2,
        default=[WIDTH, HEIGHT],
        metavar=("WIDTH", "HEIGHT"),
        help="size of the grid",
    )
    parser.add_argument(
        "-p",
        "--players",
        type=int,
        default=2,
        metavar="NB_PLAYERS",
        help="number of players (if more players than programs are provided, the other ones will be filled as real players)",
    )
    parser.add_argument(
        "-P",
        "--pos",
        type=int,
        nargs=2,
        action="append",
        metavar=("X", "Y"),
        help="initial position of a player",
    )
    parser.add_argument(
        "-s", "--silent", action="store_true", help="only show the result of the game"
    )
    parser.add_argument(
        "-n",
        "--nodebug",
        action="store_true",
        help="do not print the debug output of the programs",
    )

    parser.add_argument(
        "-G",
        "--growth_rate",
        type=int,
        default=5,
        metavar="GROWTH_RATE",
        help="the snake will grow by one cell every GROWTH_RATE moves",
    )


    args = parser.parse_args(raw_args)
    width, height = args.grid
    p_pos = args.pos
    growth_rate = args.growth_rate

    Player.ofunc = ofunc
    players = []
    ai_only = True
    pattern = re.compile(r"^\<\@[0-9]{18}\>$")
    kwargs = {"growth_rate": growth_rate}
    for i, name in enumerate(args.prog):
        if name == "user":
            players.append(Human(i, **kwargs)) 
            ai_only = False
        elif pattern.match(name):
            players.append(
                Human(i, name, ifunc, **kwargs)
            )
            ai_only = False
        else:
            players.append(
                AI(i, name, discord, **kwargs)
            )

    while len(players) < args.players:
        players.append(Human(len(players), **kwargs))
        ai_only = False

    if p_pos is None:
        p_pos = [(0, 0), (width - 1, height - 1), (0, height - 1), (width - 1, 0)][
            : len(players)
        ]

    origin_stdout = sys.stdout
    if args.silent:
        if not ai_only:
            output = StringIO("Game cannot be silent since humans are playing")
            tmp = output.getvalue()
            await Player.print(output)
            raise ValueError(tmp)

        if discord:
            Player.ofunc = None
        else:
            sys.stdout = open(os.devnull, "w")

    players, winner, errors = await game(
        players, p_pos, width, height, growth_rate, not args.nodebug
    )

    if args.silent:
        sys.stdout = origin_stdout
        Player.ofunc = ofunc
    else:
        # print whatever you want when not silent, often the final board
        ...
    ...  # another place to display things

    return (
        players,
        winner,
        errors,
    )


if __name__ == "__main__":
    asyncio.run(main())
