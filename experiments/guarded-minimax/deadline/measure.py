import sys, os, subprocess, time, select, random, csv, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root))
from snake import Board, MoveError
parser = argparse.ArgumentParser(description='Replay archived and stress layouts with a one-second watchdog on Linux.')
parser.add_argument('output', type=Path)
parser.add_argument('--limit', type=int)
args = parser.parse_args()
d = args.output
d.mkdir(parents=True, exist_ok=False)
source = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader((source / 'final-guard-comparison.csv').open()))
G = []
for r in rows:
    G.append(('final', int(r['width']), int(r['height']), int(r['growth']), int(r['seat']) - 1, *[int(r[k]) for k in ('p1x', 'p1y', 'p2x', 'p2y')], int(r['minimax_win']), int(r['minimax_turns'])))
rng = random.Random(3738)
for w, h, m in [(10, 10, 5), (14, 14, 3), (14, 14, 10)]:
    for j in range(12):
        if j == 0:
            a, b = ((0, 0), (w - 1, h - 1))
        elif j == 1:
            a, b = ((w // 2, h // 2), (w // 2 + 1, h // 2))
        else:
            a, b = rng.sample([(x, y) for x in range(w) for y in range(h)], 2)
        for seat in range(2):
            G.append(('stress', w, h, m, seat, *a, *b, -1, -1))

def readmove(p, start):
    """Charge all partial reads to one wall-clock deadline starting before input delivery."""
    deadline = start + 1000000000
    buf = b''
    while True:
        remaining = deadline - time.perf_counter_ns()
        if remaining <= 0 or not select.select([p.stdout], [], [], max(remaining, 0) / 1000000000.0)[0]:
            return ('deadline', '', (time.perf_counter_ns() - start) / 1000000.0)
        part = os.read(p.stdout.fileno(), 64)
        ns = time.perf_counter_ns()
        if ns >= deadline:
            return ('deadline', '', (ns - start) / 1000000.0)
        if not part:
            return ('eof', '', (ns - start) / 1000000.0)
        buf += part
        if b'\n' in buf:
            s = buf.decode().strip()
            return ('move' if s in ('left', 'up', 'right', 'down') else 'invalid', s, (ns - start) / 1000000.0)

def run(i):
    """Play both seats in separate processes and record every timed protocol response."""
    group, w, h, m, seat, x, y, a, b, ew, et = G[i]
    board = Board(w, h, [(x, y), (a, b)], m)
    procs = [subprocess.Popen([root / 'build/experiments/DeadlinePlayer', 'guard' if s == seat else 'opponent'], cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0) for s in (0, 1)]
    setup = lambda s: f'{w} {h}\n{m}\n2 {s + 1}\n{x} {y}\n{a} {b}\n'.encode()
    events = []
    turn = 0
    try:
        procs[1].stdin.write(setup(1))
        msg = setup(0)
        while True:
            who = turn % 2
            p = procs[who]
            board.turn = turn
            px, py = board.bodies[who][0]
            legal = any((0 <= px + dx < w and 0 <= py + dy < h and (board.grid[py + dy][px + dx] == 0) for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))))
            start = time.perf_counter_ns()
            p.stdin.write(msg)
            status, action, ms = readmove(p, start)
            if status == 'eof' and (not legal) and (p.wait() == 0):
                status = 'trapped_eof'
            events.append((turn, who == seat, status, ms))
            if status != 'move':
                break
            try:
                board.move(who, action)
            except MoveError:
                status = 'collision'
                break
            msg = f'move {who} {action}\n'.encode()
            turn += 1
        return (i, int(who != seat), turn + 1, status, events)
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
            p.wait()

def init(q):
    """Pin each worker and its two child players to one physical core."""
    cpu = q.get()
    os.sched_setaffinity(0, {cpu})
if __name__ == '__main__':
    sub = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(1.2);print("up",flush=True)'], stdout=subprocess.PIPE)
    try:
        assert readmove(sub, time.perf_counter_ns())[0] == 'deadline'
    finally:
        sub.kill()
        sub.wait()
    cores = {}
    for c in sorted(os.sched_getaffinity(0)):
        top = Path(f'/sys/devices/system/cpu/cpu{c}/topology')
        k = ((top / 'physical_package_id').read_text(), (top / 'core_id').read_text())
        cores.setdefault(k, c)
    cpus = list(cores.values())[:8]
    ctx = get_context('fork')
    q = ctx.Queue()
    for c in cpus:
        q.put(c)
    print('cpus', cpus, 'negative timeout control passed', flush=True)
    order = list(range(len(G)))[:args.limit]
    rng.shuffle(order)
    with (d / 'games.csv').open('w') as gf, (d / 'turns.csv').open('w') as tf, ProcessPoolExecutor(len(cpus), mp_context=ctx, initializer=init, initargs=(q,)) as pool:
        gw = csv.writer(gf)
        tw = csv.writer(tf)
        gw.writerow('game group w h growth seat win turns end old_win old_turns'.split())
        tw.writerow('game turn guarded status ms'.split())
        for j, (i, win, n, status, events) in enumerate(pool.map(run, order, chunksize=1)):
            group, w, h, m, seat, *_ = G[i]
            gw.writerow((i, group, w, h, m, seat + 1, win, n, status, *G[i][-2:]))
            tw.writerows(((i, t, int(guard), s, round(ms, 4)) for t, guard, s, ms in events))
            if j % 100 == 0:
                tf.flush()
                gf.flush()
                print(j + 1, 'games recorded', flush=True)
    print('complete', flush=True)
