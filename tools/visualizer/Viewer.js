/* Viewer - Draw live snapshots and saved replays without recomputing game rules.
 * Copyright (C) 2026 Enzo DOS ANJOS.
 */
const colors = ["#5bd1c1", "#f0b86c", "#a991ed", "#ed87a7"];
const arrows = {left: "←", up: "↑", right: "→", down: "↓"};
const element = (id) => document.getElementById(id);
const canvas = element("board");
const context = canvas.getContext("2d");
let replay = null;
let position = 0;
let following = true;
let playing = false;
let localReplay = false;
let pending = null;
let loadedGeneration = 0;

function drawBoard(frame)
// Algorithm : Draw ordered body paths, tail rings and numbered heads in stable player colors.
{
    const cell = Math.min(900 / replay.width, 650 / replay.height);
    canvas.width = replay.width * cell + 48;
    canvas.height = replay.height * cell + 48;
    context.fillStyle = "#0b1423";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.save();
    context.translate(24, 24);
    context.strokeStyle = "#203047";
    context.lineWidth = 1;
    for (let y = 0; y <= replay.height; y++)
    {
        context.beginPath();
        context.moveTo(0, y * cell);
        context.lineTo(replay.width * cell, y * cell);
        context.stroke();
    }
    for (let x = 0; x <= replay.width; x++)
    {
        context.beginPath();
        context.moveTo(x * cell, 0);
        context.lineTo(x * cell, replay.height * cell);
        context.stroke();
    }
    for (const player of frame.players)
    {
        const points = player.body.map(([x, y]) => [(x + 0.5) * cell, (y + 0.5) * cell]);
        const color = colors[(player.id - 1) % colors.length];
        context.globalAlpha = player.alive ? 1 : 0.38;
        context.strokeStyle = color;
        context.fillStyle = color;
        context.lineWidth = cell * 0.52;
        context.lineCap = "round";
        context.lineJoin = "round";
        context.beginPath();
        context.moveTo(...points[0]);
        for (const point of points.slice(1))
        {
            context.lineTo(...point);
        }
        context.stroke();
        const tail = points[points.length - 1];
        context.beginPath();
        context.arc(...tail, cell * 0.21, 0, Math.PI * 2);
        context.fillStyle = "#0b1423";
        context.fill();
        context.lineWidth = Math.max(2, cell * 0.05);
        context.stroke();
        const [hx, hy] = points[0];
        let angle = 0;
        if (points.length > 1)
        {
            angle = Math.atan2(hy - points[1][1], hx - points[1][0]);
        }
        context.save();
        context.translate(hx, hy);
        context.rotate(angle);
        context.beginPath();
        if (points.length === 1)
        {
            context.arc(0, 0, cell * 0.32, 0, Math.PI * 2);
        }
        else
        {
            context.moveTo(cell * 0.43, 0);
            context.lineTo(-cell * 0.3, -cell * 0.32);
            context.lineTo(-cell * 0.3, cell * 0.32);
            context.closePath();
        }
        context.fillStyle = color;
        context.fill();
        context.restore();
        context.fillStyle = "#09111d";
        context.font = `bold ${Math.max(10, cell * 0.28)}px system-ui`;
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillText(player.alive ? player.id : "×", hx - cell * 0.04, hy);
    }
    context.restore();
    context.globalAlpha = 1;
} //----- end of drawBoard

function render()
// Algorithm : Present the selected snapshot and its recorded move metrics without inferred AI scores.
{
    if (!replay || !replay.frames.length)
    {
        if (replay?.error)
        {
            element("message").textContent = `Session stopped: ${replay.error}`;
        }
        return;
    }
    const frame = replay.frames[position];
    drawBoard(frame);
    element("clock").textContent = `Turn ${frame.turn} · Round ${frame.round}`;
    element("growth").textContent = frame.round % replay.growth === 0 ? "Growth round" : `Growth in ${replay.growth - frame.round % replay.growth} rounds`;
    element("timeline").max = replay.frames.length - 1;
    element("timeline").value = position;
    element("play").textContent = playing ? "Pause" : "Play";
    element("players").replaceChildren();
    for (const player of frame.players)
    {
        const card = document.createElement("section");
        card.className = "player";
        card.style.setProperty("--player", colors[(player.id - 1) % colors.length]);
        const title = document.createElement("h2");
        title.textContent = `${replay.names[player.id - 1]} · ${player.alive ? "Alive" : "Eliminated"}`;
        card.append(title);
        const metrics = document.createElement("div");
        metrics.className = "metrics";
        for (const [label, value] of [["Length", player.body.length], ["Legal moves", player.legal.length], ["Space", player.space]])
        {
            const item = document.createElement("div");
            const number = document.createElement("strong");
            number.textContent = value;
            const caption = document.createElement("span");
            caption.textContent = label;
            item.append(number, caption);
            metrics.append(item);
        }
        card.append(metrics);
        const coordinates = document.createElement("p");
        coordinates.className = "position";
        coordinates.textContent = `Head (${player.body[0]}) · Tail (${player.body.at(-1)}) · ${player.legal.join(" / ") || "No legal moves"}`;
        card.append(coordinates);
        element("players").append(card);
    }
    const event = frame.event;
    element("action").textContent = event ? `P${event.player} ${arrows[event.action] || "·"} ${event.action || "No move"}` : "Initial positions";
    element("detail").textContent = event ? `${event.outcome}\nOptions before: ${event.legalBefore.join(" / ") || "none"}\nLength change: ${event.lengthChange >= 0 ? "+" : ""}${event.lengthChange}\nResponse: ${event.decisionMs === null ? "—" : event.decisionMs.toFixed(1) + " ms"}` : "Each player starts with one cell: head and tail coincide.";
    if (event?.decision)
    {
        element("detail").textContent += `\n${event.decision.mode}\n${event.decision.reason}`;
    }
    const survivors = frame.players.filter((player) => player.alive);
    if (replay.error)
    {
        element("message").textContent = `Session stopped: ${replay.error}`;
    }
    else
    {
        element("message").textContent = `${localReplay ? "Saved replay" : following ? "Following live" : "Reviewing history"} · Frame ${position + 1}/${replay.frames.length}` + (replay.finished && position === replay.frames.length - 1 ? ` · ${survivors.length === 1 ? "Winner: P" + survivors[0].id : "Game complete"}` : "");
    }
    const canMove = !localReplay && pending && following && position === replay.frames.length - 1;
    element("human").hidden = !canMove;
    element("directions").replaceChildren();
    if (canMove)
    {
        element("human").querySelector("h2").textContent = `Your turn · P${pending.player}`;
        for (const action of Object.keys(arrows))
        {
            const button = document.createElement("button");
            button.textContent = `${arrows[action]} ${action}`;
            button.disabled = !pending.legal.includes(action);
            button.onclick = () => submitMove(action);
            element("directions").append(button);
        }
    }
} //----- end of render

async function submitMove(action)
// Algorithm : Send a legal human action with a turn token so delayed clicks cannot play another turn.
{
    if (!pending || localReplay || !following || !pending.legal.includes(action))
    {
        return;
    }
    const move = {...pending, action};
    pending = null;
    render();
    try
    {
        const response = await fetch("/api/move", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(move)});
        if (!response.ok)
        {
            element("message").textContent = "Move rejected: refresh the current turn.";
        }
    }
    catch (error)
    {
        element("message").textContent = "Connection lost; move was not confirmed.";
    }
} //----- end of submitMove

async function poll()
// Algorithm : Append new live frames while preserving a paused replay position.
{
    if (!localReplay && location.protocol !== "file:")
    {
        const generation = loadedGeneration;
        try
        {
            const response = await fetch(`/api/state?after=${replay ? replay.frames.length : 0}`);
            if (!response.ok)
            {
                throw new Error("Server unavailable");
            }
            const data = await response.json();
            if (generation === loadedGeneration)
            {
                const changed = data.frames.length > 0 || !replay ||
                                data.finished !== replay.finished || data.error !== replay.error ||
                                JSON.stringify(data.pending) !== JSON.stringify(pending);
                const frames = (replay ? replay.frames : []).concat(data.frames);
                replay = {...data, frames};
                pending = data.pending;
                if (following)
                {
                    position = Math.max(0, frames.length - 1);
                }
                if (changed)
                {
                    render();
                }
            }
        }
        catch (error)
        {
            element("message").textContent = "Live connection unavailable. Open a saved JSON replay, or restart the viewer.";
        }
    }
    setTimeout(poll, 180);
} //----- end of poll

function validateReplay(data)
// Algorithm : Reject malformed frames before drawing imported replay data.
{
    if (data.version !== 1 || !Number.isInteger(data.width) || !Number.isInteger(data.height) ||
        data.width < 1 || data.width > 200 || data.height < 1 || data.height > 200 ||
        !Number.isInteger(data.growth) || data.growth < 1 || !Array.isArray(data.names) ||
        data.names.length < 2 || data.names.length > 4 || !Array.isArray(data.frames) || !data.frames.length)
    {
        throw new Error("Unsupported replay format");
    }
    for (const frame of data.frames)
    {
        if (!Number.isInteger(frame.turn) || !Number.isInteger(frame.round) || !Array.isArray(frame.players) || frame.players.length !== data.names.length)
        {
            throw new Error("Invalid snapshot");
        }
        for (const [index, player] of frame.players.entries())
        {
            if (player.id !== index + 1 || !Array.isArray(player.body) || !player.body.length ||
                !Array.isArray(player.legal) || !player.legal.every((action) => action in arrows) || !Number.isFinite(player.space) ||
                !player.body.every((cell) => Array.isArray(cell) && cell.length === 2 && Number.isInteger(cell[0]) && Number.isInteger(cell[1]) && cell[0] >= 0 && cell[0] < data.width && cell[1] >= 0 && cell[1] < data.height))
            {
                throw new Error("Invalid player snapshot");
            }
        }
        if (frame.event && (!Array.isArray(frame.event.legalBefore) || (frame.event.decisionMs !== null && !Number.isFinite(frame.event.decisionMs))))
        {
            throw new Error("Invalid move metrics");
        }
    }
} //----- end of validateReplay

element("file").onchange = async (event) =>
{
    try
    {
        const data = JSON.parse(await event.target.files[0].text());
        validateReplay(data);
        loadedGeneration++;
        localReplay = true;
        following = false;
        playing = false;
        pending = null;
        replay = data;
        position = 0;
        render();
    }
    catch (error)
    {
        element("message").textContent = `Cannot open replay: ${error.message}`;
    }
};
element("timeline").oninput = (event) =>
{
    following = false;
    playing = false;
    position = Number(event.target.value);
    render();
};
element("back").onclick = () =>
{
    following = false;
    playing = false;
    position = Math.max(0, position - 1);
    render();
};
element("next").onclick = () =>
{
    following = false;
    playing = false;
    position = Math.min((replay?.frames.length || 1) - 1, position + 1);
    render();
};
element("play").onclick = () =>
{
    following = false;
    playing = !playing;
    if (position === replay?.frames.length - 1)
    {
        position = 0;
    }
    render();
};
element("live").onclick = () =>
{
    loadedGeneration++;
    localReplay = false;
    replay = null;
    pending = null;
    following = true;
    playing = false;
};
element("save").onclick = () =>
{
    if (!replay)
    {
        return;
    }
    const link = document.createElement("a");
    const url = URL.createObjectURL(new Blob([JSON.stringify(replay)], {type: "application/json"}));
    link.href = url;
    link.download = "snake-replay.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
};
document.addEventListener("keydown", (event) =>
{
    const action = {ArrowLeft: "left", ArrowUp: "up", ArrowRight: "right", ArrowDown: "down"}[event.key];
    if (action && pending && following && !["INPUT", "SELECT"].includes(event.target.tagName))
    {
        event.preventDefault();
        submitMove(action);
    }
});
function tick()
// Algorithm : Advance recorded frames at the selected playback speed and stop at the end.
{
    if (playing && replay)
    {
        position = Math.min(position + 1, replay.frames.length - 1);
        playing = position < replay.frames.length - 1;
        render();
    }
    setTimeout(tick, Number(element("speed").value));
} //----- end of tick
if (location.protocol === "file:")
{
    element("message").textContent = "Open a saved JSON replay to begin.";
}
poll();
tick();
