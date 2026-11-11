#!/usr/bin/env python3
"""Smart Classroom browser launcher for WSL and Raspberry Pi."""

import argparse
from collections import deque
import fcntl
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import webbrowser


def stop(process):
    if process is None or process.poll() is not None:
        return
    for sig, seconds in ((signal.SIGINT, 3), (signal.SIGTERM, 2), (signal.SIGKILL, 2)):
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=seconds)
            return
        except ProcessLookupError:
            process.poll()
            return
        except subprocess.TimeoutExpired:
            continue


def run(args, workspace, logs):
    executables = workspace / "install/smartclass_cv/lib/smartclass_cv"
    commands = {
        name: [str(executables / name)]
        for name in ("frame", "detection", "face", "expression", "recognition", "activity", "gesture")
    }
    commands["merge"] = [str(executables / "merge_test")]
    commands["viewer"] = [str(workspace / "merge_viewer.py")]

    def set_params(name, params):
        if params:
            commands[name].append("--ros-args")
        for key, value in params.items():
            commands[name].extend(["-p", f"{key}:={value}"])

    set_params("frame", {key: value for key, value in {
        "source": args.source, "decimate_n": args.decimate_n,
    }.items() if value is not None})
    offload_nodes = {"expression", "recognition"}
    server_url = args.server_url or "http://103.28.161.71:8085"
    default_mode = "remote" if args.server_url else "local"
    if args.work_dir:
        set_params("recognition", {"work_dir": args.work_dir.expanduser().resolve()})

    optional = {"detection", "activity", "gesture"}
    selected = [name for name in commands if args.full or name not in optional]
    processes = {}
    modes = {}

    def start(name, mode):
        process = processes.get(name)
        if process is not None and process.poll() is None:
            return None
        recognition = processes.get("recognition")
        expression = processes.get("expression")
        if name == "expression" and (
            mode in {"server_match", "fikri_api"} or (
                recognition is not None and recognition.poll() is None
                and modes.get("recognition") in {"server_match", "fikri_api"}
            )
        ):
            return "Expression sudah ditangani oleh Recognition pada mode Server match."
        if name == "recognition" and mode in {"server_match", "fikri_api"} and (
            expression is not None and expression.poll() is None
        ):
            return "Stop Expression sebelum menyalakan Recognition dalam mode Server match."

        command = commands[name].copy()
        if name in offload_nodes and mode in {"remote", "server_match", "fikri_api"}:
            if "--ros-args" not in command:
                command.append("--ros-args")
            command.extend(["-p", f"mode:={mode}", "-p", f"server_url:={server_url}",
                            "-p", "timeout_s:=10.0"])
        if not Path(command[0]).is_file():
            return f"{name}: file tidak ditemukan: {command[0]}"
        try:
            with (logs / f"{name}.log").open("w") as output:
                process = subprocess.Popen(
                    [sys.executable, "-u", *command], cwd=workspace,
                    stdin=subprocess.DEVNULL, stdout=output,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
        except OSError as error:
            return f"{name}: gagal dijalankan: {error}"
        processes[name] = process
        modes[name] = mode if name in offload_nodes else None

        return None

    address = f"127.0.0.1:{args.port}"
    origin = f"http://{address}"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_body(self, body, content_type="application/json", status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, data, status=200):
            self.send_body(json.dumps(data).encode(), status=status)

        def do_GET(self):
            if self.headers.get("Host") != address:
                self.send_error(403)
                return
            if self.path == "/":
                self.send_body(PAGE.replace("__DEFAULT_MODE__", default_mode).encode(), "text/html; charset=utf-8")
            elif self.path == "/status":
                rows = []
                for name in commands:
                    process = processes.get(name)
                    code = process.poll() if process else None
                    rows.append({"name": name, "running": process is not None and code is None,
                                 "code": code, "pid": process.pid if process else None,
                                 "included": name in selected, "mode": modes.get(name)})
                self.send_json(rows)
            elif self.path.startswith("/log/") and (name := self.path[5:]) in commands:
                path = logs / f"{name}.log"
                content = "Belum ada log."
                if path.is_file():
                    with path.open("rb") as log:
                        log.seek(max(0, path.stat().st_size - 16384))
                        content = "".join(deque(log.read().decode(errors="replace").splitlines(keepends=True), maxlen=30))
                self.send_json({"log": content})
            else:
                self.send_error(404)

        def do_POST(self):
            if self.headers.get("Host") != address or self.headers.get("Origin") != origin:
                self.send_error(403)
                return
            path, _, mode = self.path.partition("?")
            mode = mode or default_mode
            if mode not in {"local", "remote", "server_match", "fikri_api"}:
                self.send_error(400)
                return
            parts = path.strip("/").split("/")
            if len(parts) != 2 or parts[0] not in {"start", "stop"}:
                self.send_error(404)
                return
            action, name = parts
            if name == "all":
                targets = selected if action == "start" else reversed(commands)
                if action == "start" and mode in {"server_match", "fikri_api"}:
                    targets = [target for target in targets if target != "expression"]
            elif name in commands:
                targets = [name]
            else:
                self.send_error(404)
                return
            errors = []
            for target in targets:
                if action == "start":
                    error = start(target, mode)
                    if error:
                        errors.append(error)
                else:
                    stop(processes.get(target))
                    process = processes.get(target)
                    if process is not None and process.poll() is None:
                        errors.append(f"{target}: proses masih aktif")
            self.send_json({"errors": errors})

    try:
        with HTTPServer(("127.0.0.1", args.port), Handler) as server:
            print(f"Buka {origin} · Ctrl+C untuk keluar", flush=True)
            if not args.no_browser:
                if shutil.which("cmd.exe"):
                    subprocess.Popen(["cmd.exe", "/c", "start", "", origin],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    webbrowser.open(origin)
            server.serve_forever()
    except KeyboardInterrupt:
        print("\nMenutup launcher.")
    finally:
        for name in reversed(commands):
            stop(processes.get(name))


PAGE = """<!doctype html>
<html lang="id"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smart Classroom · Launcher</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#101922;color:#eef4f8;font:16px system-ui,sans-serif}
main{max-width:920px;margin:auto;padding:48px 20px}h1{font-size:clamp(30px,6vw,52px);margin:8px 0}
p,small{color:#b0c4d2} .eyebrow{color:#85e7cc;letter-spacing:.16em;font-size:12px;font-weight:750}
.toolbar,.actions{display:flex;gap:8px;flex-wrap:wrap} .toolbar{margin:24px 0}
button{font:inherit;font-weight:650;border:1px solid #506879;border-radius:8px;background:#253847;color:#eef4f8;padding:8px 13px;cursor:pointer}
button.primary{background:#85e7cc;color:#102e27;border-color:#85e7cc}button:disabled{opacity:.45;cursor:default}
select{font:inherit;background:#253847;color:#eef4f8;padding:8px;border:1px solid #506879;border-radius:8px}
.row{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;background:#1b2935;padding:13px 16px;margin:6px 0;border-radius:8px}
.name{min-width:115px;text-transform:capitalize;font-weight:700}.status{flex:1;color:#b0c4d2}.active{color:#85e7cc}.error{color:#ffaeaa;white-space:pre-line}
pre{background:#0a1117;padding:16px;white-space:pre-wrap;overflow-wrap:anywhere;border-radius:8px}
</style>
<main>
<div class="eyebrow">SMART CLASSROOM</div><h1>Launcher ROS 2</h1>
<p>Jalankan node dan pantau status proses dari satu layar.</p>
<label for="mode">Mode Expression &amp; Recognition</label>
<select id="mode" data-default="__DEFAULT_MODE__">
<option value="local">Stay on local</option><option value="remote">Offload to server</option>
<option value="server_match">Server match (nama + ekspresi)</option>
<option value="fikri_api">Fikri API (Recognition baru)</option>
</select>
<p><small>Stop lalu Start untuk mengganti mode. Pada Server match, Recognition mengirim hasil nama dan ekspresi ke Merge.</small></p>
<div class="toolbar"><button class="primary" id="start">Start semua</button><button id="stop">Stop semua</button></div>
<p class="error" id="error" role="status"></p><div id="nodes"></div>
<section hidden id="log-panel"><h2 id="log-title"></h2><pre id="log-output"></pre></section>
</main>
<script>
const byId = id => document.getElementById(id);
const rows = new Map();
byId("mode").value = byId("mode").dataset.default;
let busy = false;
function button(label, onClick) {
  const element = document.createElement('button');
  element.textContent = label;
  element.onclick = onClick;
  return element;
}
async function request(path, method = 'GET') {
  const response = await fetch(path, {method, cache: 'no-store'});
  if (!response.ok) throw new Error('Launcher merespons dengan HTTP ' + response.status);
  return response.json();
}
async function refresh() {
  for (const node of await request('/status')) {
    if (!rows.has(node.name)) {
      const row = document.createElement('div'); row.className = 'row';
      const name = document.createElement('span'); name.className = 'name'; name.textContent = node.name;
      const status = document.createElement('span');
      const actions = document.createElement('div'); actions.className = 'actions';
      const start = button('Start', () => control('start', node.name));
      const stop = button('Stop', () => control('stop', node.name));
      actions.append(start, stop, button('Log', () => showLog(node.name)));
      row.append(name, status, actions); byId('nodes').append(row);
      rows.set(node.name, {status, start, stop});
    }
    const row = rows.get(node.name);
    row.status.className = node.running ? 'status active' : 'status';
    row.status.textContent = node.running ? 'Aktif · PID ' + node.pid :
      node.pid ? 'Berhenti · exit ' + node.code : 'Belum dijalankan';
    const modeNames = {local: 'Local', remote: 'Offload', server_match: 'Server match', fikri_api: 'Fikri API'};
    if (node.mode) row.status.textContent += ' · ' + modeNames[node.mode];
    if (!node.included) row.status.textContent += ' · Opsional';
    row.start.disabled = node.running;
    row.stop.disabled = !node.running;
  }
}
async function control(action, name) {
  if (busy) return;
  busy = true;
  byId('error').textContent = '';
  try {
    const result = await request('/' + action + '/' + name + '?' + byId('mode').value, 'POST');
    byId('error').textContent = result.errors.join(String.fromCharCode(10));
    await refresh();
  } catch (error) { byId('error').textContent = error.message; }
  finally { busy = false; }
}
async function showLog(name) {
  try {
    const data = await request('/log/' + name);
    byId('log-title').textContent = 'Log ' + name;
    byId('log-output').textContent = data.log;
    byId('log-panel').hidden = false;
  } catch (error) { byId('error').textContent = error.message; }
}
byId('start').onclick = () => control('start', 'all');
byId('stop').onclick = () => control('stop', 'all');
async function monitor() {
  try { if (!busy) await refresh(); }
  catch (error) { byId('error').textContent = 'Koneksi ke launcher terputus.'; }
  finally { setTimeout(monitor, 1500); }
}
monitor();
</script></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--source")
    parser.add_argument("--decimate-n", type=int)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.decimate_n is not None and args.decimate_n < 1:
        parser.error("--decimate-n must be at least 1")

    workspace = Path(__file__).resolve().parent
    logs = workspace / "log/smartclass_launcher"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "launcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, "Launcher lain sudah aktif di workspace ini.\n")
        run(args, workspace, logs)


if __name__ == "__main__":
    main()
