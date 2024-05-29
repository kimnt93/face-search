import logging
import os
import psutil
from flask import Flask, render_template, jsonify
from lib.indexer import MilvusIndexer
import subprocess
import signal
import time
import shutil

from lib.utils import delete_images, get_total_memory_usage, get_dbsize, find_master_node

app = Flask(__name__)
curr_path = os.path.dirname(os.path.abspath(__file__))
LOGGER = logging.getLogger()


def kill_processes(pattern):
    for process in psutil.process_iter(attrs=['pid', 'cmdline']):
        try:
            cmdline = process.info['cmdline']
            if cmdline and pattern in ' '.join(cmdline):
                pid = process.info['pid']
                print(f"Killing process with PID {pid}: {' '.join(cmdline)}")
                process.send_signal(signal.SIGTERM)  # Terminate the process
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/ping')
def ping():
    server = os.environ.get("master_node_ip", "127.0.0.1")
    LOGGER.info(f"Ping server {server}")
    is_available = MilvusIndexer.ping(server)
    memory_usage = get_total_memory_usage(server)
    dbsize = get_dbsize(server)
    if not is_available:
        kill_processes("python face_recognize.py")
    return jsonify({'available': is_available, "memory_usage": memory_usage, "dbsize": dbsize, "server": server})


@app.route('/join')
def join():
    master_node_ip = find_master_node()
    LOGGER.info(f"Join server {master_node_ip}")
    os.environ['master_node_ip'] = master_node_ip
    if master_node_ip == "127.0.0.1":
        return jsonify({'success': False, 'message': 'Not Found!'})
    else:
        subprocess.run(['python', 'set_node.py', '--master', master_node_ip])
        time.sleep(1)
        subprocess.run(['python', 'face_recognize.py'])
        return jsonify({'success': True})


@app.route('/start')
def start():
    kill_processes("python face_recognize.py")
    time.sleep(2)
    subprocess.run(['python', 'set_node.py', '--master', "127.0.0.1"])
    os.environ['master_node_ip'] = "127.0.0.1"
    subprocess.run(['python', 'face_recognize.py'])
    return jsonify({'success': True})


@app.route('/clear')
def clear():
    kill_processes("python face_recognize.py")
    time.sleep(2)
    std = subprocess.run(['python', 'drop_data.py'], capture_output=True, text=True)
    delete_images()
    return jsonify({'stdout': std.stdout, 'stderr': std.stderr})


@app.route('/stop')
def stop():
    kill_processes("python face_recognize.py")
    time.sleep(2)
    return jsonify({'success': True})


if __name__ == '__main__':
    app.run()
