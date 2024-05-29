import copy
import datetime
import logging
import multiprocessing
import os
import socket
import subprocess
import zlib
from functools import wraps

from lib import config
import cachetools.func
import redis
import cv2
import pickle
import time


LOGGER = logging.getLogger()
rd_client = redis.StrictRedis(host=config['master_node_ip'])


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        LOGGER.debug(f'func: {f.__name__} args:[{args}, {kw}] took: {te-ts} sec')
        return result
    return wrap


def set_frame_and_face_id(face_id, frame_id):
    rd_client.set(face_id, frame_id)


def get_frame_from_face_id(face_id):
    frame_id = rd_client.get(face_id)
    return 0 if frame_id is None else int(frame_id)


@cachetools.func.ttl_cache(maxsize=1, ttl=0.1)
def get_now_approx() -> datetime.datetime:
    return datetime.datetime.now()


@cachetools.func.lru_cache(maxsize=255)
def create_full_default_left():
    single_left_im = create_default_left_image()
    im_left = copy.deepcopy(single_left_im)
    for _ in range(1, 3):
        im_left = cv2.vconcat([im_left, single_left_im])

    return im_left


@cachetools.func.lru_cache(maxsize=255)
def create_default_left_image():
    im = cv2.imread("datasets/default.png")
    im_final = cv2.hconcat([im, im])
    return im_final


def save_frame(frame):
    frame_id = int(time.time() * 1000)
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rd_client.set(frame_id, zlib.compress(pickle.dumps(frame)))

    LOGGER.info(f"Write frame {frame_id}")
    return frame_id


def load_image(frame_id):
    im = rd_client.get(frame_id)
    if im is not None:
        frame = pickle.loads(zlib.decompress(im))
        frame = cv2.resize(frame, (0, 0), fx=2, fy=2)
        return frame
    else:
        return None


def delete_images():
    total_images = rd_client.flushdb()
    LOGGER.info(f"========== Delete {total_images} images.")


@cachetools.func.ttl_cache(maxsize=1, ttl=30)
def get_total_memory_usage(server):
    client = redis.Redis(host=server)
    try:
        total_memory = 0
        for key in client.scan_iter():
            total_memory += client.memory_usage(key)
        return round(total_memory / (1024 * 1024), 2)  # Convert bytes to MB
    finally:
        client.close()


@cachetools.func.ttl_cache(maxsize=1, ttl=10)
def get_dbsize(server):
    client = redis.Redis(host=server)
    try:
        return client.dbsize()
    finally:
        client.close()


def pinger(job_q, results_q):
    """
    Do Ping
    :param job_q:
    :param results_q:
    :return:
    """
    DEVNULL = open(os.devnull, 'w')
    while True:

        ip = job_q.get()

        if ip is None:
            break

        try:
            subprocess.check_call(['ping', '-c1', '-W0.2', ip], stdout=DEVNULL)
            results_q.put(ip)
        except:
            pass


def get_my_ip():
    """
    Find my IP address
    :return:
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip


def map_friend_networks(pool_size=255):
    """
    Maps the network
    :param pool_size: amount of parallel ping processes
    :return: list of valid ip addresses
    """

    ip_list = list()
    my_ip = get_my_ip()
    # get my IP and compose a base like 192.168.1.xxx
    ip_parts = my_ip.split('.')
    base_ip = ip_parts[0] + '.' + ip_parts[1] + '.' + ip_parts[2] + '.'

    # prepare the jobs queue
    jobs = multiprocessing.Queue()
    results = multiprocessing.Queue()

    pool = [multiprocessing.Process(target=pinger, args=(jobs, results)) for i in range(pool_size)]

    for p in pool:
        p.start()

    # cue hte ping processes
    for i in range(2, 255):
        jobs.put(base_ip + '{0}'.format(i))

    for p in pool:
        jobs.put(None)

    for p in pool:
        p.join()

    # collect he results
    while not results.empty():
        ip = results.get()
        ip_list.append(ip)

    return [x for x in ip_list if x != my_ip]


def check_port(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex((ip, port))
    sock.close()
    return result == 0


@timing
def find_master_node():
    valid_ports = [19530, 6379]
    friend_ips = map_friend_networks()
    for friend_ip in friend_ips:
        if check_port(friend_ip, valid_ports[0]) and check_port(friend_ip, valid_ports[1]):
            LOGGER.warning(f"Master node is {friend_ip}")
            return friend_ip

    LOGGER.warning("Master node not found, set default to 127.0.0.1")
    return "127.0.0.1"


