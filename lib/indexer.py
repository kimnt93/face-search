import pymilvus
import random
import logging
import datetime
import time
from lib import config
from lib.utils import get_frame_from_face_id, set_frame_and_face_id


LOGGER = logging.getLogger()
LOGGER.setLevel("INFO")
TIME_FILTER_SECOND = config['time_filter_second']


class MilvusIndexer:
    def __init__(self, collection_name, ndim, remove=False, threshold=0.8):
        self.threshold = 1 - threshold
        self.collection_name = collection_name
        self.ndim = ndim
        # self.related_frames = dict()
        pymilvus.connections.connect("default", host=config['master_node_ip'], port="19530")
        if remove and pymilvus.utility.has_collection(collection_name):
            LOGGER.info("Remove collection and indexing")
            pymilvus.utility.drop_collection(self.collection_name)
            time.sleep(2)

        if pymilvus.utility.has_collection(collection_name):
            LOGGER.info(f"Load Milvus collection {self.collection_name}")
            self.collection: pymilvus.Collection = pymilvus.Collection(self.collection_name)
        else:
            LOGGER.info(f"Create Milvus collection {self.collection_name}, dim={ndim}")
            self.__create_collection__(ndim)
        
        LOGGER.info(f"Load collection {self.collection} <=> {self.collection_name}")
        self.collection.load()  # load collection into memory
        LOGGER.info(f"Load Milvus Collection {self.collection.name}, nentities={self.collection.num_entities}")

        # Load exists data into related frames
        num_entities = self.collection.num_entities  # query(expr="pk >= 0", output_fields = ["frame_id", "pk"])
        # for e in entities:
        #     self.related_frames[e['pk']] = e['frame_id']
        LOGGER.info(f"Entities loaded : {num_entities}")
        # # IP or L2
        # If you use IP to calculate embeddings similarities, you must normalize your embeddings.
        # After normalization, the inner product equals cosine similarity.
        self.search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    @staticmethod
    def ping(server):
        try:
            # Attempt to establish a connection to the Milvus server
            pymilvus.connections.connect("default", host=server, port="19530")
            collections = pymilvus.utility.list_collections()
            LOGGER.info(f"List collection : {collections}")
            if isinstance(collections, list):
                # close connection
                LOGGER.info("Ping successfully, close Milvus connection")
                pymilvus.connections.disconnect("default")
                return True
            else:
                return False
        except Exception as e:
            LOGGER.exception(f"Milvus server is not available. Error: {str(e)}")
            return False

    @staticmethod
    def disconnect():
        pymilvus.connections.disconnect("default")

    def search(self, now, vecs):
        ts = int((now - datetime.timedelta(seconds=TIME_FILTER_SECOND)).timestamp() * 1000)
        expr = f"frame_id < {ts}"
        LOGGER.debug(f"====== expr = {expr}")
        rs = dict()
        search_result = self.collection.search(
            data=vecs,
            anns_field="embeddings", param=self.search_params,
            limit=4, expr=expr
        )
        for hits in search_result:
            for hit in hits:
                related_frame = get_frame_from_face_id(hit.id)
                if related_frame == 0:
                    continue

                # matched
                if hit.distance < self.threshold and related_frame not in rs:
                    rs[related_frame] = {
                        "frame": related_frame,
                        "score": 1 - hit.distance
                    }

                LOGGER.debug(f"d = {1 - hit.distance} ; t = {1 - self.threshold} ;")
                # LOGGER.debug(f"d = {1 - hit.distance} ; t = {1 - self.threshold} ; {hit.entity}")
        return rs

    def insert(self, frame_id, vecs):
        # vecs: list of vectors
        if vecs.shape[1] != self.ndim:
            LOGGER.warning(f"Invalid dimension: {self.ndim} <> {vecs.shape[1]}")
            return False

        n_vecs = vecs.shape[0]
        ids = list()
        frame_ids = list()
        for i in range(n_vecs):
            iid = int("2" + (frame_id + i + random.randint(0, 10000)).__str__())  # generate ids
            set_frame_and_face_id(iid, frame_id)
            # self.related_frames[iid] = frame_id
            frame_ids.append(frame_id)
            ids.append(iid)

        self.collection.insert([ids, frame_ids, vecs])
        LOGGER.debug(f"Insert id {ids}, frame {frame_id}")
        return True
    
    def __create_collection__(self, dim):
        fields = [
            pymilvus.FieldSchema(name="pk", dtype=pymilvus.DataType.INT64, is_primary=True, auto_id=False),
            pymilvus.FieldSchema(name="frame_id", dtype=pymilvus.DataType.INT64),  # frame id
            pymilvus.FieldSchema(name="embeddings", dtype=pymilvus.DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = pymilvus.CollectionSchema(fields, f"This is collection name : {self.collection_name}")
        self.collection = pymilvus.Collection(self.collection_name, schema)
        self.collection.create_index(
            field_name="embeddings",
            index_params={
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 64}
            }
        )

        return self


if __name__ == '__main__':
    pass
