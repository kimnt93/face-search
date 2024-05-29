from lib import config
import pymilvus


collection_name = config['face_model']


pymilvus.connections.connect("default", host=config['master_node_ip'], port="19530")

try:
    # Check if the collection exists before attempting to drop it
    if pymilvus.utility.has_collection(collection_name):
        # Drop the collection
        pymilvus.utility.drop_collection(collection_name)
        print(f"Xóa dữ liệu thành công")
    else:
        print(f"Dữ liệu đã được xóa")
except Exception as e:
    # Handle any errors that may occur during the operation
    print(f"Có lỗi xảy ra")
finally:
    # Disconnect from Milvus server (optional but recommended)
    pymilvus.connections.disconnect("embeddings")
