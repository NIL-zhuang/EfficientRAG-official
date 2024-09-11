class BaseIndex:
    def __init__(self):
        pass

    def search(self, query, top_k):
        raise NotImplementedError

    def serialize(self, dir_path):
        raise NotImplementedError

    def deserialize(self, dir_path):
        raise NotImplementedError

    def exist_index(self, dir_path):
        raise NotImplementedError
