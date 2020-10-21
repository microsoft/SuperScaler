import json


class AttrEnconding(json.JSONEncoder):
    def default(self, o):
        if o.__class__.__name__ == "TensorProto":
            return "Unsupport type: TensorProto"
