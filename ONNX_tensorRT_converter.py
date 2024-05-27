'''
input_shape = (1, 3, 544, 1024)
output_shape = (1,8, 544, 1024)
'''

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

with open("model.onnx", "rb") as model:
    parser.parse(model.read())

config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB

engine = builder.build_engine(network, config)
with open("FCN_.trt", "wb") as f:
    f.write(engine.serialize())
