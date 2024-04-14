from onnx import shape_inference
import onnx_graphsurgeon as gs
import numpy as np
import onnx

# 模型裁剪的主要步骤，假如不改动 graph 结构：
# 1. 如果有需要的话，修改 tensor 参数；
# 2. 重新标记 graph 的输入与输出，这一步就相当与裁剪了

# Though omitted in this example, in some cases, it may be useful to embed
# shape information in the graph. We can use ONNX shape inference to do this:
#
# model = shape_inference.infer_shapes(onnx.load("model.onnx"))
#
# IMPORTANT: In some cases, ONNX shape inference may not correctly infer shapes,
# which will result in an invalid subgraph. To avoid this, you can instead modify
# the tensors to include the shape information yourself.

# 可以使用 shape_inference 推理出没有指定大小的 tensor 的 shape
# 但是有时 shape_inference 推理会算错 shape，这种情况下，开发者就只能自己把 shape 信息写到 tensor 中了
# 我的理解是即使 shape_inference 错了，也不影响模型运行

# 加载本地 onnx 文件模型
# 先将模型文件读进来，然后再加载
model = onnx.load("model.onnx")
# 加入 shape_inference 可以在打印 tensor 信息时看到中间层输出的 shape
model = shape_inference.infer_shapes(model)
# 使用 gs.import_onnx() 加载 onnx 模型
graph = gs.import_onnx(model)

# Since we already know the names of the tensors we're interested in, we can
# grab them directly from the tensor map.
#
# NOTE: If you do not know the tensor names you want, you can view the graph in
# Netron to determine them, or use ONNX GraphSurgeon in an interactive shell
# to print the graph.

# 获取 graph 中所有的 tensors
# graph.tensors() 返回模型中的 tensor as an ordered dict
# 我的理解是这些 tensors 的顺序是经过拓扑排序的
tensors = graph.tensors()
[print(k, v) for k, v in tensors.items()]  # as ordered dict
# x0 Variable (x0): (shape=[1, 3, 224, 224], dtype=float32)
# x1 Variable (x1): (shape=[1, 3, 224, 224], dtype=float32)
# a Constant (a): (shape=[1, 3, 224, 224], dtype=<class 'numpy.float32'>)
# mul_out Variable (mul_out): (shape=None, dtype=None)
# b Constant (b): (shape=[1, 3, 224, 224], dtype=<class 'numpy.float32'>)
# add_out Variable (add_out): (shape=None, dtype=None)
# Y Variable (Y): (shape=[1, 3, 224, 224], dtype=float32)

# If you want to embed shape information, but cannot use ONNX shape inference,
# you can manually modify the tensors at this point:
# 如果希望加入 shape 信息，可以通过相应 tensor 对象，直接手动修改
# 这里是通过字典取到 tensor，然后使用 to_variable() 修改 shape，然后用新的 inputs、outputs 覆盖旧的
# graph.inputs = [tensors["x1"].to_variable(dtype=np.float32, shape=(1, 3, 224, 224))]
# graph.outputs = [tensors["add_out"].to_variable(dtype=np.float32, shape=(1, 3, 224, 224))]
#
# 可以不指定输入输出的 shape 但是必须要指定 data type
# IMPORTANT: You must include type information for input and output tensors if it is not already
# present in the graph.
#
# NOTE: ONNX GraphSurgeon will also accept dynamic shapes - simply set the corresponding
# dimension(s) to `gs.Tensor.DYNAMIC`, e.g. `shape=(gs.Tensor.DYNAMIC, 3, 224, 224)`
# 使用 gs.Tensor.DYNAMIC 代替动态维度
# 通过直接覆盖的方法，修改 graph 的输入与输出
graph.inputs = [tensors["x1"].to_variable(dtype=np.float32)]
graph.outputs = [tensors["add_out"].to_variable(dtype=np.float32)]

# Notice that we do not need to manually modify the rest of the graph. ONNX GraphSurgeon will
# take care of removing any unnecessary nodes or tensors, so that we are left with only the subgraph.
graph.cleanup()
tensors = graph.tensors()
print('After clean:')
[print(k, v) for k, v in tensors.items()]

# 需要注意的是，to_variable 是以覆盖的方式修改 tensor 的属性，而不是在原来的基础山添加

onnx.save(gs.export_onnx(graph), "subgraph.onnx")
print('done!')