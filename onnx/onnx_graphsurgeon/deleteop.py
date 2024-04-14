from onnx import shape_inference
import onnx_graphsurgeon as gs
import numpy as np
import onnx

# Inputs
x = gs.Variable(name="x", dtype=np.float32, shape=(1, 3, 224, 224))

# Intermediate tensors
i0 = gs.Variable(name="i0")
i1 = gs.Variable(name="i1")

# Outputs
y = gs.Variable(name="y", dtype=np.float32)

nodes = [
    gs.Node(op="Identity", inputs=[x], outputs=[i0]),
    gs.Node(op="FakeNodeToRemove", inputs=[i0], outputs=[i1]), # op可以是不存在的操作
    gs.Node(op="Identity", inputs=[i1], outputs=[y]),
]

graph = gs.Graph(nodes=nodes, inputs=[x], outputs=[y])
onnx.save(gs.export_onnx(graph), "removing.onnx")
print('done!')

graph = gs.import_onnx(onnx.load("removing.onnx"))

fake_node = [node for node in graph.nodes if node.op == "FakeNodeToRemove"][0]
print(fake_node)
#  (FakeNodeToRemove)
#         Inputs: [
#                 Variable (i0): (shape=None, dtype=None)
#         ]
#         Outputs: [
#                 Variable (i1): (shape=None, dtype=None)
#         ]

# Get the input node of the fake node
# node.i() 返回前驱节点
# node.o() 返回后继节点
inp_node = fake_node.i()
# inp_node = fake_node.i(0)
# Node provides i(idx) and o(idx) functions that can optionally be provided an index (default is 0)
# These serve as convenience functions for the alternative, 
# which would be to fetch the input/output tensor first, then fetch the input/output node of the tensor.
# For example, node.i() is equivalent to node.inputs[0].inputs[0]
# node 输入或者输出是 tensor，tensor的输入是生产这个 tensor 的 node，输出是消费这个 tensor 的 node
# 所以 fake_node.i()，也就是 fake_node.inputs[0].inputs[0]，返回了上一个节点

# print(inp_node)
#  (Identity)
#         Inputs: [
#                 Variable (x): (shape=[1, 3, 224, 224], dtype=float32)
#         ]
#         Outputs: [
#                 Variable (i0): (shape=None, dtype=None)
#         ]

# Reconnect the input node to the output tensors of the fake node, 
# so that the first identity node in the example graph now skips over the fake node.
inp_node.outputs = fake_node.outputs
# 下面这样也行，这个更容易理解
# inp_node.outputs = fake_node.o().inputs

# 这时打印 graph，可以看到 FakeNodeToRemove 的输出与前驱节点的输出都是 i1
# 就相当于说后继节点的输入有两个来源，这当然不行
# 所以需要删除不再使用的 FakeNodeToRemove 的输出
print(graph)

# 需要在 graph.cleanup() 之前，清空 fake_node.outputs
fake_node.outputs.clear()

# Remove the fake node from the graph completely
graph.cleanup()

onnx.save(gs.export_onnx(graph), "removed.onnx")
print('done!')