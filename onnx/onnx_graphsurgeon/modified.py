from onnx import shape_inference
import onnx_graphsurgeon as gs
import numpy as np
import onnx

graph = gs.import_onnx(onnx.load("model.onnx"))

# graph.nodes is node list
# 应该也是排过序的
# graph.tensors() 是一个方法，返回的是字典
print(graph.nodes)

# 1. Remove the `b` input of the add node
# filt node list by node's op
first_add = [node for node in graph.nodes if node.op == "Add"][0]
print(first_add)
#  (Add)
#         Inputs: [
#                 Variable (mul_out): (shape=None, dtype=None)
#                 Constant (b): (shape=[1, 3, 224, 224], dtype=<class 'numpy.float32'>)
#         ]
#         Outputs: [
#                 Variable (add_out): (shape=None, dtype=None)
#         ]

# filt inputs list by input's name
first_add.inputs = [inp for inp in first_add.inputs if inp.name != "b"]
# inputs and outputs are both list type
print(first_add)
# (Add)
#         Inputs: [
#                 Variable (mul_out): (shape=None, dtype=None)
#         ]
#         Outputs: [
#                 Variable (add_out): (shape=None, dtype=None)
#         ]

# 2. Change the Add to a LeakyRelu
# 只改变 node op，不改变其他，比如输入与输出
first_add.op = "LeakyRelu"
first_add.attrs["alpha"] = 0.02

# 3. Add an identity after the add node
# 这个 tensor 变量作为中间层的输出，没有指定 shape
identity_out = gs.Variable("identity_out", dtype=np.float32)
# 这里直接将first_add节点的输出作为了当下节点的输入
identity = gs.Node(op="Identity", 
                   inputs=first_add.outputs,
                   outputs=[identity_out])

# append new node into the list
# 为 graph 添加新的节点，就是在 graph.nodes 这个列表中 append 新的 node 对象
graph.nodes.append(identity)

# 4. Modify the graph output to be the identity output
graph.outputs = [identity_out]

# 5. Remove unused nodes/tensors, and topologically sort the graph
# ONNX requires nodes to be topologically sorted to be considered valid.
# Therefore, you should only need to sort the graph when you have added new nodes out-of-order.
# In this case, the identity node is already in the correct spot (it is the last node,
# and was appended to the end of the list), but to be on the safer side, we can sort anyway.
# 拓扑排序, 调用 graph.toposort() 保证其有序性
# when toposort(), graph will use the given inputs and outputs infos.
graph.cleanup().toposort()

onnx.save(gs.export_onnx(graph), "modified.onnx")
print('done!')