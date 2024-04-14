from onnx import shape_inference
import onnx_graphsurgeon as gs
import numpy as np
import onnx
import netron
# Register functions to make graph generation easier

# self.layer() 返回 outputs，是一个列表，这个例子中因为知道模型结构所以直接返回了列表的第一个元素，也就是 output tensor
@gs.Graph.register()
def min(self, *args):
    return self.layer(op="Min", inputs=args, outputs=["min_out"])[0]


@gs.Graph.register()
def max(self, *args):
    return self.layer(op="Max", inputs=args, outputs=["max_out"])[0]


@gs.Graph.register()
def identity(self, inp):
    return self.layer(op="Identity", inputs=[inp], outputs=["identity_out"])[0]

##########################################################################################################
# Generate the graph
graph = gs.Graph()

graph.inputs = [gs.Variable("input", shape=(4, 4), dtype=np.float32)]

# Clip values to [0, 6]
MIN_VAL = np.array(0, np.float32)
MAX_VAL = np.array(6, np.float32)

# Add identity nodes to make the graph structure a bit more interesting
inp = graph.identity(graph.inputs[0])
max_out = graph.max(graph.min(inp, MAX_VAL), MIN_VAL)
graph.outputs = [graph.identity(max_out), ]

# Graph outputs must include dtype information
# identity_out has been created by graph.identity(), and because it's an output tensor, dtype must be set
# shape is not necessarily set
graph.outputs[0].to_variable(dtype=np.float32, shape=(4, 4))

onnx.save(gs.export_onnx(graph), "replacing_model.onnx")
print('done!')

@gs.Graph.register()
def replace_with_clip(self, inputs, outputs):
    # replace_with_clip() 会先将子图分离出来，
    # 方法就是清空所有输入 tensor 的输出节点, 清空所有输出 tensor 的输入节点
    
    # Disconnect output nodes of all input tensors
    # inp 是 tensor, inp.outputs 是消费这个 tensor 的所有节点
    # 这个例子比较特殊, 因为每个 tensor 的后继节点只有一个, 如果出现一个 tensor 后继节点有多个的情况, 就不能这样直接清空了
    for inp in inputs:
        inp.outputs.clear()

    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()

    # Insert the new node.
    return self.layer(op="Clip", inputs=inputs, outputs=outputs)

##########################################################################################################
# Now we'll do the actual replacement
graph = gs.import_onnx(onnx.load("replacing_model.onnx"))

tmap = graph.tensors()
print(tmap["identity_out_0"])
print(tmap["identity_out_0"].inputs)
print(tmap["identity_out_0"].outputs)

# 按照 Subgraph Replacement Basics 的步骤，替换子图要先找到子图的输入输出，将子图从模型中分离出来
# 可以通过 graph.tensors() 方法，也可以直接 Netron 查看
# You can figure out the input and output tensors using Netron. In our case:
# Inputs: [inp, MIN_VAL, MAX_VAL]
# Outputs: [max_out]
inputs = [tmap["identity_out_0"], 
          tmap["onnx_graphsurgeon_constant_5"],
          tmap["onnx_graphsurgeon_constant_2"]]
outputs = [tmap["max_out_6"]]

# print(graph)

# replace_with_clip() 会先将子图分离出来，
# 方法就是清空所有输入 tensor 的输出节点, 清空所有输出 tensor 的输入节点
graph.replace_with_clip(inputs, outputs)

# print(graph)

# Remove the now-dangling subgraph.
graph.cleanup().toposort()

# That's it!
onnx.save(gs.export_onnx(graph), "replaced_model.onnx")
netron.start("replaced_model.onnx")
print('done!')