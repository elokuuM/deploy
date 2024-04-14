from onnx import shape_inference
import onnx_graphsurgeon as gs
import numpy as np
import onnx

# print("Graph.layer Help:\n{}".format(gs.Graph.layer.__doc__))

# For automatically propagating data types
def propagate_dtype(outputs, dtype):
    for output in outputs:
        output.dtype = dtype
    return outputs

# We can use `Graph.register()` to add a function to the Graph class. 
# Later, we can invoke the function directly on instances of the graph, e.g. graph.gemm(...)
@gs.Graph.register()
def add(self, a, b):
    # The Graph.layer function creates a node, adds inputs and outputs to it, and finally adds it to the graph.
    # It returns the output tensors of the node to make it easy to chain.
    # 下面在使用时 dense = graph.relu(*graph.add(*axt, B)) 可以用 * 拆开列表
    # The function will append an index to any strings provided for inputs/outputs prior to using them to construct tensors. 
    # This will ensure that multiple calls to the layer() function will generate distinct tensors. 
    # However, this does NOT guarantee that there will be no overlap with other tensors in the graph.
    # Graph.layer() 会在自动构造的 tensor 名称后面加入 index，尽可能每次都创建名称不同的 tensor，但还是不能百分之百的保证
    # Hence, you should choose the prefixes to minimize the possibility of collisions.
    # 所以开发者要给 tensor 起个响亮的名字，防止撞衫
    # outputs 用 string 代替，创建时会自动生成变量
    # 返回 outputs，变量可能是自动生成的，另一方面，这种设计方便使用链式搭建模型结构
    # 应该使用 Graph.Constant的地方使用了 numpy array，Graph.layer() 会自动创建常量
    # 这些自动生成 变量常量 的行为，也可以减少维护的代码量
    return propagate_dtype(self.layer(op="Add", inputs=[a, b], outputs=["add_out_gs"]), a.dtype or b.dtype)


@gs.Graph.register()
def mul(self, a, b):
    return propagate_dtype(self.layer(op="Mul", inputs=[a, b], outputs=["mul_out_gs"]), a.dtype or b.dtype)


@gs.Graph.register()
def gemm(self, a, b, trans_a=False, trans_b=False):
    attrs = {"transA": int(trans_a), "transB": int(trans_b)}
    return propagate_dtype(self.layer(op="Gemm", inputs=[a, b], outputs=["gemm_out_gs"], attrs=attrs), a.dtype or b.dtype)


# You can also specify a set of opsets when regsitering a function.
# By default, the function is registered for all opsets lower than Graph.DEFAULT_OPSET
# 默认情况下，注册的 layer 兼容所有比 Graph.DEFAULT_OPSET 版本低或者等于的 opset
print('Graph.DEFAULT_OPSET: ', gs.Graph.DEFAULT_OPSET)
@gs.Graph.register(opsets=[11])
def relu(self, a):
    return propagate_dtype(self.layer(op="Relu", inputs=[a], outputs=["act_out_gs"]), a.dtype)


# Note that the same function can be defined in different ways for different opsets.
# 可以指定操作的 opsets，可以指定多次不同版本的opset
# It will only be called if the Graph's opset matches one of the opsets for which the function is registered.
# Hence, for the opset 11 graph used in this example, the following function will never be used.
@gs.Graph.register(opsets=[1])
def relu(self, a):
    raise NotImplementedError("This function has not been implemented!")


##########################################################################################################
# The functions registered above greatly simplify the process of building the graph itself.
# 创建相应 opset 版本的 graph
graph = gs.Graph(opset=11)

# Generates a graph which computes:
# output = ReLU((A * X^T) + B) (.) C + D

# input X
X = gs.Variable(name="X", shape=(64, 64), dtype=np.float32)
graph.inputs = [X]

# axt = (A * X^T)
# Note that we can use NumPy arrays directly (e.g. Tensor A), instead of Constants. 
# These will automatically be converted to Constants.
# 可以直接使用 numpy array 替代 gs.Constant()
A = np.ones(shape=(64, 64), dtype=np.float32)
axt = graph.gemm(A, X, trans_b=True)

# dense = ReLU(axt + B)
B = np.ones((64, 64), dtype=np.float32) * 0.5
dense = graph.relu(*graph.add(*axt, B))
# dense = graph.relu(graph.add(axt[0], B)[0])

# output = dense (.) C + D
# If a Tensor instance is provided (e.g. Tensor C), it will not be modified at all.
# If you prefer to set the exact names of tensors in the graph, 
# you should construct tensors manually instead of passing strings or NumPy arrays.
# 这里展示了两种创建 Constant 的方法，一种是是调用 gs.Constant()，另一种是传入 np arrays，构造方法会自动转换成常量
# 传入 numpy arrays 时，onnx_graphsurgeon 会自动给它起个名字
# 除此之外，如果是 Variable，还可以传入 string，会被用作 tensor 名称的一部分
# Just like tensor C, its name is set manually, while tensor D's name is not.
C = gs.Constant(name="C", values=np.ones(shape=(64, 64), dtype=np.float32))
D = np.ones(shape=(64, 64), dtype=np.float32)
graph.outputs = graph.add(*graph.mul(*dense, C), D)

onnx.save(gs.export_onnx(graph), "layerapi_model.onnx")
print('done!')