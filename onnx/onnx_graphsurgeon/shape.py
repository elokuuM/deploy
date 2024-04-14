from onnx import shape_inference
import onnx_graphsurgeon as gs
import numpy as np
import onnx
import netron
# Register operators we'll need.
# NOTE: Since all the ops used here only have a single output, we return the
# first output directly instead of returning the list of outputs.

@gs.Graph.register()
def shape(self, a):
    return self.layer(op="Shape", inputs=[a], outputs=["shape_out_gs"])[0]


@gs.Graph.register()
def reduce_prod(self, a, axes, keepdims=True):
    return self.layer(op="ReduceProd", inputs=[a], attrs={"axes": axes, "keepdims": int(keepdims)}, outputs=["reduce_prod_out_gs"])[0]


@gs.Graph.register()
def reshape(self, data, shape):
    return self.layer(op="Reshape", inputs=[data, shape], outputs=["reshape_out_gs"])[0]


@gs.Graph.register()
def gather(self, data, indices):
    return self.layer(op="Gather", inputs=[data, indices], outputs=["gather_out_gs"])[0]


@gs.Graph.register()
def concat(self, inputs, axis=0):
    return self.layer(op="Concat", inputs=inputs, attrs={"axis": axis}, outputs=["concat_out_gs"])[0]

##########################################################################################################
# Create the graph
graph = gs.Graph()

# First we set up the inputs, using gs.Tensor.DYNAMIC to specify dynamic dimensions.
graph.inputs = [gs.Variable(name="data", 
                            dtype=np.float32, 
                            shape=(gs.Tensor.DYNAMIC, 3, gs.Tensor.DYNAMIC, 5))]

input_shape = graph.shape(graph.inputs[0])

# Part 1 - Flattening the input by computing its volume and reshaping.
# 1. Flattening an input tensor that includes dynamic dimensions. The layers involved are:
# - Shape: to get the input shape.
# - ReduceProd: to compute the volume of the input shape.
# - Reshape: to change the shape of the input to its volume.
volume = graph.reduce_prod(input_shape, axes=[0])
# 一个分支经过 shape->reduce_prod 计算出了 volume; 另一个分支直接传给 reshape
flattened = graph.reshape(graph.inputs[0], volume)

# Part 2 - Collapsing some, but not all, dimensions. In this case, we will flatten the last 2 dimensions.
# To do so, we'll gather the last 2 dimensions, compute their volume with reduce_prod, and concatenate the
# result with the first 2 dimensions.
# NOTE: The code here is *not* specific to images, but we use NCHW notation to make it more readable.
# 2. Collapsing some, but not all, dimensions of an input tensor that includes dynamic dimensions. The layers involved are:
# - Shape: to get the input shape.
# - Gather: to get the first 2 dimensions of the input shape.
# - Gather: to get the last 2 dimensions of the input shape.
# - ReduceProd: to compute the volume of the last 2 dimensions.
# - Concat: to combine the first dimension of the original shape with the volume of the other dimensions.
# - Reshape: to reshape the input based on the computed shape.
NC = graph.gather(input_shape, indices=[0, 1])
HW = graph.gather(input_shape, indices=[2, 3])
new_shape = graph.concat([NC, graph.reduce_prod(HW, axes=[0])])
# Part 2 与 Part 1 的区别就在于计算 reshape 之后的形状的过程不同, 
# Part 1 是算了一下 volume, Part 2 是 flatten 了最后两个维度
partially_flattened = graph.reshape(graph.inputs[0], new_shape)

# Finally, set up the outputs and export.
flattened.name = "flattened"  # Rename output tensor to make it easy to find.
# NOTE: We must include dtype information for graph outputs
flattened.dtype = np.float32
partially_flattened.name = "partially_flattened"
partially_flattened.dtype = np.float32

graph.outputs = [flattened, partially_flattened]
model = gs.export_onnx(graph)
model = shape_inference.infer_shapes(model)
onnx.save(model, "shape_op.onnx")
netron.start("shape_op.onnx")
print('done!')