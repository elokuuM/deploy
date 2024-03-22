import onnx, onnxsim, onnxruntime

model_onnx = onnx.load('./working/model.onnx')
onnx.checker.check_model(model_onnx)

# for node in model_onnx.graph.node:
#     if 'Resize' == node.op_type:
#         # while '/decoder/Constant_4_output_0' in node.input:
#         #     node.input.remove('/decoder/Constant_4_output_0')
#         print(node.input)       
# for input in model_onnx.graph.input:
#     print(input.name)

graph = model_onnx.graph
# 获得onnx节点
node = graph.node
resize_node_idx = []
resize_node_output = []
for i in range(len(node)):
    if node[i].op_type =='Resize':
        resize_node_idx.append(i)
        resize_node_output.append(node[i].output[0])
print(resize_node_idx)
print(resize_node_output)

for i in range(len(resize_node_idx)):
    new_node = onnx.helper.make_node(
        "Resize",
        inputs=[],
        outputs=resize_node_output[i],
        scales=[1, 1, 2, 2], 
        coordinate_transformation_mode_s="pytorch_half_pixel", 
        cubic_coeff_a_f=-0.75, 
        mode_s='cubic', 
        nearest_mode_s="floor")
    node.remove(node[resize_node_idx[i]])  
    node.insert(resize_node_idx[i], new_node) 
    
    

onnx.checker.check_model(model_onnx) 
onnx.save(model_onnx, './working/model-cresize.onnx') 