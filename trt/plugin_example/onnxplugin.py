# from ensurepip import version
import onnx_graphsurgeon as gs
import numpy as np
import onnx
from functools import reduce

# output = (input + offset) * mul * alpha
# element-wise multiple

# offset 与 mul 都是常量
# 这里尝试两种方法将常量传给plugin
# 第一种是将 offset 常量当作节点的输入
# 第二种是将 mul 当作节点的属性，通过 attrs 传入
# 第二种方法可能不行, 高维数组不能作为 attrs 字典的 value，可能发生了浅拷贝问题的
#                   大于一维的 numpy 或者 列表 都不行
# 所以例子都采用节点输入的方式传入 offset 与 mul
custom_shape = (1, 2, 3, 4)
input = gs.Variable(name='input',
                    dtype=np.float32,
                    shape=custom_shape)

offset = gs.Constant(name='offset',
                     values=np.ones(shape=custom_shape, dtype=np.float32))

# mul = np.empty(shape=custom_shape,
#                dtype=np.float32)
# print(mul.shape)
# print(mul)

volume = reduce(lambda x, y: x*y, custom_shape)
# values = 0, 1, 4, 9... 平方数列
values = [i*i for i in range(volume)]
values = np.array(values, dtype=np.float32).reshape(custom_shape)

mul = gs.Constant(name='mul',
                  values=values)

output = gs.Variable(name='output',
                     dtype=np.float32,
                     shape=custom_shape)

'''
attrs = {key:value, ...}
key list can be checked in the netron or onnx op illustration
'''

attrs = {'alpha': 0.1, 'plugin_version': '1', 'plugin_namespace': ''}

node = gs.Node(name='custom_layer',
               op='CustomLayer',
               attrs=attrs,
               inputs=[input, offset, mul], outputs=[output])

graph = gs.Graph(nodes=[node], inputs=[input], outputs=[output])

onnx.save(gs.export_onnx(graph), 'model.onnx')
print('done!')