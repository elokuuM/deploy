#ifndef _CUSTOM_H
#define _CUSTOM_H

#include <string>
#include <vector>
#include <iostream>

#include "NvInfer.h"
#include "common.h"

#include <NvInferRuntimeCommon.h>

using namespace nvinfer1;

class CustomLayerPlugin : public IPluginV2IOExt
{
public:
    CustomLayerPlugin(PluginFieldCollection const &field_collection) noexcept;

    CustomLayerPlugin(const CustomLayerPlugin &plugin);

    CustomLayerPlugin(void const *data,
                      size_t length);

    CustomLayerPlugin() = delete;

    ~CustomLayerPlugin() override;

    // Get the number of outputs from the layer.
    int32_t getNbOutputs() const noexcept override;

    // Get the dimension of an output tensor. return the dims of tensor at the requested index
    Dims getOutputDimensions(int32_t index,
                             Dims const *inputs,
                             int32_t num_inputs) noexcept override;

    // Initialize the layer for execution. This is called when the engine is created.
    // initialize() 被调用之前，已经进行了配置(configurePlugin)， engine 也已经被创建了
    // initialize 一般被用来初始化数据， 为 inference 进行准备
    int32_t initialize() noexcept override;

    // Release resources acquired during plugin layer initialization.
    // initialize 申请的资源都要在这里释放
    void terminate() noexcept override;

    // Get the workspace size required by the layer.
    // builder 会调用 getWorkspaceSize(), 并分配内存;
    // 在 engine 创建 context 时应该也会调用 (during engine startup), createExecutionContext()
    // engine 创建 context 还有另外一个不分配内存的方法， createExecutionContextWithoutDeviceMemory, 应该就不会调用 getWorkspaceSize
    size_t getWorkspaceSize(int32_t max_batch_size) const noexcept override;

    // Execute the layer
    int32_t enqueue(int32_t batch_size,
                    void const *const *inputs,
                    void *const *outputs,
                    void *workspace,
                    cudaStream_t stream) noexcept override;

    // Return the size of the serialization buffer
    size_t getSerializationSize() const noexcept override;

    // Serialize the layer.
    // serialize 需要做的是， 把 custom layer 用到的参数保存下来， 在反序列化时可以直接用这些参数初始化一个 IPluginV2 对象
    // buffer 指向的是已经分配好的内存，大小等于 getSerializationSize()
    void serialize(void *buffer) const noexcept override;

    // Configuration of plugin called by TensorRT builder
    //
    // configurePlugin() 里面配置一些插件的 private 参数，比如，层的输入输出大小。
    // 需要的参数都通过参数列表传进来了，有些参数写成常数也行，因为可能不经常变；
    // 序列化和反序列化可能要把参数保存下来
    void configurePlugin(PluginTensorDesc const *in,
                         int32_t num_inputs,
                         PluginTensorDesc const *out,
                         int32_t num_outputs) noexcept override;

    bool supportsFormatCombination(int32_t pos,
                                   PluginTensorDesc const *in_out,
                                   int32_t num_inputs,
                                   int32_t num_outputs) const noexcept override;

    // 在 IPluginV2Ext 声明
    // Return the DataType of the plugin output at the requested index.
    DataType getOutputDataType(int32_t index,
                               DataType const *input_types,
                               int32_t num_inputs) const noexcept override;

    // Return the plugin type.
    // Should match the plugin name returned by the corresponding plugin creator.
    AsciiChar const *getPluginType() const noexcept override;

    // Return the plugin version.
    // Should match the plugin version returned by the corresponding plugin creator.
    AsciiChar const *getPluginVersion() const noexcept override;

    // Destroy the plugin object.
    // This will be called when the network, builder or engine is destroyed.
    void destroy() noexcept override;

    // Clone the plugin object.
    // This copies over internal plugin parameters and returns a new plugin object with these parameters.
    IPluginV2Ext *clone() const noexcept override;

    // Set the namespace that this plugin object belongs to.
    // Ideally, all plugin objects from the same plugin library should have the same namespace.
    void setPluginNamespace(AsciiChar const *plugin_namespace) noexcept override;

    // Return the namespace of the plugin object.
    AsciiChar const *getPluginNamespace() const noexcept override;

    // Return true if output tensor is broadcast across a batch.
    bool isOutputBroadcastAcrossBatch(int32_t output_idx,
                                      bool const *is_broadcasted_inputs,
                                      int32_t num_inputs) const noexcept override;

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool canBroadcastInputAcrossBatch(int32_t input_idx) const noexcept override;

private:
    // 这是自己加进来的属性
    // Plugin namespace
    // 相当于 std::string plugin_namespace = "";
    // {} 用来初始化变量
    // 可以直接在类声明中直接定义初始化类成员属性
    std::string plugin_namespace{};
    std::string plugin_version = "1";
    float alpha;
    float *d_alpha;
};

class CustomLayerPluginCreator : public IPluginCreator
{
public:
    // 只在程序一开始加载 plugin lib 时， 加载一次
    // CustomLayerPluginCreator() 中要定义好所有的 PluginFieldCollection 的 name 与 type 字段
    // PluginFieldCollection data字段 可以为 nullptr
    // 如果定义的 name/type 与模型中的不相符是无法正确读取模型属性的,
    // 这里像是作一个声明，告诉 parser， 这个插件我需要这些 field， 你读取模型时帮我留意一下
    // parser 会将模型中的这些个字段保存到 PluginFieldCollection 中, 后面 createPlugin() 中就可以使用了
    CustomLayerPluginCreator() noexcept;
    // 只在程序结束时，析构一次
    ~CustomLayerPluginCreator() override;

    /**
     * 注册表的 key 由 plugin name， plugin version and plugin namespace 组成的
     * 所以在程序一开始加载 plugin lib 时， getPluginName， getPluginVersion， getPluginNamespace 会被调用若干次
     * plugin name 对应模型网络层中的 op
     * plugin version 对应模型网络层中的 plugin_version
     * plugin namespace 对应模型网络层中的 plugin_namespace，
    */

    // Return the plugin name.
    AsciiChar const *getPluginName() const noexcept override;

    // Return the plugin version.
    // For all internal TensorRT plugins, this defaults to "1"
    // 
    // getPluginVersion() 返回的版本必须与模型节点的版本一致，否则会报错没有找到插件
    // 
    // version 与 namespace 都是在模型节点的 attrs 中设置的， key 分别叫 plugin_version 和 plugin_namespace， 把 value 设成字符串就可以了
    //
    // 长度 <= 1024， 包括\0
    AsciiChar const *getPluginVersion() const noexcept override;

    // Return a list of fields that needs to be passed to createPlugin.
    // 返回的 field 是在 CustomLayerPluginCreator() 时就定义好的 field， 而不是模型文件中的
    // 注意，在这个函数中，每个字段的 data = nullptr
    PluginFieldCollection const *getFieldNames() noexcept override;

    // Return a plugin object.
    // Return nullptr in case of error.
    //
    // 使用网络模型中的字段内容，创建 IPluginV2 对象
    // 与 getFieldNames() 相比， PluginFieldCollection 中不仅有 name, type, 还有从模型中读取出来的 data
    // 从流程上看，创建 plugin layer 时， 会先调用 getFieldNames， 这样 parser 就知道从模型的layer中读取哪些参数，
    // 将参数从模型中读出到 data 字段中， 然后调用 createPlugin， 这样在 createPlugin 中就可以使用 data 字段了
    // 如果 plugin layer 中没有指定的name参数， 就会报错 std::out_of_range 与 Attribute not found: xxxxx
    IPluginV2 *createPlugin(AsciiChar const *name,
                            PluginFieldCollection const *field_collection) noexcept override;

    // Called during deserialization of plugin layer.
    // Return a plugin object.
    IPluginV2 *deserializePlugin(AsciiChar const *name,
                                 void const *data,
                                 size_t length) noexcept override;

    // Set the namespace of the plugin creator based on the plugin library it belongs to.
    // This can be set while registering the plugin creator.
    void setPluginNamespace(AsciiChar const *plugin_namespace) noexcept override;

    // Return the namespace of the plugin creator object
    AsciiChar const *getPluginNamespace() const noexcept override;

private:
    // 这是自己加进来的属性
    // Plugin namespace
    std::string plugin_namespace{};

    std::string plugin_version = "1";

    // 这是自己加进来的属性
    // 类静态成员属性必须在类外定义初始化，而且只能初始化一次
    static PluginFieldCollection field_names;

    // 这是自己加进来的属性
    // Plugin field attributes
    static std::vector<PluginField> plugin_attrs;
};

#endif