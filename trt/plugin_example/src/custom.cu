#include "custom.h"

// 类静态属性， 在类外初始化
PluginFieldCollection CustomLayerPluginCreator::field_names{};
std::vector<PluginField> CustomLayerPluginCreator::plugin_attrs{};

// output = (input + offset) * mul * alpha
__global__ void custom_layer(float *input,
                             float *offset,
                             float *mul,
                             float *alpha,
                             float *output)
{
    const int idx = threadIdx.x;
    output[idx] = (input[idx] + offset[idx]) * mul[idx] * alpha[0];
}

CustomLayerPlugin::CustomLayerPlugin(PluginFieldCollection const &field_collection) noexcept : IPluginV2IOExt{}
{
    printf("CustomLayerPlugin::CustomLayerPlugin(FC) \n");

    for (int i = 0; i < field_collection.nbFields; ++i)
    {
        // printf("field_collection[%d] name(%s) \n", i, (char *)field_collection.fields[i].name);

        if (!strcmp("alpha", field_collection.fields[i].name))
        {
            this->alpha = *((float *)field_collection.fields[i].data);
            // printf("this->alpha: %f \n", this->alpha);
            cudaMalloc(&this->d_alpha, sizeof(float));
            cudaMemcpy(this->d_alpha, &this->alpha, sizeof(float), cudaMemcpyHostToDevice);
        }
    }
}
CustomLayerPlugin::CustomLayerPlugin(const CustomLayerPlugin &plugin)
{
    this->alpha = plugin.alpha;
    cudaMalloc(&this->d_alpha, sizeof(float));
    cudaMemcpy(this->d_alpha, &this->alpha, sizeof(float), cudaMemcpyHostToDevice);
}

CustomLayerPlugin::CustomLayerPlugin(void const *data,
                                     size_t length) : IPluginV2IOExt{}
{
    printf("CustomLayerPlugin::CustomLayerPlugin(data, len) \n");

    memcpy(&this->alpha, data, sizeof(this->alpha));
    cudaMalloc(&this->d_alpha, sizeof(float));
    cudaMemcpy(this->d_alpha, &this->alpha, sizeof(float), cudaMemcpyHostToDevice);
}

CustomLayerPlugin::~CustomLayerPlugin()
{
    printf("CustomLayerPlugin::~CustomLayerPlugin() \n");

    cudaFree(this->d_alpha);
}

int32_t CustomLayerPlugin::getNbOutputs() const noexcept
{
    printf("CustomLayerPlugin::getNbOutputs() \n");
    return 1;
}

Dims CustomLayerPlugin::getOutputDimensions(int32_t index,
                                            Dims const *inputs,
                                            int32_t num_inputs) noexcept
{
    // printf("CustomLayerPlugin::getOutputDimensions(): index[%d] \n", index);
    printf("CustomLayerPlugin::getOutputDimensions(): inputs[%d] = (%d, %d, %d) \n",
           index,
           inputs[index].d[0],
           inputs[index].d[1],
           inputs[index].d[2]);

    return inputs[index];
}

int32_t CustomLayerPlugin::initialize() noexcept
{
    printf("CustomLayerPlugin::initialize() \n");

    return 0;
}

void CustomLayerPlugin::terminate() noexcept
{
    printf("CustomLayerPlugin::terminate() \n");

    return;
}

size_t CustomLayerPlugin::getWorkspaceSize(int32_t max_batch_size) const noexcept
{
    printf("CustomLayerPlugin::getWorkspaceSize(): max_batch_size[%d] \n", max_batch_size);

    return 0;
}

int32_t CustomLayerPlugin::enqueue(int32_t batch_size,
                                   void const *const *inputs,
                                   void *const *outputs,
                                   void *workspace,
                                   cudaStream_t stream) noexcept
{
    printf("CustomLayerPlugin::enqueue() \n");
    custom_layer<<<1, 24>>>((float *)inputs[0],
                            (float *)inputs[1],
                            (float *)inputs[2],
                            this->d_alpha,
                            (float *)outputs[0]);

    return 0;
}

size_t CustomLayerPlugin::getSerializationSize() const noexcept
{
    printf("CustomLayerPlugin::getSerializationSize() \n");

    size_t volume = 0;

    volume += sizeof(this->alpha);

    return volume;
}

void CustomLayerPlugin::serialize(void *buffer) const noexcept
{
    printf("CustomLayerPlugin::serialize() \n");

    memcpy(buffer, &this->alpha, sizeof(this->alpha));

    return;
}

void CustomLayerPlugin::configurePlugin(PluginTensorDesc const *in,
                                        int32_t num_inputs,
                                        PluginTensorDesc const *out,
                                        int32_t num_outputs) noexcept
{
    printf("CustomLayerPlugin::configurePlugin() \n");

    return;
}

bool CustomLayerPlugin::supportsFormatCombination(int32_t pos,
                                                  PluginTensorDesc const *in_out,
                                                  int32_t num_inputs,
                                                  int32_t num_outputs) const noexcept
{
    printf("CustomLayerPlugin::supportsFormatCombination(): pos[%d] \n", pos);

    return in_out[pos].type == DataType::kFLOAT;
}

DataType CustomLayerPlugin::getOutputDataType(int32_t index,
                                              DataType const *input_types,
                                              int32_t num_inputs) const noexcept
{
    printf("CustomLayerPlugin::getOutputDataType() \n");

    return input_types[0];
}

AsciiChar const *CustomLayerPlugin::getPluginType() const noexcept
{
    printf("CustomLayerPlugin::getPluginType() \n");

    return "CustomLayer";
}

AsciiChar const *CustomLayerPlugin::getPluginVersion() const noexcept
{
    printf("CustomLayerPlugin::getPluginVersion() \n");

    return this->plugin_version.c_str();
}

void CustomLayerPlugin::destroy() noexcept
{
    printf("CustomLayerPlugin::destroy() \n");

    delete this;
}

IPluginV2Ext *CustomLayerPlugin::clone() const noexcept
{
    printf("CustomLayerPlugin::clone() \n");

    IPluginV2Ext *plugin{nullptr};
    try
    {
        // 这里调用的是默认的拷贝构造
        // 不会拷贝 device memory 的内容，需要重写拷贝构造函数
        plugin = new CustomLayerPlugin(*this);
    }
    catch (std::exception &e)
    {
        std::cerr << "[E] Exception caught while instantiating CustomLayerPlugin in clone: " << e.what() << "\n";
    }
    return plugin;
}

void CustomLayerPlugin::setPluginNamespace(AsciiChar const *plugin_namespace) noexcept
{
    printf("CustomLayerPlugin::setPluginNamespace(): plugin_namespace[%s] \n", plugin_namespace);

    this->plugin_namespace = plugin_namespace;
}

AsciiChar const *CustomLayerPlugin::getPluginNamespace() const noexcept
{
    printf("CustomLayerPlugin::getPluginNamespace(): plugin_namespace[%s] \n", this->plugin_namespace.data());

    return this->plugin_namespace.data();
}

bool CustomLayerPlugin::isOutputBroadcastAcrossBatch(int32_t output_idx,
                                                     bool const *is_broadcasted_inputs,
                                                     int32_t num_inputs) const noexcept
{
    printf("CustomLayerPlugin::isOutputBroadcastAcrossBatch() \n");

    return false;
}

bool CustomLayerPlugin::canBroadcastInputAcrossBatch(int32_t input_idx) const noexcept
{
    printf("CustomLayerPlugin::canBroadcastInputAcrossBatch() \n");

    return false;
}

/***********************/
/***********************/
/* Plugin 基类在这里结束 */
/* PluginCreator 基类在这里开始 */
/******************************/
/******************************/

CustomLayerPluginCreator::CustomLayerPluginCreator() noexcept
{
    printf("CustomLayerPluginCreator::CustomLayerPluginCreator() \n");

    // 在创建 PluginCreator 时，就设置好有哪些 field，然后读取模型时，就会将模型中与 field 相同名称的属性一个一个取出来，
    plugin_attrs.emplace_back(PluginField("alpha", nullptr, PluginFieldType::kFLOAT32, 1));

    field_names.nbFields = plugin_attrs.size();
    field_names.fields = plugin_attrs.data();

    printf("num of attrs: %d \n", plugin_attrs.size());

    for (int i = 0; i < field_names.nbFields; ++i)
    {
        printf("field_collection[%d] name(%s) \n", i, (char *)field_names.fields[i].name);
    }
}

CustomLayerPluginCreator::~CustomLayerPluginCreator()
{
    printf("CustomLayerPluginCreator::~CustomLayerPluginCreator() \n");
}

AsciiChar const *CustomLayerPluginCreator::getPluginName() const noexcept
{
    printf("CustomLayerPluginCreator::getPluginName() \n");

    return "CustomLayer";
}

AsciiChar const *CustomLayerPluginCreator::getPluginVersion() const noexcept
{
    printf("CustomLayerPluginCreator::getPluginVersion() \n");

    return this->plugin_version.c_str();
}

PluginFieldCollection const *CustomLayerPluginCreator::getFieldNames() noexcept
{
    printf("CustomLayerPluginCreator::getFieldNames() \n");

    return &field_names;
}

IPluginV2 *CustomLayerPluginCreator::createPlugin(AsciiChar const *name,
                                                  PluginFieldCollection const *field_collection) noexcept
{
    printf("CustomLayerPluginCreator::createPlugin(): name[%s] \n", name);

    IPluginV2 *plugin = nullptr;
    if (field_collection != nullptr)
    {
        try
        {
            plugin = new CustomLayerPlugin(*field_collection);
            field_names = *field_collection;
        }
        catch (std::exception &e)
        {
            std::cerr << "[E] Exception caught while instantiating CustomLayerPlugin in createPlugin: " << e.what() << "\n";
        }
    }
    return plugin;
}

IPluginV2 *CustomLayerPluginCreator::deserializePlugin(AsciiChar const *name,
                                                       void const *data,
                                                       size_t length) noexcept
{
    printf("CustomLayerPluginCreator::deserializePlugin(): name[%s] \n", name);

    IPluginV2 *plugin{nullptr};
    try
    {
        plugin = new CustomLayerPlugin(data, length);
    }
    catch (std::exception &e)
    {
        std::cerr << "[E] Exception caught while instantiating CustomLayerPlugin in deserializePlugin: " << e.what() << "\n";
    }
    return plugin;
}

void CustomLayerPluginCreator::setPluginNamespace(AsciiChar const *plugin_namespace) noexcept
{
    printf("CustomLayerPluginCreator::setPluginNamespace() \n");

    this->plugin_namespace = plugin_namespace;
}

AsciiChar const *CustomLayerPluginCreator::getPluginNamespace() const noexcept
{
    printf("CustomLayerPluginCreator::getPluginNamespace(): plugin_namespace[%s] \n", this->plugin_namespace.c_str());

    return this->plugin_namespace.c_str();
}