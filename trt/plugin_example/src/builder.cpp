#include <builder.hpp>

REGISTER_TENSORRT_PLUGIN(CustomLayerPluginCreator);

using namespace std;
using namespace sample;
int builder(char* onnx_file_path)
{
    printf("running build demo... \n");

    char plan_file_path[256];
    strncpy(plan_file_path, onnx_file_path, strlen(onnx_file_path) - 5);
    plan_file_path[strlen(onnx_file_path) - 5] = '\0';
    strcat(plan_file_path, ".trt");

    printf("onnx file path: [%s] \n", onnx_file_path);
    printf("plan file path: [%s] \n", plan_file_path);

    int severity_level = static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE);

    /* (1) */
    printf("/*(1)*/ \n");
    Logger gLogger;
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger);

    /* (2) */
    printf("/*(2)*/ \n");
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();

    /* (3) */
    printf("/*(3)*/ \n");
    const uint32_t explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // nvinfer1::INetworkDefinition 还提供直接构建网络的方法，以及对网络结构的查询功能，这里不写了，太多了
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);

    /* (4) */
    printf("/*(4)*/ \n");
    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, gLogger);

    /* (5) */
    printf("/*(5)*/ \n");
    parser->parseFromFile(onnx_file_path, severity_level);

    config->setMaxWorkspaceSize(1_GiB);

    // PRINT_CONFIG(config);

    const int batch_size = 1;
    // std::cout << "set batch size as " << batch_size << std::endl;
    builder->setMaxBatchSize(batch_size);

    // PRINT_CONFIG(config);

    // PRINT_BUILDER(builder);

    nvinfer1::ICudaEngine *engine = nullptr;

    /* (6) */
    printf("/*(6)*/ \n");
    engine = builder->buildEngineWithConfig(*network, *config);

    std::cout << "num of inputs: " << network->getNbInputs() << std::endl;
    std::cout << "num of outputs: " << network->getNbOutputs() << std::endl;

    nvinfer1::IHostMemory *serialized_engine = engine->serialize();

    printf("host memory data type: (%d) \n", serialized_engine->type());
    printf("plan file size %.2f MB \n", serialized_engine->size() / 1000.0 / 1000.0);

    // write_bin_file(plan_file_path,
    //                (char *)serialized_engine->data(),
    //                serialized_engine->size());
    std::ofstream p(plan_file_path, std::ios::binary); 
    if (!p) { 
            std::cerr << "could not open output file to save model" << std::endl; 
            return -1; 
    } 
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size()); 
    
    printf("release engine \n");
    engine->destroy();
    printf("release serialized_engine \n");
    serialized_engine->destroy();
    printf("release parser \n");
    parser->destroy();
    printf("release config \n");
    config->destroy();
    printf("release network \n");
    network->destroy();
    printf("release builder \n");
    builder->destroy();

    return 0;
}