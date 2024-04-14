#include <builder.hpp>

using namespace std;
using namespace sample;

int buildEngine(char* onnx_file_path) {
    cout<< "running builder..."<< endl;
    
    // char onnx_file_path[256];
    // strcpy(onnx_file_path, onnxfile);

    // onnxfile[strlen(onnxfile) - 5] = '\0';

    char ege_file_path[256];
    strncpy(ege_file_path, onnx_file_path, strlen(onnx_file_path) - 5);
    ege_file_path[strlen(onnx_file_path) - 5] = '\0';
    strcat(ege_file_path, ".trt");

    cout<< "onnx file path:"<<onnx_file_path<<endl;
    cout<< "ege file path:"<<ege_file_path<<endl;

    int severity_level = static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE);

    /* (1) */
    Logger gLogger; 

    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger);

    /* (2) */
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();

    /* (3) */
    const uint32_t explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);

    /* (4) */
    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, gLogger);

    /* (5) */
    parser->parseFromFile(onnx_file_path, severity_level);

    config->setMaxWorkspaceSize(1_GiB);

    // PRINT_CONFIG(config);


    const int batch_size = 1;
    // std::cout << "set batch size as " << batch_size << std::endl;
    builder->setMaxBatchSize(batch_size);

    /*int8 quan*/
    // Int8EntropyCalibratorV2 calibrator(cache_file_path);

    // config->setFlag(nvinfer1::BuilderFlag::kINT8);
    // config->setInt8Calibrator(&calibrator);

    // PRINT_CONFIG(config);
    // PRINT_BUILDER(builder);

    nvinfer1::ICudaEngine *engine = nullptr;

    /* (6) */
    engine = builder->buildEngineWithConfig(*network, *config);

    std::cout << "num of inputs: " << network->getNbInputs() << std::endl;
    std::cout << "num of outputs: " << network->getNbOutputs() << std::endl;

    nvinfer1::IHostMemory *serialized_engine = engine->serialize();

    printf("host memory data type: (%d) \n", serialized_engine->type());
    printf("ege file size %.2f MB \n", serialized_engine->size() / 1000.0 / 1000.0);

    // write_bin_file(ege_file_path,
    //                (char *)serialized_engine->data(),
    //                serialized_engine->size());

    std::ofstream p(ege_file_path, std::ios::binary); 
    if (!p) { 
            std::cerr << "could not open output file to save model" << std::endl; 
            return -1; 
    } 
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size()); 
    std::cout << "generating file done!" << std::endl;

    engine->destroy();
    serialized_engine->destroy();
    parser->destroy();
    config->destroy();
    network->destroy();
    builder->destroy();

    return 0;
}