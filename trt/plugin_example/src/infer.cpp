#include <common.h>
#include <logger.h>

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include "NvOnnxParser.h"

#include <custom.h>
#include <builder.hpp>
using namespace std;
using namespace sample;

REGISTER_TENSORRT_PLUGIN(CustomLayerPluginCreator);

/**
 * 因为自定义层会用到 cuda， 所以编译时需要用 nvcc
 * cmake 中不仅需要找 tensorrt package， 还需要找 cuda package
 * cuda_add_executable(custom_layers_infer custom_layers_infer.cpp custom_add.cu ${UTILS})
*/

int main(int argc, char **argv)
{
    builder("model.onnx");
    printf("running infer demo... \n");

    if (argc < 2)
    {
        cout << "please specific plan file path" << endl;
        exit(1);
    }

    char *plan_file_path = argv[1];

    printf("plan file path: [%s] \n", plan_file_path);

    int severity_level = static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE);

    Logger gLogger;
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine *engine = nullptr;

    char *data;
    size_t size;
    // size = read_bin_file(plan_file_path, &data);
    std::ifstream file(plan_file_path, std::ios::binary); 
    if (file.good()) { 
            file.seekg(0, file.end); 
            size = file.tellg(); 
            file.seekg(0, file.beg); 
            data = new char[size]; 
            assert(data); 
            file.read(data, size); 
            file.close(); 
    }

    engine = runtime->deserializeCudaEngine(data, size, nullptr);

    printf("release runtime \n");
    runtime->destroy();

    free(data);

    nvinfer1::IExecutionContext *context = engine->createExecutionContext();

    // PRINT_CONTEXT(context);

    // PRINT_ENGINE(engine);

    void *d_buffers[64];

    for (int i = 0; i < context->getEngine().getNbBindings(); i++)
    {
        nvinfer1::Dims dims = context->getBindingDimensions(i);
        printf("dims = {%d, %d, %d, %d} \n", dims.d[0], dims.d[1], dims.d[2], dims.d[3]);
        int volume = dims.d[0] * dims.d[1] * dims.d[2] * dims.d[3] * sizeof(float);

        cudaMalloc((void **)&d_buffers[i], volume);

        cudaMemset(d_buffers[i], 0, volume);
    }
    printf("malloc %d buffers\n", context->getEngine().getNbBindings());

    context->enqueueV2(d_buffers, 0, nullptr);

    for (int i = 0; i < context->getEngine().getNbBindings(); i++)
    {
        if (context->getEngine().bindingIsInput(i))
        {
            continue;
        }
        nvinfer1::Dims dims = context->getBindingDimensions(i);
        printf("output dims = {%d, %d, %d, %d} \n", dims.d[0], dims.d[1], dims.d[2], dims.d[3]);
        int volume = dims.d[0] * dims.d[1] * dims.d[2] * dims.d[3] * sizeof(float);

        float *h_output = (float *)malloc(volume);
        cudaMemcpy(h_output, d_buffers[i], volume, cudaMemcpyDeviceToHost);

        printf("buffers[%d]: \n", i);
        for (int _idx = 0; _idx < volume / sizeof(float) && _idx < 24; ++_idx)
        {
            printf("%f \n", h_output[_idx]);
        }

        free(h_output);
    }

    for (int i = 0; i < context->getEngine().getNbBindings(); i++)
    {
        cudaFree(d_buffers[i]);
    }
    printf("release %d buffers\n", context->getEngine().getNbBindings());

    printf("release context \n");
    context->destroy();
    printf("release engine \n");
    engine->destroy();

    return 0;
}