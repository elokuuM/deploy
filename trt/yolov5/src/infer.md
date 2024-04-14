#include <infer.hpp>

using namespace std;
using namespace sample;

int inferEngine(char* ege_file_path) {
    cout<<"running infer..."<<endl;

    cout<<"plan file path:"<<ege_file_path<<endl;

    // std::vector<cv::String> file_names;
    // cv::glob(input_path, file_names);

    int severity_level = static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE);

    Logger gLogger;
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine *engine = nullptr;

    char *data;
    size_t size;
    // size = read_bin_file(ege_file_path, &data);

    std::ifstream file(ege_file_path, std::ios::binary); 
    if (file.good()) { 
            file.seekg(0, file.end); 
            size = file.tellg(); 
            file.seekg(0, file.beg); 
            data = new char[size]; 
            assert(data); 
            file.read(data, size); 
            file.close(); 
    }

    //通过反序列化构建engine
    engine = runtime->deserializeCudaEngine(data, size, nullptr);
    runtime->destroy();

    free(data);

    /**
     * 
     * key: 引擎包括了网络的定义与权重参数
     *      进行推理还需要用来保存中间激活值的空间，这个上下文空间就是context
     *      用engine初始化context, context持有engine的引用
     *      允许 n*context + 1*engine 这种模式
     * 
     * 以下四种执行模型推理方式，区别在于同步与异步，是否需要指定 batchsize
     * enqueue 系列还多了两个与异步相关的参数，一个是 cuda stream；另一个是判断 inputs 内存是否可以刷新
     * 
     * enqueue(): 异步，batchsize
     * enqueueV2(): 异步，按照最大值执行
     * execute(): 同步，batchsize
     * executeV2(): 同步，按照最大值执行
     * 
     * */
    nvinfer1::IExecutionContext *context = engine->createExecutionContext();

    // for (int i = 0; i < file_names.size(); i++){
    // cv::Mat img = cv::imread(file_names[0], cv::IMREAD_COLOR);
    // float* input = (float*)malloc(3*640*640*sizeof(float));
    // preprocess(img, input);
    // cout <<"size of input: "<< sizeof(input)<<endl;
    // float* output = (float*)malloc(25200*85*sizeof(float));

    void *d_buffers[2]; //d_buffers是一个数组，数组里存的是void类型的指针。

    for (int i = 0; i < context->getEngine().getNbBindings(); i++)
    {
        nvinfer1::Dims dims = context->getBindingDimensions(i);
        printf("dims = {%d, %d, %d, %d} \n", dims.d[0], dims.d[1], dims.d[2], dims.d[3]);
        int volume = dims.d[0] * dims.d[1] * dims.d[2] * (dims.d[3]?dims.d[3]:1) * sizeof(float);

        CHECK(cudaMalloc((void **)&d_buffers[i], volume));
        // if (context->getEngine().bindingIsInput(i))
        // {
        //     CHECK(cudaMemcpy(d_buffers[i], input, volume, cudaMemcpyHostToDevice));
        // }
        // else
            cudaMemset(d_buffers[i], 0, volume);
    }
    printf("malloc %d buffers\n", context->getEngine().getNbBindings());

    // context->enqueue(1, d_buffers, 0, nullptr);

    /**
     * 
     * 一个 engine， 可以有多个 context；每个 context 都与 engine 在相同的 GPU 上，共用一套权重
     * 多个流并发进行推理时，要给每个 stream 配一个 context， 否则会发生错误
     * 
     * 最后一个参数可以传入一个 CudaEvent，
     * 作用是 An optional event which will be signaled when the input buffers can be refilled with new data
     * 也就是说，可以根据判断这个参数是否执行完毕，来判断能不能刷新 input buffers
     * 
    */
    context->enqueueV2(d_buffers, 0, nullptr);

    /**
     * 
     * key: 结果保存在 d_buffers[num_inputs:]，
     *      从device拷贝host，打印结果
     * 
     * */
    for (int i = 0; i < context->getEngine().getNbBindings(); i++)
    {
        if (context->getEngine().bindingIsInput(i))
        {
            continue;
        }
        nvinfer1::Dims dims = context->getBindingDimensions(i);
        printf("output dims = {%d, %d, %d, %d} \n", dims.d[0], dims.d[1], dims.d[2], dims.d[3]);
        int volume = dims.d[0] * dims.d[1] * dims.d[2] * (dims.d[3]?dims.d[3]:1) * sizeof(float);

        float *output = (float *)malloc(volume);
        cudaMemcpy(output, d_buffers[i], volume, cudaMemcpyDeviceToHost);

        printf("buffers[%d]: \n", i);
        for (int _idx = 0; _idx < volume / sizeof(float) && _idx < 5; ++_idx)
        {
            printf("%f \n", output[_idx]);
        }
        free(output);
    }

    for (int i = 0; i < context->getEngine().getNbBindings(); i++)
    {
        cudaFree(d_buffers[i++]);
    }
    printf("release %d buffers\n", context->getEngine().getNbBindings());

    // PRINT_CONTEXT(context);

    // PRINT_ENGINE(engine);

    

    // 先释放context，后释放engine
    context->destroy();
    engine->destroy();

    return 0;
}
void doInference(IExecutionContext* context, float* input, float* output) {
    /**
     * 
     * key: context->enqueueV2(d_buffers, 0, nullptr);
     *      使用engine创建的context进行推理
     *      d_buffers[num_inputs + num_outputs], 每个元素都是一个tensor，其实就是一维数组
     *          inputs在前，outputs在后，必须是设备内存
     *      0: 代表默认流
     * 
     * */
    
}

void preprocess(cv::Mat src, float *data) {
    static const float norm_means[] = {0.406, 0.456, 0.485}; // src
    static const float norm_stds[] = {0.225, 0.224, 0.229};
    static const int INPUT_H = 640;
    static const int INPUT_W = 640;
    // 1.resize
    cv::resize(src, src, cv::Size(INPUT_W, INPUT_H));

    // 2.uchar->CV_32F, scale to [0,1]
    src.convertTo(src, CV_32F);
    src /= 255.0;

    // 3.split R,G,B and normal each channel using norm_means,norm_stds
    vector<cv::Mat> channels;
    cv::split(src, channels);
    cv::Scalar means, stds;
    for (int i = 0; i < 3; ++i) {
        cv::Mat a = channels[i]; // b
        cv::meanStdDev(a, means, stds);
        a = a / stds.val[0] * norm_stds[i]; // change std, mean also change
        means = cv::mean(a); // recompute mean!
        a = a - means.val[0] + norm_means[i];
        channels[i] = a;
    }


    // 4.pass to data, ravel()
    int index = 0;
    for (int c = 2; c >= 0; --c) { // R,G,B
        for (int h = 0; h < INPUT_H; ++h) {
            for (int w = 0; w < INPUT_W; ++w) {
                data[index] = channels[c].at<float>(h, w); // R->G->B
                index++;
            }
        }
    }

}