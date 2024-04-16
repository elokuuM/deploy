#include <infer.hpp>
#include <nms.h>
using namespace std;
using namespace sample;

int inferEngine(char* ege_file_path, string input_path) {
    cout<<"running infer..."<<endl;

    cout<<"plan file path:"<<ege_file_path<<endl;

    std::vector<cv::String> file_names;
    cv::glob(input_path, file_names);

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

    cout<< "start infer "<< file_names.size()<<" images"<<endl;
    float* input = (float*)malloc(3*640*640*sizeof(float));

    // float* output = (float*)malloc(8400*84*sizeof(float)); 
    float* output=NULL;
    cudaMalloc((void **)&output, 8400*84*sizeof(float));
    size_t nmsout_size = 1+1024*7*sizeof(float);
    float* nmsout_d=NULL;
    cudaMalloc((void **)&nmsout_d, nmsout_size);
    float* nmsout_h = (float*)malloc(nmsout_size);

    float latency=0;
    for (int i = 0; i < file_names.size(); i++){
        
        clock_t start, end;
        start = clock();

        cv::Mat img = cv::imread(file_names[i], cv::IMREAD_COLOR);
        cv::Mat img2;
        cv::resize(img, img2, cv::Size(640, 640));
        preprocess(img, input); //cv::Mat to float*

        doInference(context, input, output);


        cudaMemset(nmsout_d, 0, nmsout_size);
        memset(nmsout_h, 0, nmsout_size);
        decode_kernel_invoker(output, 8400, 80, 0.25, 0.7, nmsout_d, 1024);
        cudaMemcpy(nmsout_h, nmsout_d, nmsout_size, cudaMemcpyDeviceToHost);
        std::vector<detect_result> results;
        nmspostprocess(nmsout_h, results);
        end = clock();
        float second = float(end-start) / CLK_TCK;
        if(latency==0)
            latency = second;
        else  
            latency = (latency + second)/2;
        draw_frame(img2, results);
        cout<<"["<<i+1<<"/"<<file_names.size()<<"] "<<"./pred/" + Getfilename(input_path, file_names[i])<<" latency = "<<latency*1000<<"ms"<<endl;
        cv::imwrite("./workspace/Debug/pred/" + Getfilename(input_path, file_names[i]), img2);


        // std::vector<detect_result> results;
        // yolopostprocess(output, results);
        // end = clock();
        // float second = float(end-start) / CLK_TCK;
        // if(latency==0)
        //     latency = second;
        // else  
        //     latency = (latency + second)/2;
        // draw_frame(img2, results);
        // cout<<"["<<i+1<<"/"<<file_names.size()<<"] "<<"./pred/" + Getfilename(input_path, file_names[i])<<" latency = "<<latency*1000<<"ms"<<endl;
        // cv::imwrite("./workspace/Debug/pred/" + Getfilename(input_path, file_names[i]), img2);

    }
    free(input);

    // free(output);

    cudaFree(output);
    free(nmsout_h);
    cudaFree(nmsout_d);
    // PRINT_CONTEXT(context);

    // PRINT_ENGINE(engine);

    // 先释放context，后释放engine
    context->destroy();
    engine->destroy();

    return 0;
}

void doInference(IExecutionContext* context, float* input, float* output) {
    void *d_buffers[2]; //d_buffers是一个数组，数组里存的是void类型的指针。

    for (int i = 0; i < context->getEngine().getNbBindings(); i++)
    {
        nvinfer1::Dims dims = context->getBindingDimensions(i);
        // printf("dims = {%d, %d, %d, %d} \n", dims.d[0], dims.d[1], dims.d[2], dims.d[3]);
        int volume = dims.d[0] * dims.d[1] * dims.d[2] * (dims.d[3]?dims.d[3]:1) * sizeof(float);

        cudaMalloc((void **)&d_buffers[i], volume);
        if (context->getEngine().bindingIsInput(i))
        {
            cudaMemcpy(d_buffers[i], input, volume, cudaMemcpyHostToDevice);
        }
        else
            cudaMemset(d_buffers[i], 0, volume);
    }
    // printf("malloc %d buffers\n", context->getEngine().getNbBindings());

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
        // printf("output dims = {%d, %d, %d, %d} \n", dims.d[0], dims.d[1], dims.d[2], dims.d[3]);
        int volume = dims.d[0] * dims.d[1] * dims.d[2] * (dims.d[3]?dims.d[3]:1) * sizeof(float);
        cudaMemcpy(output, d_buffers[i], volume, cudaMemcpyDeviceToDevice);

        // printf("buffers[%d]: \n", i);
        // for (int _idx = 0; _idx < volume / sizeof(float) && _idx < 5; ++_idx)
        // {
        //     printf("%f \n", output[_idx]);
        // }
    }

    for (int i = 0; i < context->getEngine().getNbBindings(); i++)
    {
        cudaFree(d_buffers[i]);
    }
    // printf("release %d buffers\n", context->getEngine().getNbBindings());
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

void yolopostprocess(float* netout, std::vector<detect_result> &results) {
    const float confidence_threshold_ =0.25f;
    const float nms_threshold_ = 0.4f;
    cv::Mat det_output(8400, 84, CV_32FC1);
    memcpy((uchar*)det_output.data, netout, 8400*84* sizeof(float));
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;
    for (int i = 0; i < det_output.rows; i++)
    {
        cv::Mat classes_confidences = det_output.row(i).colRange(4, 84);
        cv::Point classIdPoint;
        double cls_conf;
        cv::minMaxLoc(classes_confidences, 0, &cls_conf, 0, &classIdPoint);


        if (cls_conf > confidence_threshold_)
        {
            float cx = det_output.at<float>(i, 0);
            float cy = det_output.at<float>(i, 1);
            float ow = det_output.at<float>(i, 2);
            float oh = det_output.at<float>(i, 3);
            int x = static_cast<int>((cx - 0.5 * ow) * 1);
            int y = static_cast<int>((cy - 0.5 * oh) * 1);
            int width = static_cast<int>(ow * 1);
            int height = static_cast<int>(oh * 1);
            cv::Rect box;
            box.x = x;
            box.y = y;
            box.width = width;
            box.height = height;

            boxes.push_back(box);
            classIds.push_back(classIdPoint.x);
            confidences.push_back(cls_conf);
        }
    }

    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold_, nms_threshold_, indexes);
    for (size_t i = 0; i < indexes.size(); i++)
    {
        detect_result dr;
        int index = indexes[i];
        int idx = classIds[index];
        dr.box = boxes[index];
        dr.classId = idx;
        dr.confidence = confidences[index];
        results.push_back(dr);
    }
}

void nmspostprocess(float* nmsout_d, std::vector<detect_result> &results) {
    cv::Mat det_output(1024, 7, CV_32FC1);
    memcpy((uchar*)det_output.data, nmsout_d, 1 + nmsout_d[0]*7* sizeof(float));

    for (size_t i = 0; i < nmsout_d[0]; i++)
    {   
        float* boxid = nmsout_d + 1 + 7*i;
        if(boxid[6]==0){
            continue;
        }

        detect_result dr;

        cv::Rect box;
        box.x = boxid[0];
        box.y = boxid[1];
        box.width = boxid[2] - boxid[0];
        box.height = boxid[3] - boxid[1];

        dr.box = box;
        dr.classId = boxid[5];
        dr.confidence = boxid[4];
        results.push_back(dr);
    }
}

void draw_frame(cv::Mat & frame, std::vector<detect_result> &results)
{
    static const string class_names[] = {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush"
	};

    for(auto dr : results)
    {
        // cout<< class_names[dr.classId ]<<endl;
        cv::rectangle(frame, dr.box , cv::Scalar(0, 0, 255), 2, 8);
        cv::rectangle(frame, cv::Point(dr.box .tl().x, dr.box .tl().y - 20), cv::Point(dr.box .br().x, dr.box .tl().y), cv::Scalar(255, 0, 0), -1);

        std::string label = cv::format("%.2f", dr.confidence);
        label = class_names[dr.classId ] + ":" + label;

        cv::putText(frame, label, cv::Point(dr.box.x, dr.box.y + 6), 1, 2, cv::Scalar(0, 255, 0),2);

    }
}

std::string Getfilename(std::string &Orgfolder, cv::String &Orgfilenames)
{
	char a[20];//保存字符
	int i = 0;
	int n = Orgfilenames.size() - Orgfolder.size();//文件名长度
	for ( i; i < 16; i++)
	{
		a[i] = Orgfilenames[Orgfolder.size() + i];
	}
	a[i] = '\0';//结束标志
	std::string filename = a;//构造String类型数据
	return filename;
}
