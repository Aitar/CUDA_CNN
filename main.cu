#include <iostream>
#include <queue>
#include "mnist.h"
#include "src/utils.cuh"
#include "src/autograd.h"

using namespace std;
using namespace cuDL;

int main() {
    auto cuda = make_shared<CudaContext>();
    float aa[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto tensorA = make_shared<Tensor>(3, 1, 1, 3, aa);
    float bb[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    auto tensorB = make_shared<Tensor>(3, 1, 1, 3, bb);
    auto tensorC = make_shared<Tensor>(1, 1, 1, 1);
    tensorC->valueInit(2);

    auto idenA = make_shared<Identity>(tensorA, cuda);
    auto idenB = make_shared<Identity>(tensorB, cuda);
    auto idenGrad = make_shared<Identity>(tensorC, cuda);

    auto add = make_shared<Add>(nodes{idenA, idenB}, 1, 1, cuda);
    auto mul = make_shared<Mul>(nodes{add, idenA}, cuda);
    auto gemm = make_shared<Gemm>(nodes{mul, idenB}, cuda);
    auto sum = make_shared<Sum>(nodes{gemm}, cuda);

    Executor executor(sum, cuda);
    auto res = executor.forward();
    res->print();
    auto grad = make_shared<Tensor>(1, 1, 1, 1);
    grad->valueInit(2);
    executor.backward(grad);
    executor.print();

//    const string trainDataPath = "/nfs/volume-73-1/huangdingli/my_workspace/cuda/myNN/data/train-images-idx3-ubyte";
//    const string trainLabelPath = "/nfs/volume-73-1/huangdingli/my_workspace/cuda/myNN/data/train-labels-idx1-ubyte";
//    const string testDataPath = "/nfs/volume-73-1/huangdingli/my_workspace/cuda/myNN/data/t10k-images-idx3-ubyte";
//    const string testLabelPath = "/nfs/volume-73-1/huangdingli/my_workspace/cuda/myNN/data/t10k-labels-idx1-ubyte";
//
//    const int batchSize = 2;
//    const int maxEpoch = 10;
//    const int testFrequency = 10;
//    const int printFrequency = 1;
//    const float lr = 0.01;
//
//    auto trainDataset = make_shared<MNIST>(trainDataPath, trainLabelPath);
//    auto testDataset = make_shared<MNIST>(testDataPath, testLabelPath);
//    auto trainDataLoader = new DataLoader(trainDataset, batchSize, true);
//    auto testDataLoader = new DataLoader(testDataset, batchSize, true);
//    auto model = new Net(10);
//    model->setLr(lr);
//    auto cudaCtx = make_shared<CudaContext>();
//    model->init(cudaCtx);
//
//    int trainSteps = trainDataset->getLen() / batchSize + 1;
//    int testSteps = testDataset->getLen() / batchSize + 1;
//
//    auto trainLogger = new MerticsLogger(maxEpoch, trainSteps, printFrequency);
//    auto testLogger = new MerticsLogger(maxEpoch, testSteps, printFrequency);
//
//
//    for (int epoch = 0; epoch < maxEpoch; ++epoch) {
//        model->train();
//        for (int step = 0; step < trainSteps; ++step) {
//            auto pair = trainDataLoader->getData(step);
//            auto pre = model->forward(pair[0]);
//            float lossValue = model->backward(pair[1]);
//            trainLogger->log(epoch, step, lossValue, pre, pair[1]);
//        }
//
//        if (epoch % testFrequency == 0) {
//            model->infer();
//            for (int step = 0; step < testSteps; ++step) {
//                auto pair = testDataLoader->getData(step);
//                auto pre = model->forward(pair[0]);
//                float lossValue = model->backward(pair[1]);
//                testLogger->log(epoch, step, lossValue, pre, pair[1]);
//            }
//        }
//        trainLogger->newEpoch();
//    }

    return 0;
}
