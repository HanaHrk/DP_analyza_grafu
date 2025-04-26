#include <inferencetools/LibTorchEngineSequential.hpp>
#include <inferencetools/exception/LibTorchBuildException.hpp>
#include <inferencetools/exception/LibTorchRuntimeException.hpp>


void LibtorchEngineSequential::loadModel(const std::string& enginePath)
{
    try
    {
        model_ = std::make_unique<torch::jit::Module>(torch::jit::load(enginePath));
        model_->eval();
    }
    catch (const c10::Error& e)
    {
        throw LibTorchBuildException("Error during LibTorch model loading: " + e.msg());
    }
}

Tensor LibtorchEngineSequential::predict(const InferInput& input) const
{
    try
    {
        const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        const torch::Tensor inputTensor = torch::from_blob(const_cast<float*>(input.input.data()),
                                                           {1, 3, 224, 224},
                                                           options).clone();
        const torch::Tensor output = model_->forward({torch::jit::IValue(inputTensor)}).toTensor();
        std::vector outputTensor(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
        return outputTensor;
    }
    catch (c10::Error& e)
    {
        throw LibTorchRuntimeException("Error during LibTorch predicting: " + e.msg());
    }
}
