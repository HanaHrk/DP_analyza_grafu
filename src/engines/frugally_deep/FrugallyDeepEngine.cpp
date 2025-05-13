#include <inferencetools/FrugallyDeepEngine.hpp>
#include <execution>
#include <inferencetools/exception/EngineRuntimeException.hpp>


void FrugallyDeepEngine::loadModel(const std::string& enginePath)
{
    const auto model = fdeep::load_model(enginePath);
    this->model_ = std::make_unique<fdeep::model>(model);
}

Tensor FrugallyDeepEngine::predict(const InferInput& input) const
{
    const auto fdeepTensor = this->toFdeepTensor(input);
    const auto outputTensors = this->model_->predict({fdeepTensor});
    const auto floatVector = outputTensors.at(0).to_vector();
    return Tensor{floatVector};
}

std::vector<Tensor> FrugallyDeepEngine::predictAll(const std::vector<InferInput>& inputs, const ParallelMode& mode) const
{
    std::vector<Tensor> output(inputs.size());
    std::atomic_int index(0);

    if (mode == ParallelMode::PARALLEL_FOREACH)
    {
        std::for_each(std::execution::par_unseq, inputs.begin(), inputs.end(),
                      [this, &output, &index](const auto& input)
                      {
                          this->inferenceCallback()(input, output, index);
                      });
    }
    else if (mode == ParallelMode::STD_THREADING)
    {
        std::vector<std::shared_future<void>> futures;
        futures.reserve(inputs.size());
        for (const auto& input : inputs)
        {
            auto future = std::async(std::launch::async,
                                     [this, &output, &index, &input]
                                     {
                                         this->inferenceCallback()(input, output, index);
                                     });
            futures.push_back(future.share());
        }
        for (auto& future : futures)
        {
            future.wait();
        }
    }
    return output;
}

std::vector<Tensor> FrugallyDeepEngine::predictAll(const std::vector<InferInput>& input) const
{
    return this->predictAll(input, ParallelMode::PARALLEL_FOREACH);
}

std::function<void(const InferInput& input, std::vector<Tensor>& output, std::atomic_int& index)> FrugallyDeepEngine::inferenceCallback() const
{
    return [this](const InferInput& input, std::vector<Tensor>& output, std::atomic_int& index)
    {
        auto vector = this->toFdeepTensor(input);
        const auto out_tensor = this->model_->predict({vector});
        const auto current_index = index.fetch_add(1);
        output[current_index] = out_tensor[0].to_vector();
    };
}


ImageShape FrugallyDeepEngine::modelInputShape() const
{
    const auto inputShapes = this->model_->get_input_shapes();
    const auto& inputShape = inputShapes.at(0);
    const auto height = inputShape.height_.get_with_default(0);
    const auto width = inputShape.width_.get_with_default(0);
    const auto depth = inputShape.depth_.get_with_default(0);
    return ImageShape{height, width, depth};
}


fdeep::tensor FrugallyDeepEngine::toFdeepTensor(const InferInput& input) const
{
    auto inputWidth = input.inputWidth;
    auto inputHeight = input.inputHeight;
    auto inputDepth = input.inputDepth;
    const auto modelSize = this->modelInputShape();

    if (inputWidth == MODEL_PROPERTIES)
    {
        inputWidth = modelSize.width;
    }
    if (inputHeight == MODEL_PROPERTIES)
    {
        inputHeight = modelSize.height;
    }
    if (inputDepth == MODEL_PROPERTIES)
    {
        inputDepth = modelSize.depth;
    }
    if (inputWidth == UNKNOWN_PROPERTY || inputHeight == UNKNOWN_PROPERTY || inputDepth == UNKNOWN_PROPERTY)
    {
        throw EngineRuntimeException("Input Shape is not defined properly. Model or user must define input shape for inference.");
    }
    const auto shape = fdeep::tensor_shape(inputWidth, inputHeight, inputDepth);
    return fdeep::tensor{shape, input.input};
}
