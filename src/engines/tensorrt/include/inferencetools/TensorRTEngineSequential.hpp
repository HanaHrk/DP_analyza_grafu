#pragma once

#include <inferencetools/InferenceEngine.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>

class TensorRTLogger final : public nvinfer1::ILogger
{
public:
    explicit TensorRTLogger(const Severity& severity = Severity::kINFO)
    {
        this->severity_ = severity;
    }

    void log(Severity severity, const char* msg) noexcept override
    {
        if (static_cast<int32_t>(severity) <= static_cast<int32_t>(this->severity_))
        {
            const std::vector<std::string> ERRORS = {
                "FATAL   ",
                "ERROR   ",
                "WARNING ",
                "INFO    ",
                "DEBUG   ",
            };
            std::cout << ERRORS[static_cast<int32_t>(severity)] << " : " << msg << std::endl;
        }
    }

private:
    Severity severity_;
};

class TensorRTEngineSequential final : public InferenceEngineSequential
{
public:
    ~TensorRTEngineSequential() override;

    void loadModel(const std::string& modelPath) override;

    [[nodiscard]] Tensor predict(const InferInput& input) const override;

private:
    std::unique_ptr<TensorRTLogger> logger_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    [[nodiscard]] long getInputTensorSize(const InferInput& input) const;

    [[nodiscard]] long getOutputSize(const InferInput& input) const;

    [[nodiscard]] std::string getInputTensorName() const;

    [[nodiscard]] std::string getOutputTensorName() const;

    static nvinfer1::ICudaEngine* buildTensorRTEngine(const std::string& enginePath, nvinfer1::ILogger& logger);
};
