#pragma once

#ifdef ACCELERATE_ONNX_RUNTIME_OPEN_VINO_GPU

#include "OnnxAbstractInference.h"
#include <CL/cl.hpp>

struct OpenCL {
    cl::Context _context;
    cl::Device _device;
    cl::CommandQueue _queue;

    explicit OpenCL(const std::shared_ptr<std::vector<cl_context_properties>> &media_api_context_properties = nullptr) {

        const unsigned int refVendorID = 0x8086;
        cl_uint n = 0;
        clGetPlatformIDs(0, nullptr, &n);

        // Get platform list
        std::vector<cl_platform_id> platform_ids(n);
        clGetPlatformIDs(n, platform_ids.data(), nullptr);

        for (auto &id: platform_ids) {
            cl::Platform platform = cl::Platform(id);
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            for (auto &d: devices) {
                if (refVendorID == d.getInfo<CL_DEVICE_VENDOR_ID>()) {
                    _device = d;
                    _context = cl::Context(_device);
                    break;
                }
            }
        }
        cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        _queue = cl::CommandQueue(_context, _device, props);
    }

    explicit OpenCL(cl_context context) {
        _context = cl::Context(context);
        _device = cl::Device(_context.getInfo<CL_CONTEXT_DEVICES>()[0]);
        cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        _queue = cl::CommandQueue(_context, _device, props);
    }
};

class OnnxOpenVinoGpuInference final : public OnnxAbstractInference {
private:
    std::unique_ptr<OpenCL> open_cl;
protected:
    Ort::SessionOptions build_session_options() override;

    Ort::MemoryInfo build_memory_info() override;

public:
    explicit OnnxOpenVinoGpuInference(const std::string &model_path);

    [[nodiscard]] out_tensor predict(const cv::Mat &image) const override;

    [[nodiscard]] out_parallel_tensors predict_all(const std::vector<cv::Mat> &images) const override;
};

#endif
