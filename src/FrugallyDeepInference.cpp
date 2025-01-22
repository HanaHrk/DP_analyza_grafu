#if ACCELERATE_FRUGALLY_DEEP
#include "ImageUtils.h"
#include "include/inference/FrugallyDeepInference.h"

constexpr int NS_TO_MS = 1000000.0f;

FrugallyDeepInference::FrugallyDeepInference(const std::string& model_path): model_(fdeep::load_model(model_path)),
                                                                             parallel_strategy_(
                                                                                 ParallelStrategy::STD_PARALLEL_FOREACH)
{
}

void FrugallyDeepInference::set_parallel_strategy(const ParallelStrategy& parallel_strategy)
{
    this->parallel_strategy_ = parallel_strategy;
}


FrugallyDeepInference::~FrugallyDeepInference() = default;

OutTensor FrugallyDeepInference::predict(const cv::Mat& image) const
{
    OutTensor out_tensor;
    const std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::high_resolution_clock::now();
    auto input_tensor = to_tensor(image);
    const auto output_tensors = this->model_.predict({input_tensor});
    const std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<float, std::milli> milliseconds(end - start);
    out_tensor.milliseconds = milliseconds.count();
    out_tensor.predictions = output_tensors[0].to_vector();
    return out_tensor;
}

void FrugallyDeepInference::run_parallel(const std::vector<cv::Mat>& images,
                                         std::vector<OutParTensor>& out_parallel_tensors,
                                         std::atomic<int>& index) const
{
    if (parallel_strategy_ == ParallelStrategy::STD_PARALLEL_FOREACH)
    {
        std::cout << "Parallel foreach" << std::endl;
        std::for_each(std::execution::par, images.begin(), images.end(),
                      [this, &out_parallel_tensors, &index](const auto& image)
                      {
                          this->inference_callback()(std::chrono::high_resolution_clock::now(), out_parallel_tensors, index,
                                                     image);
                      });
    }
    else if (parallel_strategy_ == ParallelStrategy::STD_THREADING)
    {
        std::cout << "Parallel threads" << std::endl;
        std::vector<std::shared_future<void>> futures;
        futures.reserve(images.size());
        for (const auto& image : images)
        {
            auto future = std::async(std::launch::async, [this, &out_parallel_tensors, &index, &image]()
            {
                this->inference_callback()(std::chrono::high_resolution_clock::now(), out_parallel_tensors, index, image);
            });
            futures.push_back(future.share());
        }
        for (auto& future : futures)
        {
            future.wait();
        }
    }
}

OutParTensors FrugallyDeepInference::predict_all(const std::vector<cv::Mat>& images) const
{
    OutParTensors out_parallel_tensors;
    std::vector<OutParTensor> out_parallel_tensor_vector;
    out_parallel_tensor_vector.resize(images.size());
    std::vector<std::shared_future<OutParTensor>> futures;
    const auto start = std::chrono::high_resolution_clock::now();
    std::atomic<int> index(0);
    this->run_parallel(images, out_parallel_tensor_vector, index);
    const auto end = std::chrono::high_resolution_clock::now();
    out_parallel_tensors.milliseconds = static_cast<float>((end - start).count()) / NS_TO_MS;
    out_parallel_tensors.out_tensors = out_parallel_tensor_vector;
    return out_parallel_tensors;
}

std::function<void(const std::chrono::time_point<std::chrono::steady_clock>,
                   std::vector<OutParTensor>&, std::atomic<int>&,
                   const cv::Mat&)> FrugallyDeepInference::inference_callback() const
{
    return [this](const std::chrono::time_point<std::chrono::steady_clock> start,
                  std::vector<OutParTensor>& out_par_tensors, std::atomic<int>& index,
                  const cv::Mat& image)
    {
        OutParTensor out_parallel_tensor;
        const auto current_start = std::chrono::high_resolution_clock::now();
        auto vector = to_tensor(image);
        const auto out_tensor = this->model_.predict({vector});
        const auto current_end = std::chrono::high_resolution_clock::now();
        out_parallel_tensor.predictions = out_tensor[0].to_vector();
        out_parallel_tensor.offset_milliseconds = static_cast<float>((current_start - start).count()) /
            NS_TO_MS;
        out_parallel_tensor.milliseconds = static_cast<float>((current_end - current_start).count()) /
            NS_TO_MS;
        const auto current_index = index.fetch_add(1);
        out_par_tensors[current_index] = out_parallel_tensor;
    };
}

fdeep::tensor FrugallyDeepInference::to_tensor(const cv::Mat& image) const
{
    const auto input_shapes = this->model_.get_input_shapes();
    const auto& input_shape = input_shapes.at(0);
    const auto height = input_shape.height_.get_with_default(0);
    const auto width = input_shape.width_.get_with_default(0);
    const auto depth = input_shape.depth_.get_with_default(0);

    const auto converted_image = modify_image(image, static_cast<int>(width), static_cast<int>(height));
    const auto image_data = to_vector_input(converted_image, static_cast<int>(width), static_cast<int>(height));
    auto tensor = fdeep::tensor(fdeep::tensor_shape{height, width, depth}, image_data);
    return tensor;
}

#endif
