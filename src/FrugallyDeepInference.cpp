#ifdef ACCELERATE_FRUGALLY_DEEP

#include "ImageUtils.h"
#include "FrugallyDeepInference.h"
#include "ThreadUtils.h"

constexpr int NS_TO_MS = 1000000.0f;

FrugallyDeepInference::FrugallyDeepInference(const std::string &model_path,
                                             const ParallelStrategy &parallel_strategy = ParallelStrategy::STD_THREADING)
        : model_(fdeep::load_model(model_path)),
          parallel_strategy_(parallel_strategy) {
}

void FrugallyDeepInference::set_parallel_strategy(const ParallelStrategy &parallel_strategy) {
    this->parallel_strategy_ = parallel_strategy;
}


FrugallyDeepInference::~FrugallyDeepInference() = default;

OutTensor FrugallyDeepInference::predict(const cv::Mat &image) const {
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

void FrugallyDeepInference::run_parallel(const std::vector<cv::Mat> &images,
                                         std::vector<OutParTensor> &out_parallel_tensors,
                                         std::atomic_int &index, std::atomic_int &max_threads) const {
    const auto start = std::chrono::high_resolution_clock::now();
    if (parallel_strategy_ == ParallelStrategy::STD_PARALLEL_FOREACH) {
        std::cout << "Parallel foreach" << std::endl;
        std::for_each(std::execution::par_unseq, images.begin(), images.end(),
                      [this, &out_parallel_tensors, start, &index, &max_threads](const auto &image) {
                          this->inference_callback()(start, out_parallel_tensors, index, max_threads, image);
                      });
    } else if (parallel_strategy_ == ParallelStrategy::STD_THREADING) {
        std::cout << "Parallel threads" << std::endl;
        std::vector<std::shared_future<void>> futures;
        futures.reserve(images.size());
        for (const auto &image: images) {
            auto future = std::async(std::launch::async,
                                     [this, &out_parallel_tensors, start, &index, &max_threads, &image]() {
                                         this->inference_callback()(start, out_parallel_tensors, index, max_threads,
                                                                    image);
                                     });
            futures.push_back(future.share());
        }
        for (auto &future: futures) {
            future.wait();
        }
    }
}

OutParTensors FrugallyDeepInference::predict_all(const std::vector<cv::Mat> &images) const {
    OutParTensors out_parallel_tensors;
    std::atomic_int max_threads;
    std::vector<OutParTensor> out_parallel_tensor_vector;
    out_parallel_tensor_vector.resize(images.size());
    std::vector<std::shared_future<OutParTensor>> futures;
    const auto start = std::chrono::high_resolution_clock::now();
    std::atomic<int> index(0);
    this->run_parallel(images, out_parallel_tensor_vector, index, max_threads);
    const auto end = std::chrono::high_resolution_clock::now();
    out_parallel_tensors.milliseconds = static_cast<float>((end - start).count()) / NS_TO_MS;
    out_parallel_tensors.out_tensors = out_parallel_tensor_vector;
    std::cout << "Max Thread Count: " << max_threads.load() << std::endl;
    return out_parallel_tensors;
}

std::function<void(const std::chrono::time_point<std::chrono::steady_clock> &,
                   std::vector<OutParTensor> &, std::atomic_int &, std::atomic_int &,
                   const cv::Mat &)> FrugallyDeepInference::inference_callback() const {
    return [this](const std::chrono::time_point<std::chrono::steady_clock> &start,
                  std::vector<OutParTensor> &out_par_tensors, std::atomic_int &index, std::atomic_int &max_threads,
                  const cv::Mat &image) {
#if WIN32
        if(index.load() % 1000 == 500) {
            const auto number_of_threads = get_number_of_threads_current();
            if (number_of_threads > max_threads.load()) {
                max_threads.exchange(static_cast<int>(number_of_threads));
            }
        }
#endif
        OutParTensor out_parallel_tensor;
        const auto current_start = std::chrono::high_resolution_clock::now();
        auto vector = to_tensor(image);
        const auto out_tensor = this->model_.predict({vector});
        const auto current_end = std::chrono::high_resolution_clock::now();
        out_parallel_tensor.predictions = out_tensor[0].to_vector();
        out_parallel_tensor.offset_milliseconds = static_cast<float>((current_start - start).count()) / NS_TO_MS;
        out_parallel_tensor.milliseconds = static_cast<float>((current_end - current_start).count()) / NS_TO_MS;
        const auto current_index = index.fetch_add(1);
        out_par_tensors[current_index] = out_parallel_tensor;
    };
}

fdeep::tensor FrugallyDeepInference::to_tensor(const cv::Mat &image) const {
    const auto input_shapes = this->model_.get_input_shapes();
    const auto &input_shape = input_shapes.at(0);
    const auto height = input_shape.height_.get_with_default(0);
    const auto width = input_shape.width_.get_with_default(0);
    const auto depth = input_shape.depth_.get_with_default(0);

    const auto converted_image = modify_image(image, static_cast<int>(width), static_cast<int>(height));
    const auto image_data = to_vector_input(converted_image, static_cast<int>(width), static_cast<int>(height));
    auto tensor = fdeep::tensor(fdeep::tensor_shape{height, width, depth}, image_data);
    return tensor;
}

#endif
