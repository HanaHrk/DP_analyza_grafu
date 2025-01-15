#pragma once
#if ACCELERATE_FRUGALLY_DEEP

#include <fdeep/fdeep.hpp>
#include "../AbstractInference.h"

class FrugallyDeepInference final : public AbstractInference {
    fdeep::model model_;

    fdeep::tensor to_tensor(const cv::Mat &image) const;
public:

    explicit FrugallyDeepInference(const std::string &model_path);

    ~FrugallyDeepInference() override;


    /**
     * Generates predictions for a single input using the internal model.
     *
     * @param image A cv::Mat object representing the single input data to be processed.
     * @return An OutParTensor object containing the prediction and relevant computation details.
     */
    [[nodiscard]] OutTensor predict(const cv::Mat &image) const override;

    /**
     * Predicts outputs for a batch of images using the internal model.
     *
     * @param images A vector of cv::Mat objects representing the input images to be processed.
     * @return An OutParTensors object containing the predictions and timing information for the batch inference.
     */
    [[nodiscard]] OutParTensors predict_all(const std::vector<cv::Mat> &images) const override;
};
#endif

