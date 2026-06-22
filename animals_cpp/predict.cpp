#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

torch::Tensor preprocess_image(const std::string& image_path) {
    // Load image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Error: Could not load image: " << image_path << "\n";
        exit(1);
    }

    // Convert BGR to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Resize to 224x224
    cv::resize(img, img, cv::Size(224, 224));

    // Convert to float and scale to [0,1]
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    // ImageNet normalization
    cv::subtract(img, cv::Scalar(0.485, 0.456, 0.406), img);
    cv::divide(img, cv::Scalar(0.229, 0.224, 0.225), img);

    // Convert HWC -> CHW and add batch dimension
    auto tensor = torch::from_blob(img.data, {224, 224, 3}, torch::kFloat);
    tensor = tensor.permute({2, 0, 1}).unsqueeze(0).clone();

    return tensor;
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: predict.exe <image_path>\n";
        std::cerr << "Example: predict.exe E:/test.jpg\n";
        return 1;
    }

    const std::string model_path = "E:/animals_cpp/mobilenetv3_scripted.pt";
    const std::vector<std::string> class_names = {
        "butterfly", "cat", "dog", "elephant", "horse"
    };

    // Load model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);
        model.eval();
        std::cout << "Model loaded successfully!\n";
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return 1;
    }

    // Preprocess real image
    std::string image_path = argv[1];
    std::cout << "Loading image: " << image_path << "\n";
    auto input_tensor = preprocess_image(image_path);

    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    at::Tensor output;
    {
        torch::NoGradGuard no_grad;
        output = model.forward(inputs).toTensor();
    }

    // Softmax
    auto probabilities = torch::softmax(output, 1);
    auto max_result = probabilities.max(1);
    auto max_prob = std::get<0>(max_result);
    auto predicted_idx = std::get<1>(max_result);

    int predicted_class = predicted_idx.item<int>();
    float confidence = max_prob.item<float>() * 100.0f;

    std::cout << "\n=== Results ===\n";
    std::cout << "Predicted: " << class_names[predicted_class] << "\n";
    std::cout << "Confidence: " << confidence << "%\n";

    std::cout << "\nAll probabilities:\n";
    for (int i = 0; i < (int)class_names.size(); i++) {
        float prob = probabilities[0][i].item<float>() * 100.0f;
        std::cout << "  " << class_names[i] << ": " << prob << "%\n";
    }

    return 0;
}