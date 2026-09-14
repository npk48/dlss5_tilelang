#pragma once
#include <vector>
#include <string>
#include <cstdint>
namespace d5server {
struct Image {uint32_t width=0,height=0,channels=4; std::vector<float> rgb;};
Image decode_image(const std::string& bytes);
std::string encode_png(const Image& image);
}
