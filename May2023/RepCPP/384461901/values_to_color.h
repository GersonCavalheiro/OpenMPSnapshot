
#pragma once

#include "tinycolormap.hpp"

#include <vector>
#include <algorithm>



template<class T>
std::vector<T> normalize(const std::vector<T> data) {
if(data.size() < 2) {
throw std::invalid_argument("The 'data' must contain at least 2 elements.");
}
T min = *std::min_element(data.begin(), data.end());
T max = *std::max_element(data.begin(), data.end());
if(min == max) {
throw std::invalid_argument("The 'data' must contain at least 2 unique elements.");
}
T range = max - min;

std::vector<T> scaled(data.size());
for(size_t i=0; i<data.size(); i++) {
scaled[i] = (data[i] - min) / range;
}
return(scaled);
}


template<class T>
std::vector<uint8_t> data_to_colors(const std::vector<T> data, const tinycolormap::ColormapType cmap = tinycolormap::ColormapType::Viridis) {
std::vector<T> dnorm = normalize(data);
std::vector<uint8_t> colors;
colors.reserve(data.size() * 3);    
for(size_t i=0; i<dnorm.size(); i++) {
tinycolormap::Color color = tinycolormap::GetColor(dnorm[i], cmap);
colors.push_back(color.ri());
colors.push_back(color.gi());
colors.push_back(color.bi());
}
return(colors);
}

template<>
std::vector<uint8_t> data_to_colors(const std::vector<float> data, const tinycolormap::ColormapType cmap) {
return(data_to_colors(std::vector<double>(data.begin(), data.end()), cmap));
}

template<>
std::vector<uint8_t> data_to_colors(const std::vector<int> data, const tinycolormap::ColormapType cmap) {
return(data_to_colors(std::vector<double>(data.begin(), data.end()), cmap));
}








