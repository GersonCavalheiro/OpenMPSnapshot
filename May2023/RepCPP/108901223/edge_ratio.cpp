#include "edge_ratio.hpp"

#include <atomic>
#include <numeric>
#include <parallel/numeric>

double edge_ratio(cv::Mat_<uint8_t> const& dilated_frame,
cv::Mat_<uint8_t> const& threshed_next_frame) {

int edges_in_frame      = 0,
similar_edges_count = std::inner_product(
dilated_frame.begin(),
dilated_frame.end(),
threshed_next_frame.begin(),
0,
std::plus<>(),
[&edges_in_frame](uint8_t frame_pixel, uint8_t next_frame_pixel) {
if(next_frame_pixel != 0) {
++edges_in_frame;
if(frame_pixel != 0) {
return 1;
}
}
return 0;
});

if(edges_in_frame == 0) {
return 1;
} else {
return static_cast<double>(similar_edges_count) / static_cast<double>(edges_in_frame);
}
}

double edge_ratio_omp(cv::Mat_<uint8_t> const& dilated_frame,
cv::Mat_<uint8_t> const& threshed_next_frame) {

int edges_in_frame = 0, similar_edges_count = 0;

#pragma omp parallel for reduction(+ : edges_in_frame) reduction(+ : similar_edges_count)
for(int i = 0; i < dilated_frame.rows * dilated_frame.cols; ++i) {

if(threshed_next_frame(i) != 0) {
++edges_in_frame;
if(dilated_frame(i) != 0) {
++similar_edges_count;
}
}
}

if(edges_in_frame == 0) {
return 1;
} else {
return static_cast<double>(similar_edges_count) / static_cast<double>(edges_in_frame);
}
}
