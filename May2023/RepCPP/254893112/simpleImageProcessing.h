
#ifndef SIMPLEIMAGEPROCESSING__SIMPLEIMAGEPROCESSING_H_
#define SIMPLEIMAGEPROCESSING__SIMPLEIMAGEPROCESSING_H_
#include <string>
#include <thread>
#include <deque>

class simpleImageProcessing {
private:
std::vector<std::string> filePaths;

static void RunParallel(const long start, const long end, const std::vector<std::string> *filePathsPointer) {
std::vector<cv::Mat> images(filePathsPointer->size());

for (int i = start; i < end; i++) {
images[i] = cv::imread((*filePathsPointer)[i]);
cv::GaussianBlur(images[i], images[i], cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
cv::Sobel(images[i], images[i], -1, 1, 1);
std::string res = "img/" + std::to_string(i) + ".jpg";
cv::imwrite(res, images[i]);
}
}

public:
simpleImageProcessing(std::string imgPath, long repetition) {
for (int i = 0; i < repetition; i++) {
for (const auto &entry : std::filesystem::directory_iterator(imgPath)) {
filePaths.emplace_back(entry.path());
}
}
};
void sequential() {

std::vector<cv::Mat> images(filePaths.size());

auto start = std::chrono::high_resolution_clock::now();

for (int i = 0; i < filePaths.size(); i++) {

images[i] = cv::imread(filePaths[i]);

cv::GaussianBlur(images[i], images[i], cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
cv::Sobel(images[i], images[i], -1, 1, 1);

std::string res = "img/" + std::to_string(i) + ".jpg";
cv::imwrite(res, images[i]);
}
auto end = std::chrono::system_clock::now();
auto elapsed = end - start;
std::cout << "Total sequential time: "
<< std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "ms"
<< std::endl;

};
void par(const long nw) {
std::vector<std::thread> coda(nw);
auto startTime = std::chrono::high_resolution_clock::now();
long stepElem = std::floor((float) (filePaths.size()) / (float) nw);
long remaining = filePaths.size() % nw;
long start = 0;
long end = 0;

for (int i = 0; i < nw; ++i) {
start = end;
end += stepElem + (remaining ? 1 : 0);
if (remaining)
remaining--;

coda.emplace_back(std::thread(&simpleImageProcessing::RunParallel,
start,
end, &filePaths));

}

for (auto &it : coda) {
if (it.joinable())
it.join();
}

auto endTime = std::chrono::system_clock::now();
auto elapsed = endTime - startTime;
std::cout << "Number worker: " << nw << " Total parallel time with c++ thread: "
<< std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
<< std::endl;
};

void openMPImpl(long nw) {
std::vector<cv::Mat> images(filePaths.size());

auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(nw) schedule(dynamic, 1)
for (int i = 0; i < filePaths.size(); i++) {
images[i] = cv::imread(filePaths[i]);
cv::GaussianBlur(images[i], images[i], cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
cv::Sobel(images[i], images[i], -1, 1, 1);
std::string res = "img/" + std::to_string(i) + ".jpg";
cv::imwrite(res, images[i]);
}

auto end = std::chrono::system_clock::now();
auto elapsed = end - start;
std::cout << "Number worker: " << nw << " Total parallel time with OpenMP: "
<< std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
<< std::endl;

};

};

#endif 
