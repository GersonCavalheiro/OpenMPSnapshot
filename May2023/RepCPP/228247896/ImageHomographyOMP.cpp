#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

using namespace std;
using namespace std::filesystem;
using namespace cv;

inline int main_omp(int argc, char* argv[]) {
int x, y, x1, y1;
double h2w_ratio;
Mat t, img;
Matx<double, 4, 1> Xp, Yp;

String image_path;
if (argc == 11)
image_path = argv[1];
else
image_path = "tests/test.png";

if (!exists(image_path)) throw "No such a file!";

img = imread(image_path);

if (img.empty()) throw "Could not read the image";

if (argc == 11) {
h2w_ratio = atof(argv[2]);
Xp = { atof(argv[3]), atof(argv[4]), atof(argv[5]), atof(argv[6]) };
Yp = { atof(argv[7]), atof(argv[8]), atof(argv[9]), atof(argv[10]) };
}
else {
h2w_ratio = 9. / 16;
Xp = { 299, 1096, 1197, 84 };
Yp = { 134, 57, 768, 592 };
}


Matx<double, 4, 1> X, Y;

const double center_x = sum(Xp)[0] / 4;
const double center_y = sum(Yp)[0] / 4;

uint8_t r[2][2];
for (uint8_t i = 0; i < 4; ++i) {
r[Xp(i, 0) >= center_x][Yp(i, 0) >= center_y] = i;
}

double  _width1 = min(abs(Xp(r[0][0]) - Xp(r[1][0])), abs(Xp(r[0][1]) - Xp(r[1][1])));
double _height1 = min(abs(Yp(r[0][0]) - Yp(r[0][1])), abs(Yp(r[1][0]) - Yp(r[1][1])));
int width1 = static_cast<int>(max(1., _width1));
int height1 = static_cast<int>(max(1., min(h2w_ratio * width1, _height1)));

for (uint8_t i = 0; i < 2; ++i) {
for (uint8_t j = 0; j < 2; ++j) {
X(r[i][j]) = i * width1;
Y(r[i][j]) = j * height1;
}
}

Mat B;
hconcat(X, Y, B);
hconcat(B, Mat::ones(4, 1, CV_64FC1), B);
hconcat(B, Mat::zeros(4, 3, CV_64FC1), B);
hconcat(B, -Xp.mul(X), B);
hconcat(B, -Xp.mul(Y), B);
hconcat(B, Mat::zeros(4, 3, CV_64FC1), B);
hconcat(B, X, B);
hconcat(B, Y, B);
hconcat(B, Mat::ones(4, 1, CV_64FC1), B);
hconcat(B, -Yp.mul(X), B);
hconcat(B, -Yp.mul(Y), B);

B = B.reshape(0, 8);

Mat D;
hconcat(Xp, Yp, D);
D = D.reshape(0, 8);

Mat l = (B.t() * B).inv() * B.t() * D;

Mat A;
hconcat(l(Range(0, 6), Range(0, 1)).t(), Matx<double, 1, 3>(0, 0, 1), A);
A = A.reshape(0, 3);

Mat C;
hconcat(l(Range(6, 8), Range(0, 1)).t(), Matx<double, 1, 1>(1), C);

auto tick = chrono::high_resolution_clock::now();

Mat img1 = Mat(height1, width1, CV_8UC3);

#pragma omp parallel for collapse(2) shared(img, img1, A, C) private(x, x1, y, y1, t)	
for (y = 0; y < img1.rows; ++y) {
for (x = 0; x < img1.cols; ++x) {
t = A * Matx<double, 3, 1>(x, y, 1) / (C * Matx<double, 3, 1>(x, y, 1));
x1 = static_cast<int>(round(t.at<double>(0, 0)));
y1 = static_cast<int>(round(t.at<double>(1, 0)));
if (x1 >= 0 && y1 >= 0 && x1 < img.cols && y1 < img.rows) {
img1.at<Vec3b>(Point(x, y)) = img.at<Vec3b>(Point(x1, y1));
}
}
}
auto tock = chrono::high_resolution_clock::now();

cout << chrono::duration_cast<chrono::nanoseconds>(tock - tick).count() / 1.e9 << endl;



return 0;
}