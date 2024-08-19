#include <chrono> 
#include <iostream>
#include <math.h> 
#include <omp.h>
#include <opencv2/opencv.hpp>

using namespace std;

enum cube_faces { X_POS, X_NEG, Y_POS, Y_NEG, Z_POS, Z_NEG };

struct cart3D {
float x;
float y;
float z;
int faceIndex;
};

struct cart2D {
float x;
float y;
int faceIndex;
};

cv::Vec2f unit3DToUnit2D(float x, float y, float z, int faceIndex) {
float x2D, y2D;

if (faceIndex == X_POS) { 
x2D = y + 0.5;
y2D = z + 0.5;
} else if (faceIndex == X_NEG) { 
x2D = (y * -1) + 0.5;
y2D = z + 0.5;
} else if (faceIndex == Y_POS) { 
x2D = x + 0.5;
y2D = z + 0.5;
} else if (faceIndex == Y_NEG) { 
x2D = (x * -1) + 0.5;
y2D = z + 0.5;
} else if (faceIndex == Z_POS) { 
x2D = y + 0.5;
y2D = (x * -1) + 0.5;
} else { 
x2D = y + 0.5;
y2D = x + 0.5;
}

y2D = 1 - y2D;

return cv::Vec2f(x2D, y2D);
}

cart3D projectX(float theta, float phi, int sign) {
cart3D result;

result.x = sign * 0.5;
result.faceIndex = sign == 1 ? X_POS : X_NEG;
float rho = result.x / (cos(theta) * sin(phi));
result.y = rho * sin(theta) * sin(phi);
result.z = rho * cos(phi);
return result;
}

cart3D projectY(float theta, float phi, int sign) {
cart3D result;

result.y = -sign * 0.5;
result.faceIndex = sign == 1 ? Y_POS : Y_NEG;
float rho = result.y / (sin(theta) * sin(phi));
result.x = rho * cos(theta) * sin(phi);
result.z = rho * cos(phi);
return result;
}

cart3D projectZ(float theta, float phi, int sign) {
cart3D result;

result.z = sign * 0.5;
result.faceIndex = sign == 1 ? Z_POS : Z_NEG;
float rho = result.z / cos(phi);
result.x = rho * cos(theta) * sin(phi);
result.y = rho * sin(theta) * sin(phi);
return result;
}

cart2D convertEquirectUVtoUnit2D(float theta, float phi, int square_length) {
float x = cos(theta) * sin(phi);
float y = sin(theta) * sin(phi);
float z = cos(phi);

float maximum = max(abs(x), max(abs(y), abs(z)));
float xx = x / maximum;
float yy = -y / maximum;
float zz = z / maximum;

cart3D equirectUV;
if (xx == 1 or xx == -1) {
equirectUV = projectX(theta, phi, xx);
} else if (yy == 1 or yy == -1) {
equirectUV = projectY(theta, phi, yy);
} else {
equirectUV = projectZ(theta, phi, zz);
}

cv::Vec2f unit2D = unit3DToUnit2D(equirectUV.x, equirectUV.y, equirectUV.z,
equirectUV.faceIndex);

unit2D[0] *= square_length;
unit2D[1] *= square_length;

cart2D result;
result.x = (int)unit2D[0];
result.y = (int)unit2D[1];
result.faceIndex = equirectUV.faceIndex;

return result;
}

int main(int argc, char **argv) {
cv::Mat d_result, posY, posX, negY, negX, posZ, negZ;

std::string imgs_path(argv[1]);
posY = cv::imread(imgs_path + "posy.jpg");
posX = cv::imread(imgs_path + "posx.jpg");
negY = cv::imread(imgs_path + "negy.jpg");
negX = cv::imread(imgs_path + "negx.jpg");
negZ = cv::imread(imgs_path + "negz.jpg");
posZ = cv::imread(imgs_path + "posz.jpg");


const int output_width = posY.rows * 2;
const int output_height = posY.rows;
const int square_length = output_height;

cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);


cv::Mat destination(output_height, output_width, CV_8UC3,
cv::Scalar(255, 255, 255));

auto begin = chrono::high_resolution_clock::now();

#pragma omp parallel for
for (int j = 0; j < destination.cols; j++) {
for (int i = 0; i < destination.rows; i++) {
float U = (float)j / (output_width - 1); 
float V = (float)i /
(output_height - 1); 
float theta = U * 2 * M_PI;
float phi = V * M_PI;
cart2D cart = convertEquirectUVtoUnit2D(theta, phi, square_length);

cv::Vec3b val;
if (cart.faceIndex == X_POS) {
val = posX.at<cv::Vec3b>(cart.y, cart.x);
} else if (cart.faceIndex == X_NEG) {
val = negX.at<cv::Vec3b>(cart.y, cart.x);
} else if (cart.faceIndex == Y_POS) {
val = posY.at<cv::Vec3b>(cart.y, cart.x);
} else if (cart.faceIndex == Y_NEG) {
val = negY.at<cv::Vec3b>(cart.y, cart.x);
} else if (cart.faceIndex == Z_POS) {
val = posZ.at<cv::Vec3b>(cart.y, cart.x);
} else if (cart.faceIndex == Z_NEG) {
val = negZ.at<cv::Vec3b>(cart.y, cart.x);
}

destination.at<cv::Vec3b>(i, j) = val;
}
}
auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> diff = end - begin;

cv::imshow("Processed Image", destination);

std::string output_file(argv[2]);
cv::imwrite(output_file, destination);

cout << "Processing time: " << diff.count() << " s" << endl;

cv::waitKey();
return 0;
}
