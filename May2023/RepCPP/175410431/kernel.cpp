
#include "benchmark.h"
#include "datatypes.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <omp.h>

#define MAX_EPS 0.001

class points2image : public kernel {
private:
int deviceId = 0;
private:
int read_testcases = 0;
std::ifstream input_file, output_file;
bool error_so_far = false;
double max_delta = 0.0;
PointCloud2* pointcloud2 = nullptr;
Mat44* cameraExtrinsicMat = nullptr;
Mat33* cameraMat = nullptr;
Vec5* distCoeff = nullptr;
ImageSize* imageSize = nullptr;
PointsImage* results = nullptr;
public:

virtual void init();

virtual void run(int p = 1);

virtual bool check_output();

protected:

virtual int read_next_testcases(int count);

virtual void check_next_outputs(int count);

int read_number_testcases(std::ifstream& input_file);
};


void  parsePointCloud(std::ifstream& input_file, PointCloud2* pointcloud2) {
try {
input_file.read((char*)&(pointcloud2->height), sizeof(int32_t));
input_file.read((char*)&(pointcloud2->width), sizeof(int32_t));
input_file.read((char*)&(pointcloud2->point_step), sizeof(uint32_t));
pointcloud2->data = 
(float*) omp_target_alloc(pointcloud2->height * pointcloud2->width * pointcloud2->point_step * sizeof(float), 0);
input_file.read((char*)pointcloud2->data, pointcloud2->height * pointcloud2->width * pointcloud2->point_step);
}  catch (std::ifstream::failure) {
throw std::ios_base::failure("Error reading the next point cloud.");
}
}

void  parseCameraExtrinsicMat(std::ifstream& input_file, Mat44* cameraExtrinsicMat) {
try {
for (int h = 0; h < 4; h++)
for (int w = 0; w < 4; w++)
input_file.read((char*)&(cameraExtrinsicMat->data[h][w]),sizeof(double));
} catch (std::ifstream::failure) {
throw std::ios_base::failure("Error reading the next extrinsic matrix.");
}
}

void parseCameraMat(std::ifstream& input_file, Mat33* cameraMat ) {
try {
for (int h = 0; h < 3; h++)
for (int w = 0; w < 3; w++)
input_file.read((char*)&(cameraMat->data[h][w]), sizeof(double));
} catch (std::ifstream::failure) {
throw std::ios_base::failure("Error reading the next camera matrix.");
}
}

void  parseDistCoeff(std::ifstream& input_file, Vec5* distCoeff) {
try {
for (int w = 0; w < 5; w++)
input_file.read((char*)&(distCoeff->data[w]), sizeof(double));
} catch (std::ifstream::failure) {
throw std::ios_base::failure("Error reading the next set of distance coefficients.");
}
}

void  parseImageSize(std::ifstream& input_file, ImageSize* imageSize) {
try {
input_file.read((char*)&(imageSize->width), sizeof(int32_t));
input_file.read((char*)&(imageSize->height), sizeof(int32_t));
} catch (std::ifstream::failure) {
throw std::ios_base::failure("Error reading the next image size.");
}
}

void parsePointsImage(std::ifstream& output_file, PointsImage* goldenResult) {
try {
output_file.read((char*)&(goldenResult->image_width), sizeof(int32_t));
output_file.read((char*)&(goldenResult->image_height), sizeof(int32_t));
output_file.read((char*)&(goldenResult->max_y), sizeof(int32_t));
output_file.read((char*)&(goldenResult->min_y), sizeof(int32_t));
int pos = 0;
int elements = goldenResult->image_height * goldenResult->image_width;
goldenResult->intensity = new float[elements];
goldenResult->distance = new float[elements];
goldenResult->min_height = new float[elements];
goldenResult->max_height = new float[elements];
for (int h = 0; h < goldenResult->image_height; h++)
for (int w = 0; w < goldenResult->image_width; w++)
{
output_file.read((char*)&(goldenResult->intensity[pos]), sizeof(float));
output_file.read((char*)&(goldenResult->distance[pos]), sizeof(float));
output_file.read((char*)&(goldenResult->min_height[pos]), sizeof(float));
output_file.read((char*)&(goldenResult->max_height[pos]), sizeof(float));
pos++;
}
} catch (std::ios_base::failure) {
throw std::ios_base::failure("Error reading the next reference image.");
}
}

int points2image::read_next_testcases(int count)
{
if (pointcloud2)
for (int m = 0; m < count; ++m)
omp_target_free(pointcloud2[m].data, 0);
delete [] pointcloud2;
pointcloud2 = new PointCloud2[count];
delete [] cameraExtrinsicMat;
cameraExtrinsicMat = new Mat44[count];
delete [] cameraMat;
cameraMat = new Mat33[count];
delete [] distCoeff;
distCoeff = new Vec5[count];
delete [] imageSize;
imageSize = new ImageSize[count];
if (results)
for (int m = 0; m < count; ++m)
{
delete [] results[m].intensity;
delete [] results[m].distance;
delete [] results[m].min_height;
delete [] results[m].max_height;
}
delete [] results;
results = new PointsImage[count];

int i;
for (i = 0; (i < count) && (read_testcases < testcases); i++,read_testcases++)
{
try {
parsePointCloud(input_file, pointcloud2 + i);
parseCameraExtrinsicMat(input_file, cameraExtrinsicMat + i);
parseCameraMat(input_file, cameraMat + i);
parseDistCoeff(input_file, distCoeff + i);
parseImageSize(input_file, imageSize + i);
} catch (std::ios_base::failure& e) {
std::cerr << e.what() << std::endl;
exit(-3);
}
}
return i;
}
int points2image::read_number_testcases(std::ifstream& input_file)
{
int32_t number;
try {
input_file.read((char*)&(number), sizeof(int32_t));
} catch (std::ifstream::failure) {
throw std::ios_base::failure("Error reading the number of testcases.");
}

return number;
}

void points2image::init() {
std::cout << "init\n";

input_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
output_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
try {
input_file.open("../../../data/p2i_input.dat", std::ios::binary);
} catch (std::ifstream::failure) {
std::cerr << "Error opening the input data file" << std::endl;
exit(-2);
}
try {
output_file.open("../../../data/p2i_output.dat", std::ios::binary);
} catch (std::ifstream::failure) {
std::cerr << "Error opening the output data file" << std::endl;
exit(-2);
}
try {
testcases = read_number_testcases(input_file);
} catch (std::ios_base::failure& e) {
std::cerr << e.what() << std::endl;
exit(-3);
}
int deviceNo = omp_get_num_devices();
deviceId = std::max(0, deviceNo -1);
std::cout << "Selected device " << deviceId;
std::cout << " out of " << deviceNo << std::endl;

error_so_far = false;
max_delta = 0.0;
pointcloud2 = nullptr;
cameraExtrinsicMat = nullptr;
cameraMat = nullptr;
distCoeff = nullptr;
imageSize = nullptr;
results = nullptr;

std::cout << "done\n" << std::endl;
}



PointsImage pointcloud2_to_image(
const PointCloud2& pointcloud2,
const Mat44& cameraExtrinsicMat,
const Mat33& cameraMat, const Vec5& distCoeff,
const ImageSize& imageSize)
{
int w = imageSize.width;
int h = imageSize.height;
PointsImage msg;
msg.intensity = new float[w*h];
std::memset(msg.intensity, 0, sizeof(float)*w*h);
msg.distance = new float[w*h];
std::memset(msg.distance, 0, sizeof(float)*w*h);
msg.min_height = new float[w*h];
std::memset(msg.min_height, 0, sizeof(float)*w*h);
msg.max_height = new float[w*h];
std::memset(msg.max_height, 0, sizeof(float)*w*h);
msg.max_y = -1;
msg.min_y = h;
msg.image_height = imageSize.height;
msg.image_width = imageSize.width;
int32_t max_y = -1;
int32_t min_y = h;

float* cloud = (float *)pointcloud2.data;
Mat33 invR;
for (int row = 0; row < 3; row++)
for (int col = 0; col < 3; col++)
invR.data[row][col] = cameraExtrinsicMat.data[col][row];
Mat13 invT;
for (int row = 0; row < 3; row++) {
invT.data[row] = 0.0;
for (int col = 0; col < 3; col++)
invT.data[row] -= invR.data[row][col] * cameraExtrinsicMat.data[col][3];
}
int sizeMat = pointcloud2.width * pointcloud2.height;
int sizeMaxCp = pointcloud2.height * pointcloud2.width * pointcloud2.point_step;
double* distanceArr = (double*) omp_target_alloc(sizeMat * sizeof(double), 0);
Point2d* imagePointArr = (Point2d*) omp_target_alloc(sizeMat * sizeof(Point2d), 0);
int cloudHeight = pointcloud2.height;
int cloudWidth = pointcloud2.width;
int cloudStepSize = pointcloud2.point_step;

#pragma omp target \
is_device_ptr(distanceArr, imagePointArr, cloud) \
map(to:distCoeff,cameraMat,invT,invR,cloudHeight,cloudWidth,cloudStepSize)
{
#pragma omp teams distribute parallel for collapse(2)
for (uint32_t x = 0; x < cloudWidth; ++x) {
for (uint32_t y = 0; y < cloudHeight; ++y) {
int iPoint =x + y * cloudWidth;
float* fp = (float *)(((uintptr_t)cloud) + (x + y*cloudWidth) * cloudStepSize);

double intensity = fp[4];

Mat13 point, point2;
point2.data[0] = double(fp[0]);
point2.data[1] = double(fp[1]);
point2.data[2] = double(fp[2]);
for (int row = 0; row < 3; row++) {
point.data[row] = invT.data[row];
for (int col = 0; col < 3; col++)
point.data[row] += point2.data[col] * invR.data[row][col];
}
distanceArr[iPoint] = point.data[2] * 100.0;
if (point.data[2] <= 2.5) {
Point2d imagepointError;
imagepointError.x = -1;
imagepointError.y = -1;
imagePointArr[iPoint] = imagepointError;
continue;
}
double tmpx = point.data[0] / point.data[2];
double tmpy = point.data[1] / point.data[2];
double r2 = tmpx * tmpx + tmpy * tmpy;
double tmpdist = 1 + distCoeff.data[0] * r2
+ distCoeff.data[1] * r2 * r2
+ distCoeff.data[4] * r2 * r2 * r2;
Point2d imagepoint;
imagepoint.x = tmpx * tmpdist
+ 2 * distCoeff.data[2] * tmpx * tmpy
+ distCoeff.data[3] * (r2 + 2 * tmpx * tmpx);
imagepoint.y = tmpy * tmpdist
+ distCoeff.data[2] * (r2 + 2 * tmpy * tmpy)
+ 2 * distCoeff.data[3] * tmpx * tmpy;
imagepoint.x = cameraMat.data[0][0] * imagepoint.x + cameraMat.data[0][2];
imagepoint.y = cameraMat.data[1][1] * imagepoint.y + cameraMat.data[1][2];
imagePointArr[iPoint] = imagepoint;
}
}
}
for (uint32_t x = 0; x < cloudWidth; ++x) {
for (uint32_t y = 0; y < cloudHeight; ++y) {
int iPoint =x + y * cloudWidth;
double distance = distanceArr[iPoint];
if (distance <= (2.5 * 100.0)) {
continue;
}
float* fp = (float *)(((uintptr_t)cloud) + (x + y*cloudWidth) * cloudStepSize);
double intensity = fp[4];
Point2d imagepoint = imagePointArr[iPoint];
int px = int(imagepoint.x + 0.5);
int py = int(imagepoint.y + 0.5);
if(0 <= px && px < w && 0 <= py && py < h)
{
int pid = py * w + px;
if(msg.distance[pid] == 0 || msg.distance[pid] > distance)
{
msg.distance[pid] = float(distance); 
msg.intensity[pid] = float(intensity);
msg.max_y = py > msg.max_y ? py : msg.max_y;
msg.min_y = py < msg.min_y ? py : msg.min_y;
}
msg.min_height[pid] = -1.25;
msg.max_height[pid] = 0;
}
}
}
omp_target_free(distanceArr, 0);
omp_target_free(imagePointArr, 0);
return msg;
}



void points2image::run(int p) {
pause_func();
while (read_testcases < testcases)
{
int count = read_next_testcases(p);
unpause_func();
for (int i = 0; i < count; i++)
{
results[i] = pointcloud2_to_image(pointcloud2[i],
cameraExtrinsicMat[i],
cameraMat[i], distCoeff[i],
imageSize[i]);
}
pause_func();
check_next_outputs(count);
}

}

void points2image::check_next_outputs(int count)
{
PointsImage reference;

for (int i = 0; i < count; i++)
{
parsePointsImage(output_file, &reference);
if ((results[i].image_height != reference.image_height)
|| (results[i].image_width != reference.image_width))
{
error_so_far = true;
}
if ((results[i].min_y != reference.min_y)
|| (results[i].max_y != reference.max_y))
{
error_so_far = true;
}

int pos = 0;
for (int h = 0; h < reference.image_height; h++)
for (int w = 0; w < reference.image_width; w++)
{
if (fabs(reference.intensity[pos] - results[i].intensity[pos]) > max_delta)
max_delta = fabs(reference.intensity[pos] - results[i].intensity[pos]);
if (fabs(reference.distance[pos] - results[i].distance[pos]) > max_delta)
max_delta = fabs(reference.distance[pos] - results[i].distance[pos]);
if (fabs(reference.min_height[pos] - results[i].min_height[pos]) > max_delta)
max_delta = fabs(reference.min_height[pos] - results[i].min_height[pos]);
if (fabs(reference.max_height[pos] - results[i].max_height[pos]) > max_delta)
max_delta = fabs(reference.max_height[pos] - results[i].max_height[pos]);
pos++;
}
delete [] reference.intensity;
delete [] reference.distance;
delete [] reference.min_height;
delete [] reference.max_height;
}
}

bool points2image::check_output() {
std::cout << "checking output \n";
input_file.close();
output_file.close();

std::cout << "max delta: " << max_delta << "\n";
if ((max_delta > MAX_EPS) || error_so_far)
return false;
return true;
}

points2image a = points2image();
kernel& myKernel = a;
