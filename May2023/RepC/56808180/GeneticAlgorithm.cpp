#include "GeneticAlgorithm.h"
#include <omp.h>
#include <functional>
#define MAX_NUMBER_OF_RECTANGLES 500
#define ALLOC_ALIGN 64
#define ALLOC_TRANSFER_ALIGN 4096
#pragma offload_attribute(push, target(mic))
unsigned long GeneticAlgorithm::validsqrt = sqrtl(255 * 255 * 3);
#pragma offload_attribute(pop)
void GeneticAlgorithm::mutateElite() {
int i, nthreads;
int height = this->height, width = this->width;
auto imagesRectanglesPhi = this->imagesRectangles;
auto comparisonResultsPhi = this->comparisonResults;
auto outputImagesPhi = this->outputImages;
auto nativeImagePhi = this->nativeImage;
auto nativeImageAreaPhi = this->nativeImage->Area;
auto sizeOfRectangleTablePhi = this->sizeOfRectangleTable;
auto populationPhi = this->population;
auto elitePhi = this->elite;
#pragma offload mandatory target(mic) in(comparisonResultsPhi[0:populationPhi]:alloc(comparisonResultsPhi[0:0]) alloc_if(0) free_if(0)) in(imagesRectanglesPhi[0:0]:alloc(imagesRectanglesPhi[0:0]) alloc_if(0) free_if(0))
#pragma omp parallel for private(i)
for (i = populationPhi - 1; i >= elitePhi; --i) {
unsigned long i1 = LightRand::Rand() % elitePhi;
int i2 = i % elitePhi;
mutation(imagesRectanglesPhi + comparisonResultsPhi[i].index * MAX_NUMBER_OF_RECTANGLES,
imagesRectanglesPhi + comparisonResultsPhi[i2].index * MAX_NUMBER_OF_RECTANGLES,
imagesRectanglesPhi + comparisonResultsPhi[i1].index * MAX_NUMBER_OF_RECTANGLES, height,
width);
}
}
void GeneticAlgorithm::generateRectangles() {
for (int j = this->population - 1; j >= 0; --j) {
generateRectanglesForImages(this->imagesRectangles + j * MAX_NUMBER_OF_RECTANGLES);
}
}
void GeneticAlgorithm::generateRectanglesForImages(Rectangle *rectangle) {
for (int i = 0; i < MAX_NUMBER_OF_RECTANGLES; ++i) {
rectangle[i] = getNewRectangle(this->height, this->width);
}
}
#pragma offload_attribute(push, target(mic))
void GeneticAlgorithm::mutation(Rectangle *returnArrayOfRectangles, Rectangle *first, Rectangle *second, int height, int width) {
for (int i = 0; i < MAX_NUMBER_OF_RECTANGLES; ++i) {
if (LightRand::Rand() % 2 > 0) {
returnArrayOfRectangles[i] = LightRand::Rand() % 4 > 0 ? first[i] : getNewRectangle(height, width);
}
else {
returnArrayOfRectangles[i] = LightRand::Rand() % 4 > 0 ? second[i] : getNewRectangle(height, width);
}
}
}
#pragma offload_attribute(pop)
#pragma offload_attribute(push, target(mic))
Rectangle GeneticAlgorithm::getNewRectangle(int height, int width) {
Rectangle rectangle;
rectangle.rightDown.y = LightRand::Rand() % (height - 1) + 1;
rectangle.leftUp.y = LightRand::Rand() % rectangle.rightDown.y;
rectangle.rightDown.x = LightRand::Rand() % (width - 1) + 1;
rectangle.leftUp.x = LightRand::Rand() % rectangle.rightDown.x;
rectangle.color.r = LightRand::Rand() % 256;
rectangle.color.g = LightRand::Rand() % 256;
rectangle.color.b = LightRand::Rand() % 256;
return rectangle;
}
#pragma offload_attribute(pop)
GeneticAlgorithm::GeneticAlgorithm(int population, int generation, int elite, const char *fileName, int numberOfThreads) :
mainTimer(Timer(Timer::Mode::Single)), generationTimer(Timer(Timer::Mode::Median)), scoreTimer(Timer(Timer::Mode::Median)), mutationTimer(Timer(Timer::Mode::Median)), numberOfThreads(numberOfThreads){
mainTimer.Start();
this->population = population;
this->generation = generation;
this->GenerationsLeft = 0;
this->elite = elite;
this->sizeOfRectangleTable = this->population * MAX_NUMBER_OF_RECTANGLES;
png::image<png::rgb_pixel> loadImage(fileName);
nativeImage = TransformPngToNativeImage(loadImage);
inputImage = loadImage;
this->height = inputImage.get_height();
this->width = inputImage.get_width();
this->OutputImage = new Image();
Image::InitImage(this->OutputImage, height, width);
this->imagesRectangles = (Rectangle *) _mm_malloc(this->sizeOfRectangleTable * sizeof(Rectangle), ALLOC_ALIGN);
this->comparisonResults = (Comparison *) _mm_malloc(this->population * sizeof(Comparison), ALLOC_ALIGN);
auto imagesRectanglesPhi = this->imagesRectangles;
auto comparisionResultsPhi = this->comparisonResults;
for (int j = 0; j < population; ++j) {
this->comparisonResults[j].index = j;
}
this->outputImages = (Image *) _mm_malloc(population * sizeof(Image), ALLOC_ALIGN);
auto outputImagesPhi = this->outputImages;
auto nativeImagePhi = this->nativeImage;
auto nativeImageAreaPhi = this->nativeImage->Area;
int sizeOfRectangleTablePhi = this->sizeOfRectangleTable;
int numberOfThreadsOnPhi;
#pragma offload target(mic) mandatory
#pragma omp parallel
{
#pragma omp single
numberOfThreadsOnPhi = omp_get_num_threads();
}
#pragma offload mandatory target(mic) in(outputImagesPhi[0:0]:alloc(outputImagesPhi[0:numberOfThreadsOnPhi]) alloc_if(1) free_if(0) align(64)) in(comparisionResultsPhi[0:0]:alloc(comparisionResultsPhi[0:population]) alloc_if(1) free_if(0) align(64)) in(imagesRectanglesPhi[0:sizeOfRectangleTablePhi]:alloc(imagesRectanglesPhi[0:sizeOfRectangleTablePhi]) alloc_if(1) free_if(0) align(64)) in(nativeImagePhi[0:1]:alloc(nativeImagePhi[0:1]) alloc_if(1) free_if(0) align(64)) in(nativeImageAreaPhi[0:width*height]:alloc(nativeImageAreaPhi[0:width*height]) alloc_if(1) free_if(0) align(64))
{
nativeImagePhi->Area = nativeImageAreaPhi;
for (int i = 0; i < numberOfThreadsOnPhi; ++i) {
Image::InitImage(outputImagesPhi + i, height, width);
}
}
for (int i = 0; i < population; ++i) {
Image::InitImage(this->outputImages + i, height, width);
}
}
GeneticAlgorithm::~GeneticAlgorithm() {
auto imagesRectanglesPhi = this->imagesRectangles;
auto comparisionResultsPhi = this->comparisonResults;
auto outputImagesPhi = this->outputImages;
auto nativeImagePhi = this->nativeImage;
auto nativeImageAreaPhi = this->nativeImage->Area;
int sizeOfRectangleTablePhi = this->sizeOfRectangleTable;
#pragma offload mandatory target(mic) in(outputImagesPhi[0:0]:alloc(outputImagesPhi[0:0]) alloc_if(0) free_if(1)) in(comparisionResultsPhi[0:0]:alloc(comparisionResultsPhi[0:0]) alloc_if(0) free_if(1)) in(imagesRectanglesPhi[0:0]:alloc(imagesRectanglesPhi[0:0]) alloc_if(0) free_if(1)) in(nativeImagePhi[0:0]:alloc(nativeImagePhi[0:0]) alloc_if(0) free_if(1)) in(nativeImageAreaPhi[0:0]:alloc(nativeImageAreaPhi[0:0]) alloc_if(0) free_if(1))
{
int numberOfThreadsOnPhi = omp_get_num_threads();
for (int i = 0; i < numberOfThreadsOnPhi; ++i) {
_mm_free(outputImagesPhi[i].Area);
}
}
_mm_free(this->outputImages);
if (this->imagesRectangles != nullptr) {
_mm_free(this->imagesRectangles);
}
if (this->comparisonResults != nullptr) {
_mm_free(this->comparisonResults);
}
if (this->nativeImage != nullptr) {
_mm_free(this->nativeImage->Area);
_mm_free(this->nativeImage);
}
if(this->OutputImage != nullptr) {
_mm_free(OutputImage->Area);
delete OutputImage;
}
}
void GeneticAlgorithm::Calculate() {
generateRectangles();
for (GenerationsLeft = generation; GenerationsLeft > 0; --GenerationsLeft) {
generationTimer.Start();
scoreTimer.Start();
CalculateValuesInParallel(GenerationsLeft, generation);
scoreTimer.Stop();
SortComparisions();
auto theBest = this->comparisonResults;
#ifdef debug
std::ostringstream stringStream;
stringStream << "results" << GenerationsLeft << ".png";
std::string copyOfStr = stringStream.str();
drawRectanglesOnImage(OutputImage, imagesRectangles + theBest[0].index * MAX_NUMBER_OF_RECTANGLES, width, height);
ConvertToPng(OutputImage).write(copyOfStr.c_str());
#endif
mutationTimer.Start();
mutateElite();
mutationTimer.Stop();
generationTimer.Stop();
}
auto theBest = this->comparisonResults;
auto imagesRectanglesPhi = this->imagesRectangles;
#pragma offload_transfer mandatory target(mic) out(imagesRectanglesPhi[theBest->index*MAX_NUMBER_OF_RECTANGLES:MAX_NUMBER_OF_RECTANGLES]:alloc(imagesRectanglesPhi[0:0]) alloc_if(0) free_if(0))
mainTimer.Stop();
printf("OpenMP,%d,%d,%d,%lu,%lu,%lu,%lu,", numberOfThreads, width, population , mainTimer.Get(), generationTimer.Get(), scoreTimer.Get(), mutationTimer.Get());
}
void GeneticAlgorithm::CalculateValuesInParallel(int generationsLeft, int generation1) {
int i, nthreads, th_id;
int height = this->height, width = this->width;
auto imagesRectanglesPhi = this->imagesRectangles;
auto comparisionResultsPhi = this->comparisonResults;
auto outputImagesPhi = this->outputImages;
auto nativeImagePhi = this->nativeImage;
auto nativeImageAreaPhi = this->nativeImage->Area;
auto sizeOfRectangleTablePhi = this->sizeOfRectangleTable;
auto populationPhi = this->population;
auto elitePhi = this->elite;
if (generationsLeft == generation1) {
#pragma offload mandatory target(mic) in(outputImagesPhi[0:0]:alloc(outputImagesPhi[0:0]) alloc_if(0) free_if(0)) inout(comparisionResultsPhi[0:populationPhi]:alloc(comparisionResultsPhi[0:0]) alloc_if(0) free_if(0)) in(imagesRectanglesPhi[0:sizeOfRectangleTablePhi]:alloc(imagesRectanglesPhi[0:0]) alloc_if(0) free_if(0)) in(nativeImagePhi[0:0]:alloc(nativeImagePhi[0:0]) alloc_if(0) free_if(0)) in(nativeImageAreaPhi[0:0]:alloc(nativeImageAreaPhi[0:0]) alloc_if(0) free_if(0))
#pragma omp parallel private(i, th_id)
{
th_id = omp_get_thread_num();
#pragma omp for
for (i = populationPhi - 1; i >= 0; --i) {
comparisionResultsPhi[i] = Comparison(i, CalculateValueOfImage(
imagesRectanglesPhi + i * MAX_NUMBER_OF_RECTANGLES, width, height,
outputImagesPhi+th_id, nativeImagePhi));
}
}
}
else {
#pragma offload mandatory target(mic) in(outputImagesPhi[0:0]:alloc(outputImagesPhi[0:0]) alloc_if(0) free_if(0)) out(comparisionResultsPhi[0:populationPhi]:alloc(comparisionResultsPhi[0:0]) alloc_if(0) free_if(0)) in(imagesRectanglesPhi[0:0]:alloc(imagesRectanglesPhi[0:0]) alloc_if(0) free_if(0)) in(nativeImagePhi[0:0]:alloc(nativeImagePhi[0:0]) alloc_if(0) free_if(0)) in(nativeImageAreaPhi[0:0]:alloc(nativeImageAreaPhi[0:0]) alloc_if(0) free_if(0))
#pragma omp parallel private(i, th_id)
{
th_id = omp_get_thread_num();
#pragma omp for
for (i = populationPhi - 1; i >= elitePhi; --i) {
comparisionResultsPhi[i] = Comparison(comparisionResultsPhi[i].index, CalculateValueOfImage(
imagesRectanglesPhi + comparisionResultsPhi[i].index * MAX_NUMBER_OF_RECTANGLES, width, height,
outputImagesPhi+th_id, nativeImagePhi));
}
}
}
}
#pragma offload_attribute(push, target(mic))
double GeneticAlgorithm::CalculateValueOfImage(Rectangle *imageRectangles, int width, int height, Image *outputImage,
Image *originalImage) {
ClearImage(outputImage, height, width);
drawRectanglesOnImage(outputImage, imageRectangles, width, height);
return compare(originalImage, outputImage, width, height);
}
#pragma offload_attribute(pop)
#pragma offload_attribute(push, target(mic))
void GeneticAlgorithm::ClearImage(Image *image, int height, int width) {
for (int i = width * height - 1; i >= 0; --i) {
image->Area[i].r = 0;
image->Area[i].g = 0;
image->Area[i].b = 0;
image->Area[i].drawed = 0;
}
}
#pragma offload_attribute(pop)
#pragma offload_attribute(push, target(mic))
void GeneticAlgorithm::drawRectanglesOnImage(Image* image, Rectangle *rectangles, int width, int height) {
for (int i = MAX_NUMBER_OF_RECTANGLES; i >= 0; --i) {
Rectangle rectangle = rectangles[i];
auto rHeight = abs(rectangle.rightDown.y - rectangle.leftUp.y);
auto rWidth = abs(rectangle.rightDown.x - rectangle.leftUp.x);
for (int y = rHeight - 1; y >= 0; --y) {
int baseY = (rectangle.leftUp.y + y) * width;
for (int x = rWidth - 1; x >= 0; --x) {
int linearIndex = baseY + rectangle.leftUp.x + x;
image->Area[linearIndex].r =
(image->Area[linearIndex].r * image->Area[linearIndex].drawed + rectangle.color.r) /
(image->Area[linearIndex].drawed + 1);
image->Area[linearIndex].g =
(image->Area[linearIndex].g * image->Area[linearIndex].drawed + rectangle.color.g) /
(image->Area[linearIndex].drawed + 1);
image->Area[linearIndex].b =
(image->Area[linearIndex].b * image->Area[linearIndex].drawed + rectangle.color.b) /
(image->Area[linearIndex].drawed + 1);
++image->Area[linearIndex].drawed;
}
}
}
}
#pragma offload_attribute(pop)
#pragma offload_attribute(push, target(mic))
double GeneticAlgorithm::compare(Image *first, Image *second, int width, int height) {
double maxDiff = width * height * validsqrt;
auto diff = 0.0;
for (int y = height - 1; y >= 0; --y) {
for (int x = width - 1; x >= 0; --x) {
auto firstPixel = first->Area[y * width + x];
auto secondPixel = second->Area[y * width + x];
diff += sqrtl(
(firstPixel.r - secondPixel.r) * (firstPixel.r - secondPixel.r)
+ (firstPixel.g - secondPixel.g) * (firstPixel.g - secondPixel.g)
+ (firstPixel.b - secondPixel.b) * (firstPixel.b - secondPixel.b));
}
}
return (maxDiff - diff) / maxDiff;
}
#pragma offload_attribute(pop)
void GeneticAlgorithm::SortComparisions() const {
std::sort(comparisonResults, comparisonResults + population, Comparison::CompareComparison);
}
Image *GeneticAlgorithm::TransformPngToNativeImage(png::image<png::rgb_pixel> image) {
const size_t imgHeight = image.get_height();
const size_t imgWidth = image.get_width();
Image *outArray = Image::CreateImage(imgHeight, imgWidth);
for (int y = 0; y < imgHeight; ++y) {
for (int x = 0; x < imgWidth; ++x) {
outArray->Area[y * imgWidth + x] = Pixel(image[y][x].red, image[y][x].green, image[y][x].blue);
}
}
return outArray;
}
png::image<png::rgb_pixel> GeneticAlgorithm::ConvertToPng(Image &image) {
png::image<png::rgb_pixel> resultImage = png::image<png::rgb_pixel>(width, height);
for (int y = 0; y < height; ++y) {
for (int x = 0; x < width; ++x) {
resultImage.set_pixel(x, y, png::rgb_pixel(image.Area[y * width + x].r, image.Area[y * width + x].g,
image.Area[y * width + x].b));
}
}
return resultImage;
}