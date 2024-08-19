#ifndef GENETICALGORITHM_GENETICALGORITHM_H
#define GENETICALGORITHM_GENETICALGORITHM_H
#include <vector>
#include <list>
#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <time.h>
#include <C++/png++-0.2.5/png.hpp>
#include <tgmath.h>
#include "Structures/Point.h"
#include "Structures/Comparison.h"
#include "Structures/Color.h"
#include "Structures/Rectangle.h"
#include "Structures/Pixel.h"
#include "Structures/LightRand.h"
#include "Structures/Image.h"
#include "../../Libraries/Timer.h"
class GeneticAlgorithm
{
private:
int height, width;
int population, generation, elite, numberOfThreads;
long sizeOfRectangleTable;
png::image<png::rgb_pixel> inputImage;
Image* outputImages = nullptr;
Rectangle* imagesRectangles = nullptr;
Comparison *comparisonResults = nullptr;
Image* nativeImage = nullptr;
Timer mainTimer, scoreTimer, mutationTimer, generationTimer;
static unsigned long validsqrt;
void generateRectanglesForImages(Rectangle *rectangle);
#pragma offload_attribute(push, target(mic))
Rectangle getNewRectangle(int height, int width);
static double compare(Image *first, Image *second, int width, int height);
static void drawRectanglesOnImage(Image* image, Rectangle *rectangles, int width, int height);
void mutation(Rectangle *source, Rectangle *first, Rectangle *second, int height, int width);
#pragma offload_attribute(pop)
void generateRectangles();
void mutateElite();
public:
int GenerationsLeft;
Image* OutputImage;
GeneticAlgorithm(int population, int generation, int elite, const char* fileName, int numberOfThreads);
Image * TransformPngToNativeImage(png::image<png::rgb_pixel> image);
void Calculate();
void CalculateValuesInParallel(int generationsLeft, int generation1);
png::image<png::rgb_pixel> ConvertToPng(Image& image);
#pragma offload_attribute(push, target(mic))
static void ClearImage(Image* image, int height, int width);
static double CalculateValueOfImage(Rectangle *vector, int width, int height, Image *outputImage,
Image *originalImage);
#pragma offload_attribute(pop)
void SortComparisions() const;
~GeneticAlgorithm();
};
#endif 
