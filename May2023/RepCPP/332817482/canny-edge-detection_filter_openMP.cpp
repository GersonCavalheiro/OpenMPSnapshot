#include "../../utils/filter.h"
#include "../../utils/filter_factory.h"
#include <string>
#include <omp.h>



void CannyEdgeDetectionFilter::applyFilter(Image *image, Image *newImage) {

std::string filter = "black-white";
Filter *blackWhiteFilter = FilterFactory::filterCreate(filter);
blackWhiteFilter->applyFilter(image, newImage);
delete blackWhiteFilter;

filter = "gaussian-blur";
Filter *gaussianBlurFilter = FilterFactory::filterCreate(filter);
gaussianBlurFilter->applyFilter(newImage, image);
delete gaussianBlurFilter;


filter = "gradient";
GradientFilter *gradientFilter = static_cast<GradientFilter *>(FilterFactory::filterCreate(filter));
gradientFilter->applyFilter(image, newImage);

filter = "non-maximum-suppression";
Filter *nonMaximumSuppressionFilter = FilterFactory::filterCreate(filter, 0.0, gradientFilter->theta,
gradientFilter->thetaHeight, gradientFilter->thetaWidth);
nonMaximumSuppressionFilter->applyFilter(newImage, image);
delete nonMaximumSuppressionFilter;
delete gradientFilter;


filter = "double-threshold";
Filter *doubleTresholdFilter = FilterFactory::filterCreate(filter);
doubleTresholdFilter->applyFilter(image, newImage);
delete doubleTresholdFilter;

filter = "edge-tracking";
Filter *edgeTrackingFilter = FilterFactory::filterCreate(filter);;
edgeTrackingFilter->applyFilter(newImage, image);
delete edgeTrackingFilter;


#pragma omp parallel for
for (unsigned int i = 1; i < image->height - 1; ++i) {
Pixel *swp = image->matrix[i];
image->matrix[i] = newImage->matrix[i];
newImage->matrix[i] = swp;
for (unsigned int j = 1; j < image->width - 1; ++j) {
if (newImage->matrix[i][j].r < 100) {
newImage->matrix[i][j] = Pixel(0, 0, 0, newImage->matrix[i][j].a);
}
}
}
}
