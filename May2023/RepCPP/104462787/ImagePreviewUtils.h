#ifndef CAPTURE3_IMAGE_PREVIEW_UTILS_H
#define CAPTURE3_IMAGE_PREVIEW_UTILS_H


#include <cmath>
#include <omp.h>
#include <QtGui/QImage>
#include <QtGui/QPixmap>


#include "../engine/objects/image/ImageChannel.h"
#include "../engine/objects/image/ImageSize.h"


namespace Capture3
{

static QImage generatePreview(
const ImageChannel &imageChannel,
const unsigned int channel = 3,
const bool showShadowClipping = true,
const bool showHighlightClipping = true
)
{
const unsigned int imageArea = imageChannel.getSize().getArea();
const unsigned int imageWidth = imageChannel.getSize().getWidth();
const unsigned int imageHeight = imageChannel.getSize().getHeight();
const double *imageData = imageChannel.getData();

QImage image(imageWidth, imageHeight, QImage::Format_ARGB32_Premultiplied);

unsigned char *output = image.bits();

const unsigned int indexInputX = channel != 3 ? channel : 0;
const unsigned int indexInputY = channel != 3 ? channel : 1;
const unsigned int indexInputZ = channel != 3 ? channel : 2;

#pragma omp parallel for schedule(static)
for (unsigned int i = 0; i < imageArea; i++) {

const unsigned int indexInput = i * 3;
const unsigned int indexOutput = i * 4;

const double valueX = imageData[indexInput + indexInputX];
const double valueY = imageData[indexInput + indexInputY];
const double valueZ = imageData[indexInput + indexInputZ];

auto colorR = (int) lround(valueX * 255.0);
auto colorG = (int) lround(valueY * 255.0);
auto colorB = (int) lround(valueZ * 255.0);
colorR = colorR < 0 ? 0 : colorR > 255 ? 255 : colorR;
colorG = colorG < 0 ? 0 : colorG > 255 ? 255 : colorG;
colorB = colorB < 0 ? 0 : colorB > 255 ? 255 : colorB;

if (showShadowClipping) {
if (valueX < 0 || valueY < 0 || valueZ < 0) {
colorR = 0;
colorG = 0;
colorB = 255;
}
}
if (showHighlightClipping) {
if (valueX > 1 || valueY > 1 || valueZ > 1) {
colorR = 255;
colorG = 0;
colorB = 0;
}
}

output[indexOutput + 0] = (unsigned char) colorB;
output[indexOutput + 1] = (unsigned char) colorG;
output[indexOutput + 2] = (unsigned char) colorR;
output[indexOutput + 3] = 255;
}

return image;
}


static QImage generatePreviewImage(
const ImageChannel &imageChannel,
const bool showShadowClipping = true,
const bool showHighlightClipping = true
)
{
return generatePreview(imageChannel, 3, showShadowClipping, showHighlightClipping);
}


static QImage generatePreviewImageX(
const ImageChannel &imageChannel,
const bool showShadowClipping = true,
const bool showHighlightClipping = true
)
{
return generatePreview(imageChannel, 0, showShadowClipping, showHighlightClipping);
}


static QImage generatePreviewImageY(
const ImageChannel &imageChannel,
const bool showShadowClipping = true,
const bool showHighlightClipping = true
)
{
return generatePreview(imageChannel, 1, showShadowClipping, showHighlightClipping);
}


static QImage generatePreviewImageZ(
const ImageChannel &imageChannel,
const bool showShadowClipping = true,
const bool showHighlightClipping = true
)
{
return generatePreview(imageChannel, 2, showShadowClipping, showHighlightClipping);
}


}


#endif 
