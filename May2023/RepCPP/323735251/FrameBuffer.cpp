#pragma once
#include "FrameBuffer.hpp"
#include "Math.hpp"
#include "Color.hpp"
#include <iostream>
#include <assert.h>


FrameBuffer::FrameBuffer(const Vec2& dimensions)
: FrameBuffer(static_cast<size_t>(dimensions.x), static_cast<size_t>(dimensions.y)) {}

FrameBuffer::FrameBuffer(size_t width, size_t height)
: width_     (width),
height_    (height),
bufferSize_(width * height),
pixels_    (std::make_unique<Color[]>(bufferSize_)) {
if (width <= 0 || height <= 0 || bufferSize_ <= 0) {
throw std::invalid_argument("frame buffer must have dimensions greater than zero");
}
}

size_t FrameBuffer::width() const {
return width_;
}

size_t FrameBuffer::height() const {
return height_;
}

size_t FrameBuffer::numPixels() const {
return bufferSize_;
}

float FrameBuffer::aspectRatio() const {
return width_ / static_cast<float>(height_);
}

float FrameBuffer::megaPixels() const {
return Math::roundToNearestDigit(bufferSize_ / 1000000.00f, 2);
}



std::pair<size_t, size_t> FrameBuffer::getPixelRowCol(size_t i) const noexcept {
assert((i >= 0 && i < bufferSize_));
const size_t row = i / width_;
const size_t col = i - (row * width_);
return std::make_pair(row, col);
}

Color FrameBuffer::getPixel(size_t i) const noexcept {
assert((i >= 0 && i < bufferSize_));
return pixels_[i];
}

Color FrameBuffer::getPixel(size_t row, size_t col) const noexcept {
assert((row >= 0 && row < height_) && (col >= 0 && col < width_));
return pixels_[width_ * row + col];
}


void FrameBuffer::setPixel(size_t i, const Color& color) noexcept {
assert((i >= 0 && i < bufferSize_));
pixels_[i] = color;
}

void FrameBuffer::setPixel(size_t row, size_t col, const Color& color) noexcept {
assert((row >= 0 && row < height_) && (col >= 0 && col < width_));
pixels_[width_ * row + col] = color;
}



std::ostream& operator<<(std::ostream& os, const FrameBuffer& frameBuffer) {
os << "FrameBuffer("
<< "Size{"
<< "width:"       << frameBuffer.width()     << ","
<< "height:"      << frameBuffer.height()    << ","
<< "pixel-count:" << frameBuffer.numPixels() << "}, "
<< "MetaData{"
<< "mega-pixels:"  << frameBuffer.megaPixels()  << ","
<< "aspect-ratio:" << frameBuffer.aspectRatio() << "}"
<< ")";
return os;
}
