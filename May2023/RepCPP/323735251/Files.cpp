#pragma once
#include "Files.hpp"
#include "FrameBuffer.hpp"
#include <fstream>
#include <exception>
#include <filesystem>
#include <assert.h>


namespace detail {

std::ifstream openPpmFileForReading(const std::string& filepath) {
if (std::filesystem::path(filepath).extension() != ".ppm") {
throw std::runtime_error("Cannot read file \'" + filepath + "\' - does not end with .ppm");
}
if (!std::filesystem::exists(filepath)) {
throw std::runtime_error("Cannot read file \'" + filepath + "\' - does not exist");
}

std::ifstream ifs(filepath, std::ios::in | std::ios::binary);
if (!ifs) {
throw std::runtime_error("Cannot read file \'" + filepath + "\' - error while opening for write");
}
return ifs;
}

std::ofstream openPpmFileForWriting(const std::string& filepath) {
if (std::filesystem::path(filepath).extension() != ".ppm") {
throw std::runtime_error("Cannot write file \'" + filepath + "\' - does not end with .ppm");
}

std::ofstream ofs(filepath, std::ios::out | std::ios::binary);
if (!ofs) {
throw std::runtime_error("Cannot write file \'" + filepath + "\' - error while opening for write");
}
return ofs;
}
}


namespace Files {

bool pathExists(const std::string& filepath) {
return std::filesystem::exists(filepath);
}

std::string fileName(const std::string& filepath) {
return std::filesystem::path(filepath).filename().stem().string();
}

std::string fileExtension(const std::string filepath) {
return std::filesystem::path(filepath).extension().string();
}

std::string resolveAbsolutePath(const std::string& filepath) {
return std::filesystem::canonical(filepath).string();
}

FrameBuffer readPpm(const std::string& filepath) {
std::ifstream ifs = detail::openPpmFileForReading(filepath);

std::string header;
int bufferWidth, bufferHeight, numBytes;
ifs >> header >> bufferWidth >> bufferHeight >> numBytes;
if (header != "P6" || bufferWidth <= 0 || bufferHeight <= 0 || numBytes != 255) {
throw std::runtime_error("File \'" + filepath + "\' has an invalid header - expected magic number (P6), " +
"non-zero image dimensions, and byte size of 255");
}
ifs.ignore(256, '\n');

FrameBuffer frameBuffer{ static_cast<size_t>(bufferWidth), static_cast<size_t>(bufferHeight) };
unsigned char colorValues[3];
for (size_t i = 0; i < frameBuffer.numPixels(); i++) {
ifs.read(reinterpret_cast<char*>(colorValues), 3);
frameBuffer.setPixel(i, Color(colorValues[0], colorValues[1], colorValues[2]));
}
return frameBuffer;
}

void writePpm(const std::string& filepath, const FrameBuffer& frameBuffer) {
std::ofstream ofs = detail::openPpmFileForWriting(filepath);
ofs << "P6\n" << frameBuffer.width() << " " << frameBuffer.height() << "\n255\n";

for (size_t i = 0; i < frameBuffer.numPixels(); i++) {
const Color color = frameBuffer.getPixel(i);
ofs << static_cast<unsigned char>(color.r * 255)
<< static_cast<unsigned char>(color.g * 255)
<< static_cast<unsigned char>(color.b * 255);
}
}

void writePpmWithGammaCorrection(const std::string& filepath, const FrameBuffer& frameBuffer,
float gammaCorrection) {
const float invGamma = 1.00f / gammaCorrection;
std::ofstream ofs = detail::openPpmFileForWriting(filepath);
ofs << "P6\n" << frameBuffer.width() << " " << frameBuffer.height() << "\n255\n";

for (size_t i = 0; i < frameBuffer.numPixels(); i++) {
const Color color = frameBuffer.getPixel(i);
ofs << static_cast<unsigned char>((Math::pow(color.r, invGamma) * 255) + 0.50f)
<< static_cast<unsigned char>((Math::pow(color.g, invGamma) * 255) + 0.50f)
<< static_cast<unsigned char>((Math::pow(color.b, invGamma) * 255) + 0.50f);
}
}
}
