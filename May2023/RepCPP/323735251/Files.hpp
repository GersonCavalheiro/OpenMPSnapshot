#pragma once
#include <fstream>
#include <exception>
#include <filesystem>
#include <assert.h>


class FrameBuffer;

namespace Files {

bool pathExists(const std::string& filepath);
std::string fileName(const std::string& filepath);
std::string fileExtension(const std::string filepath);
std::string resolveAbsolutePath(const std::string& filepath);

FrameBuffer readPpm(const std::string& filepath);

void writePpm(const std::string& filepath, const FrameBuffer& frameBuffer);
void writePpmWithGammaCorrection(const std::string& filepath, const FrameBuffer& frameBuffer, float gammaCorrection = 2.20f);
}
