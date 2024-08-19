#pragma once
#include "Color.hpp"
#include <iostream>


namespace CommonResolutions {
inline constexpr Vec2 SD_240p  = Vec2(  426,  240);
inline constexpr Vec2 SD_360p  = Vec2(  640,  360);
inline constexpr Vec2 SD_480p  = Vec2(  854,  480);
inline constexpr Vec2 HD_720p  = Vec2( 1280,  720);
inline constexpr Vec2 HD_1080p = Vec2( 1920, 1080);
inline constexpr Vec2 HD_2K    = Vec2( 2560, 1440);
inline constexpr Vec2 HD_4K    = Vec2( 3840, 2160);
inline constexpr Vec2 HD_5K    = Vec2( 5120, 2880);
inline constexpr Vec2 HD_8K    = Vec2( 7680, 4320);
inline constexpr Vec2 HD_10K   = Vec2(10328, 7760);
inline constexpr Vec2 HD_12K   = Vec2(12288, 6480);
}

class FrameBuffer {
public:
FrameBuffer()                         = delete;
FrameBuffer(const FrameBuffer&)       = delete;
FrameBuffer& operator=(FrameBuffer&)  = delete;
FrameBuffer& operator=(FrameBuffer&&) = default;
FrameBuffer(FrameBuffer&&)            = default;

explicit FrameBuffer(const Vec2& dimensions);
FrameBuffer(size_t width, size_t height);

size_t width()      const;
size_t height()     const;
size_t numPixels()  const;
float aspectRatio() const;
float megaPixels()  const;

Color getPixel(size_t i) const noexcept;
Color getPixel(size_t row, size_t col) const noexcept;
std::pair<size_t, size_t> getPixelRowCol(size_t i) const noexcept;

void setPixel(size_t i, const Color& color) noexcept;
void setPixel(size_t row, size_t col, const Color& color) noexcept;

private:
size_t width_;
size_t height_;
size_t bufferSize_;
std::unique_ptr<Color[]> pixels_;
};

std::ostream& operator<<(std::ostream& os, const FrameBuffer& frameBuffer);
