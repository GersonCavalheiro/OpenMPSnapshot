#pragma once

#include <chrono>
#include <SFML/Graphics.hpp>

inline auto calculateImageSize(const sf::Image& image)
{
return image.getSize().x * image.getSize().y * 4;
}

template <typename Callable, typename... Args>
auto runWithTimeMeasurementCpu(Callable&& function, Args&&... params)
{
const auto start = std::chrono::high_resolution_clock::now();
std::forward<Callable>(function)(std::forward<Args>(params)...);
const auto stop = std::chrono::high_resolution_clock::now();
const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
return duration;
}