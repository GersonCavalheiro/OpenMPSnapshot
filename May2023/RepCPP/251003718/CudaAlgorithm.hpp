#pragma once

#include <SFML/Graphics.hpp>
#include "FiltersProvider.hpp"

namespace Cuda
{
void applyFilter(sf::Image&, const Filter::Kernel&);
}