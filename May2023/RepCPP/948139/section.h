#pragma once

#include "./colors.h"
#include <2DCoordinates.hpp>
#include <nbt/nbt.hpp>

struct Section {
using block_array = std::array<uint16_t, 4096>;

using color_array = std::vector<const Colors::Block *>;
using light_array = std::array<uint8_t, 2048>;

int8_t Y;
Coordinates parent_chunk_coordinates;

block_array blocks;
color_array colors;
light_array lights;
std::vector<nbt::NBT> palette;

block_array::value_type beaconIndex;

Section();
Section(const nbt::NBT &, const int, const Coordinates = {0, 0});
Section(Section &&other) { *this = std::move(other); }

Section &operator=(Section &&other) {
Y = other.Y;
parent_chunk_coordinates = other.parent_chunk_coordinates;

beaconIndex = other.beaconIndex;

blocks = std::move(other.blocks);
lights = std::move(other.lights);
colors = std::move(other.colors);
palette = std::move(other.palette);
return *this;
}

inline bool empty() const {
return palette.empty() ||
(palette.size() == 1 &&
palette[0]["Name"].get<std::string>() == "minecraft:air");
}

inline block_array::value_type block_at(uint8_t x, uint8_t y,
uint8_t z) const {
return blocks[x + 16 * z + 16 * 16 * y];
}

inline color_array::value_type color_at(uint8_t x, uint8_t y,
uint8_t z) const {
return colors[blocks[x + 16 * z + 16 * 16 * y]];
}

inline light_array::value_type light_at(uint8_t x, uint8_t y,
uint8_t z) const {
const uint16_t index = x + 16 * z + 16 * 16 * y;

return (index % 2 ? lights[index / 2] >> 4 : lights[index / 2] & 0x0f);
}

inline const nbt::NBT &state_at(uint8_t x, uint8_t y, uint8_t z) const {
return palette[blocks[x + 16 * z + 16 * 16 * y]];
}

void loadPalette(const Colors::Palette &);
};
