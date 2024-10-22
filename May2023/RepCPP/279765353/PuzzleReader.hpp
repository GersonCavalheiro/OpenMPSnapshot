
#pragma once
#ifndef PUZZLE_READER_HPP
#define PUZZLE_READER_HPP
#include <exception>
#include <iostream>
#include <nlohmann/json.hpp>
#include "Puzzle.hpp"

class PuzzleReader {
public:
Puzzle read(std::istream &is);
};

void from_json(const nlohmann::json &j, Puzzle &puzzle);
void from_json(const nlohmann::json &j, Shape &shape);
void from_json(const nlohmann::json &j, Coord &coord);
void from_json(const nlohmann::json &j, Polyomino &polyomino);

#endif
