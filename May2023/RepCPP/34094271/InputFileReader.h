
#pragma once
#include <iostream>
#include <fstream>
#include <vector>

class InputFileReader
{
private:
std::fstream fs;
std::vector<std::string> & split(const std::string &s, char delim, std::vector<std::string> &elems);

public:
bool readNextLine(std::vector<std::string> & str);
void Open(std::string fileName);
InputFileReader(std::string fileName);
InputFileReader(void);
~InputFileReader();
};
