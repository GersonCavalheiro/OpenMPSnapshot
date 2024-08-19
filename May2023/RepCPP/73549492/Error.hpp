#pragma once

#include <string>





class Error {

public:

Error(std::string error, std::string file, unsigned int line);

std::string GetMessage() const;
const std::string &GetError() const;
const std::string &GetFile() const;
const unsigned int &GetLine() const;

private:

const std::string mError;
const std::string mFile;
const unsigned int mLine = 0;

};



#define THROW_ERROR(error) throw Error(error, __FILE__, __LINE__)
