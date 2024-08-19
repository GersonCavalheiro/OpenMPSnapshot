
#pragma once

#include <string>

#ifdef _MSC_VER
typedef unsigned __int8  uint8_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#else
#include <stdint.h>
#endif



class SHA1 
{
public:
enum { BlockSize = 512 / 8, HashBytes = 20 };

SHA1();

std::string operator()(const void* data, size_t numBytes);
std::string operator()(const std::string& text);

void add(const void* data, size_t numBytes);

std::string getHash();
void        getHash(unsigned char buffer[HashBytes]);

void reset();

private:
void processBlock(const void* data);
void processBuffer();

uint64_t m_numBytes;
size_t   m_bufferSize;
uint8_t  m_buffer[BlockSize];

enum { HashValues = HashBytes / 4 };
uint32_t m_hash[HashValues];
};
