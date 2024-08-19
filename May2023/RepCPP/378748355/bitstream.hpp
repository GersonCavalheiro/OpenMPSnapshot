
#pragma once

#include "gtaquat.hpp"
#include "types.hpp"
#include <cassert>
#include <cmath>

constexpr int bitsToBytes(int bits)
{
return (((bits) + 7) >> 3);
}

constexpr int bytesToBits(int bytes)
{
return ((bytes) << 3);
}

class NetworkBitStream {
constexpr static const size_t StackAllocationSize = 256;
constexpr static const float CompressedVecMagnitudeEpsilon = 0.00001f;

public:
constexpr static const int Version = 3;

NetworkBitStream();

NetworkBitStream(int initialBytesToAllocate);

NetworkBitStream(unsigned char* _data, unsigned int lengthInBytes, bool _copyData);

~NetworkBitStream();

void reset(void);

void resetReadPointer(void);

void resetWritePointer(void);

public:
inline void writeBIT(bool data);

template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>, T>>
inline void writeUINT8(T data)
{
Write(uint8_t(data));
}

template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>, T>>
inline void writeUINT16(T data)
{
Write(uint16_t(data));
}

template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>, T>>
inline void writeUINT32(T data)
{
Write(uint32_t(data));
}

template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>, T>>
inline void writeUINT64(T data)
{
Write(uint64_t(data));
}

template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>, T>>
inline void writeINT8(T data)
{
Write(int8_t(data));
}

template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>, T>>
inline void writeINT16(T data)
{
Write(int16_t(data));
}

template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>, T>>
inline void writeINT32(T data)
{
Write(int32_t(data));
}

template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>, T>>
inline void writeINT64(T data)
{
Write(int64_t(data));
}

inline void writeDOUBLE(double data)
{
Write(data);
}

inline void writeFLOAT(float data)
{
Write(data);
}

inline void writeVEC2(Vector2 data)
{
Write(data);
}

inline void writeVEC3(Vector3 data)
{
Write(data);
}

inline void writeVEC4(Vector4 data)
{
Write(data);
}

inline void writeDynStr8(StringView data)
{
Write(static_cast<uint8_t>(data.length()));
Write(data.data(), data.length());
}

inline void writeDynStr16(StringView data)
{
Write(static_cast<uint16_t>(data.length()));
Write(data.data(), data.length());
}

inline void writeDynStr32(StringView data)
{
Write(static_cast<uint32_t>(data.length()));
Write(data.data(), data.length());
}

template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>, T>>
inline void writeArray(Span<T> data)
{
Write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
}

template <typename T, size_t S, typename U = std::enable_if_t<std::is_integral_v<T>, T>>
inline void writeArray(StaticArray<T, S> data)
{
Write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
}

inline void writeCompressedPercentPair(Vector2 data)
{
const uint8_t ha = (data.x >= 100 ? 0x0F : (uint8_t)CEILDIV((int)data.x, 7)) << 4 | (data.y >= 100 ? 0x0F : (uint8_t)CEILDIV((int)data.y, 7));
Write(ha);
}

inline void writeCompressedVEC3(Vector3 data);

inline void writeGTAQuat(GTAQuat data)
{
WriteNormQuat(data.q.w, data.q.x, data.q.y, data.q.z);
}

void WriteCompressedStr(StringView data);

inline bool readBIT(bool& data);

template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>, T>>
inline bool readUINT8(T& data)
{
uint8_t tmp;
const bool res = Read(tmp);
data = tmp;
return res;
}

template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>, T>>
inline bool readUINT16(T& data)
{
uint16_t tmp;
const bool res = Read(tmp);
data = tmp;
return res;
}

template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>, T>>
inline bool readUINT32(T& data)
{
uint32_t tmp;
const bool res = Read(tmp);
data = tmp;
return res;
}

template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>, T>>
inline bool readUINT64(T& data)
{
uint64_t tmp;
const bool res = Read(tmp);
data = tmp;
return res;
}

template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>, T>>
inline bool readINT8(T& data)
{
int8_t tmp;
const bool res = Read(tmp);
data = tmp;
return res;
}

template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>, T>>
inline bool readINT16(T& data)
{
int16_t tmp;
const bool res = Read(tmp);
data = tmp;
return res;
}

template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>, T>>
inline bool readINT32(T& data)
{
int32_t tmp;
const bool res = Read(tmp);
data = tmp;
return res;
}

template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>, T>>
inline bool readINT64(T& data)
{
int64_t tmp;
const bool res = Read(tmp);
data = tmp;
return res;
}

[[nodiscard]] inline bool readDOUBLE(double& data)
{
const bool res = Read(data);
return res && std::isfinite(data);
}

[[nodiscard]] inline bool readFLOAT(float& data)
{
const bool res = Read(data);
return res && std::isfinite(data);
}

[[nodiscard]] inline bool readVEC2(Vector2& data)
{
const bool res = Read(data);
return res && std::isfinite(data.x) && std::isfinite(data.y);
}

[[nodiscard]] inline bool readVEC3(Vector3& data)
{
const bool res = Read(data);
return res && std::isfinite(data.x) && std::isfinite(data.y) && std::isfinite(data.z);
}

[[nodiscard]] inline bool readPosVEC3(Vector3& data)
{
const bool res = Read(data);
return res && std::isfinite(data.x) && std::isfinite(data.y) && std::isfinite(data.z) && data.x < 20000.0f && data.x > -20000.0f && data.y < 20000.0f && data.y > -20000.0f && data.z < 200000.0f && data.z > -1000.0f;
}

[[nodiscard]] inline bool readVelVEC3(Vector3& data)
{
const bool res = Read(data);
return res && std::isfinite(data.x) && std::isfinite(data.y) && std::isfinite(data.z) && glm::dot(data, data) <= 100.0f * 100.0f;
}

[[nodiscard]] inline bool readVEC4(Vector4& data)
{
const bool res = Read(data);
return res && std::isfinite(data.x) && std::isfinite(data.y) && std::isfinite(data.z) && std::isfinite(data.w);
}

template <size_t Size>
inline bool readDynStr8(HybridString<Size>& data)
{
return readDynamicStr<uint8_t>(data);
}

template <size_t Size>
inline bool readDynStr16(HybridString<Size>& data)
{
return readDynamicStr<uint16_t>(data);
}

template <size_t Size>
inline bool readDynStr32(HybridString<Size>& data)
{
return readDynamicStr<uint32_t>(data);
}

template <typename T, typename U = std::enable_if_t<std::is_integral_v<T>, T>>
inline bool readArray(Span<T> data)
{
if (data.size() * sizeof(T) > unsigned(bitsToBytes(GetNumberOfUnreadBits()))) {
return false;
}

return Read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(T));
}

inline bool readCompressedPercentPair(Vector2& data)
{
uint8_t
health,
armour;
if (!Read(health)) {
return false;
}
if (!Read(armour)) {
return false;
}
data = Vector2(health, armour);
return true;
}

[[nodiscard]] inline bool readGTAQuat(GTAQuat& data)
{
auto res = Read(data.q);
return res && std::isfinite(data.q.x) && std::isfinite(data.q.y) && std::isfinite(data.q.z) && std::isfinite(data.q.w);
}

template <typename LenType, size_t Size>
bool readDynamicStr(HybridString<Size>& data)
{
LenType len;
if (!Read(len)) {
return false;
}
if (len > unsigned(bitsToBytes(GetNumberOfUnreadBits()))) {
return false;
}
data.reserve(len);
return Read(data.data(), len);
}

void WriteBits(const unsigned char* input, int numberOfBitsToWrite, const bool rightAlignedBits = true);

template <class templateType>
void WriteCompressed(templateType var);

bool ReadBits(unsigned char* output, int numberOfBitsToRead, const bool alignBitsToRight = true);

bool ReadBit(void);

template <class templateType>
bool ReadCompressed(templateType& var);

void AssertStreamEmpty(void);

void PrintBits(void) const;

void IgnoreBits(const int numberOfBits);

void SetWriteOffset(const int offset);

inline int GetNumberOfBitsUsed(void) const { return GetWriteOffset(); }
inline int GetWriteOffset(void) const { return numberOfBitsUsed; }

inline int GetNumberOfBytesUsed(void) const { return bitsToBytes(numberOfBitsUsed); }

inline int GetReadOffset(void) const { return readOffset; }

inline void SetReadOffset(int newReadOffset) { readOffset = newReadOffset; }

inline int GetNumberOfUnreadBits(void) const { return readOffset > numberOfBitsUsed ? 0 : numberOfBitsUsed - readOffset; }

int CopyData(unsigned char** _data) const;

void SetData(unsigned char* input);

inline unsigned char* GetData(void) const { return data; }

private:
void Write0(void);

void Write1(void);

template <class templateType>
bool Serialize(bool writeToBitstream, templateType& var);

template <class templateType>
bool SerializeDelta(bool writeToBitstream, templateType& currentValue, templateType lastValue);

template <class templateType>
bool SerializeDelta(bool writeToBitstream, templateType& currentValue);

template <class templateType>
bool SerializeCompressed(bool writeToBitstream, templateType& var);

template <class templateType>
bool SerializeCompressedDelta(bool writeToBitstream, templateType& currentValue, templateType lastValue);

template <class templateType>
bool SerializeCompressedDelta(bool writeToBitstream, templateType& currentValue);

bool Serialize(bool writeToBitstream, char* input, const int numberOfBytes);

template <class templateType> 
bool SerializeNormVector(bool writeToBitstream, templateType& x, templateType& y, templateType& z);

template <class templateType> 
bool SerializeVector(bool writeToBitstream, templateType& x, templateType& y, templateType& z);

template <class templateType> 
bool SerializeNormQuat(bool writeToBitstream, templateType& w, templateType& x, templateType& y, templateType& z);

template <class templateType> 
bool SerializeOrthMatrix(
bool writeToBitstream,
templateType& m00, templateType& m01, templateType& m02,
templateType& m10, templateType& m11, templateType& m12,
templateType& m20, templateType& m21, templateType& m22);

bool SerializeBits(bool writeToBitstream, unsigned char* input, int numberOfBitsToSerialize, const bool rightAlignedBits = true);

template <class templateType>
void Write(templateType var);

template <class templateType>
void WriteDelta(templateType currentValue, templateType lastValue);

template <class templateType>
void WriteDelta(templateType currentValue);

template <class templateType>
void WriteCompressedDelta(templateType currentValue, templateType lastValue);

template <class templateType>
void WriteCompressedDelta(templateType currentValue);

template <class templateType>
bool Read(templateType& var);

template <class templateType>
bool ReadDelta(templateType& var);

template <class templateType>
bool ReadCompressedDelta(templateType& var);

void Write(const char* input, const int numberOfBytes);

void Write(NetworkBitStream* bitStream, int numberOfBits);
void Write(NetworkBitStream* bitStream);

template <class templateType> 
void WriteNormVector(templateType x, templateType y, templateType z);

template <class templateType> 
void WriteVector(templateType x, templateType y, templateType z);

template <class templateType> 
void WriteNormQuat(templateType w, templateType x, templateType y, templateType z);

template <class templateType> 
void WriteOrthMatrix(
templateType m00, templateType m01, templateType m02,
templateType m10, templateType m11, templateType m12,
templateType m20, templateType m21, templateType m22);

bool Read(char* output, const int numberOfBytes);

template <class templateType> 
bool ReadNormVector(templateType& x, templateType& y, templateType& z);

template <class templateType> 
bool ReadVector(templateType& x, templateType& y, templateType& z);

template <class templateType> 
bool ReadNormQuat(templateType& w, templateType& x, templateType& y, templateType& z);

template <class templateType> 
bool ReadOrthMatrix(
templateType& m00, templateType& m01, templateType& m02,
templateType& m10, templateType& m11, templateType& m12,
templateType& m20, templateType& m21, templateType& m22);

void WriteAlignedBytes(const unsigned char* input, const int numberOfBytesToWrite);

bool ReadAlignedBytes(unsigned char* output, const int numberOfBytesToRead);

void AlignWriteToByteBoundary(void);

void AlignReadToByteBoundary(void);

void SetNumberOfBitsAllocated(const unsigned int lengthInBits);

void AddBitsAndReallocate(const int numberOfBitsToWrite);

private:
void WriteCompressed(const unsigned char* input, const int size, const bool unsignedData);

bool ReadCompressed(unsigned char* output, const int size, const bool unsignedData);

int numberOfBitsUsed;

int numberOfBitsAllocated;

int readOffset;

unsigned char* data;

bool copyData;

unsigned char stackData[StackAllocationSize];
};

template <>
inline void NetworkBitStream::Write(bool var)
{
if (var)
Write1();
else
Write0();
}

template <class templateType>
inline bool NetworkBitStream::Serialize(bool writeToBitstream, templateType& var)
{
if (writeToBitstream)
Write(var);
else
return Read(var);
return true;
}

template <class templateType>
inline bool NetworkBitStream::SerializeDelta(bool writeToBitstream, templateType& currentValue, templateType lastValue)
{
if (writeToBitstream)
WriteDelta(currentValue, lastValue);
else
return ReadDelta(currentValue);
return true;
}

template <class templateType>
inline bool NetworkBitStream::SerializeDelta(bool writeToBitstream, templateType& currentValue)
{
if (writeToBitstream)
WriteDelta(currentValue);
else
return ReadDelta(currentValue);
return true;
}

template <class templateType>
inline bool NetworkBitStream::SerializeCompressed(bool writeToBitstream, templateType& var)
{
if (writeToBitstream)
WriteCompressed(var);
else
return ReadCompressed(var);
return true;
}

template <class templateType>
inline bool NetworkBitStream::SerializeCompressedDelta(bool writeToBitstream, templateType& currentValue, templateType lastValue)
{
if (writeToBitstream)
WriteCompressedDelta(currentValue, lastValue);
else
return ReadCompressedDelta(currentValue);
return true;
}

template <class templateType>
inline bool NetworkBitStream::SerializeCompressedDelta(bool writeToBitstream, templateType& currentValue)
{
if (writeToBitstream)
WriteCompressedDelta(currentValue);
else
return ReadCompressedDelta(currentValue);
return true;
}

inline bool NetworkBitStream::Serialize(bool writeToBitstream, char* input, const int numberOfBytes)
{
if (writeToBitstream)
Write(input, numberOfBytes);
else
return Read(input, numberOfBytes);
return true;
}

template <class templateType>
inline bool NetworkBitStream::SerializeNormVector(bool writeToBitstream, templateType& x, templateType& y, templateType& z)
{
if (writeToBitstream)
WriteNormVector(x, y, z);
else
return ReadNormVector(x, y, z);
return true;
}

template <class templateType>
inline bool NetworkBitStream::SerializeVector(bool writeToBitstream, templateType& x, templateType& y, templateType& z)
{
if (writeToBitstream)
WriteVector(x, y, z);
else
return ReadVector(x, y, z);
return true;
}

template <class templateType>
inline bool NetworkBitStream::SerializeNormQuat(bool writeToBitstream, templateType& w, templateType& x, templateType& y, templateType& z)
{
if (writeToBitstream)
WriteNormQuat(w, x, y, z);
else
return ReadNormQuat(w, x, y, z);
return true;
}

template <class templateType>
inline bool NetworkBitStream::SerializeOrthMatrix(
bool writeToBitstream,
templateType& m00, templateType& m01, templateType& m02,
templateType& m10, templateType& m11, templateType& m12,
templateType& m20, templateType& m21, templateType& m22)
{
if (writeToBitstream)
WriteOrthMatrix(m00, m01, m02, m10, m11, m12, m20, m21, m22);
else
return ReadOrthMatrix(m00, m01, m02, m10, m11, m12, m20, m21, m22);
return true;
}

inline bool NetworkBitStream::SerializeBits(bool writeToBitstream, unsigned char* input, int numberOfBitsToSerialize, const bool rightAlignedBits)
{
if (writeToBitstream)
WriteBits(input, numberOfBitsToSerialize, rightAlignedBits);
else
return ReadBits(input, numberOfBitsToSerialize, rightAlignedBits);
return true;
}

template <class templateType>
inline void NetworkBitStream::Write(templateType var)
{
WriteBits((unsigned char*)&var, sizeof(templateType) * 8, true);
}

template <class templateType>
inline void NetworkBitStream::WriteDelta(templateType currentValue, templateType lastValue)
{
if (currentValue == lastValue) {
Write(false);
} else {
Write(true);
Write(currentValue);
}
}

template <>
inline void NetworkBitStream::WriteDelta(bool currentValue, bool lastValue)
{
Write(currentValue);
}

template <class templateType>
inline void NetworkBitStream::WriteDelta(templateType currentValue)
{
Write(true);
Write(currentValue);
}

template <class templateType>
inline void NetworkBitStream::WriteCompressed(templateType var)
{
WriteCompressed((unsigned char*)&var, sizeof(templateType) * 8, true);
}

template <>
inline void NetworkBitStream::WriteCompressed(bool var)
{
Write(var);
}

template <>
inline void NetworkBitStream::WriteCompressed(float var)
{
if (var < -1.0f)
var = -1.0f;
if (var > 1.0f)
var = 1.0f;
Write((unsigned short)((var + 1.0f) * 32767.5f));
}

template <>
inline void NetworkBitStream::WriteCompressed(double var)
{
assert(var > -1.01 && var < 1.01);
if (var < -1.0f)
var = -1.0f;
if (var > 1.0f)
var = 1.0f;

assert(sizeof(unsigned long) == 4);

Write((unsigned long)((var + 1.0) * 2147483648.0));
}

template <class templateType>
inline void NetworkBitStream::WriteCompressedDelta(templateType currentValue, templateType lastValue)
{
if (currentValue == lastValue) {
Write(false);
} else {
Write(true);
WriteCompressed(currentValue);
}
}

template <>
inline void NetworkBitStream::WriteCompressedDelta(bool currentValue, bool lastValue)
{
Write(currentValue);
}

template <class templateType>
inline void NetworkBitStream::WriteCompressedDelta(templateType currentValue)
{
Write(true);
WriteCompressed(currentValue);
}

template <>
inline void NetworkBitStream::WriteCompressedDelta(bool currentValue)
{
Write(currentValue);
}

template <class templateType>
inline bool NetworkBitStream::Read(templateType& var)
{
return ReadBits((unsigned char*)&var, sizeof(templateType) * 8, true);
}

template <>
inline bool NetworkBitStream::Read(bool& var)
{
if (GetNumberOfUnreadBits() == 0)
return false;

if (data[readOffset >> 3] & (0x80 >> (readOffset % 8))) 
var = true;
else
var = false;

++readOffset;

return true;
}

template <class templateType>
inline bool NetworkBitStream::ReadDelta(templateType& var)
{
bool dataWritten;
bool success;
success = Read(dataWritten);
if (dataWritten)
success = Read(var);
return success;
}

template <>
inline bool NetworkBitStream::ReadDelta(bool& var)
{
return Read(var);
}

template <class templateType>
inline bool NetworkBitStream::ReadCompressed(templateType& var)
{
return ReadCompressed((unsigned char*)&var, sizeof(templateType) * 8, true);
}

template <>
inline bool NetworkBitStream::ReadCompressed(bool& var)
{
return Read(var);
}

template <>
inline bool NetworkBitStream::ReadCompressed(float& var)
{
unsigned short compressedFloat;
if (Read(compressedFloat)) {
var = ((float)compressedFloat / 32767.5f - 1.0f);
return true;
}
return false;
}

template <>
inline bool NetworkBitStream::ReadCompressed(double& var)
{
unsigned long compressedFloat;
if (Read(compressedFloat)) {
var = ((double)compressedFloat / 2147483648.0 - 1.0);
return true;
}
return false;
}

template <class templateType>
inline bool NetworkBitStream::ReadCompressedDelta(templateType& var)
{
bool dataWritten;
bool success;
success = Read(dataWritten);
if (dataWritten)
success = ReadCompressed(var);
return success;
}

template <>
inline bool NetworkBitStream::ReadCompressedDelta(bool& var)
{
return Read(var);
}

template <class templateType> 
void NetworkBitStream::WriteNormVector(templateType x, templateType y, templateType z)
{

RakAssert(x <= 1.01 && y <= 1.01 && z <= 1.01 && x >= -1.01 && y >= -1.01 && z >= -1.01);

if (x > 1.0)
x = 1.0;
if (y > 1.0)
y = 1.0;
if (z > 1.0)
z = 1.0;
if (x < -1.0)
x = -1.0;
if (y < -1.0)
y = -1.0;
if (z < -1.0)
z = -1.0;

Write((bool)(x < 0.0));
if (y == 0.0)
Write(true);
else {
Write(false);
WriteCompressed((float)y);
}
if (z == 0.0)
Write(true);
else {
Write(false);
WriteCompressed((float)z);
}
}

template <class templateType> 
void NetworkBitStream::WriteVector(templateType x, templateType y, templateType z)
{
templateType magnitude = sqrt(x * x + y * y + z * z);
Write((float)magnitude);
if (magnitude > 0.0) {
WriteCompressed((float)(x / magnitude));
WriteCompressed((float)(y / magnitude));
WriteCompressed((float)(z / magnitude));
}
}

template <class templateType> 
void NetworkBitStream::WriteNormQuat(templateType w, templateType x, templateType y, templateType z)
{
Write((bool)(w < 0.0));
Write((bool)(x < 0.0));
Write((bool)(y < 0.0));
Write((bool)(z < 0.0));
Write((unsigned short)(fabs(x) * 65535.0));
Write((unsigned short)(fabs(y) * 65535.0));
Write((unsigned short)(fabs(z) * 65535.0));
}

template <class templateType> 
void NetworkBitStream::WriteOrthMatrix(
templateType m00, templateType m01, templateType m02,
templateType m10, templateType m11, templateType m12,
templateType m20, templateType m21, templateType m22)
{
double qw;
double qx;
double qy;
double qz;

float sum;
sum = 1 + m00 + m11 + m22;
if (sum < 0.0f)
sum = 0.0f;
qw = sqrt(sum) / 2;
sum = 1 + m00 - m11 - m22;
if (sum < 0.0f)
sum = 0.0f;
qx = sqrt(sum) / 2;
sum = 1 - m00 + m11 - m22;
if (sum < 0.0f)
sum = 0.0f;
qy = sqrt(sum) / 2;
sum = 1 - m00 - m11 + m22;
if (sum < 0.0f)
sum = 0.0f;
qz = sqrt(sum) / 2;
if (qw < 0.0)
qw = 0.0;
if (qx < 0.0)
qx = 0.0;
if (qy < 0.0)
qy = 0.0;
if (qz < 0.0)
qz = 0.0;
qx = _copysign(qx, m21 - m12);
qy = _copysign(qy, m02 - m20);
qz = _copysign(qz, m10 - m01);

WriteNormQuat(qw, qx, qy, qz);
}

template <class templateType> 
bool NetworkBitStream::ReadNormVector(templateType& x, templateType& y, templateType& z)
{
bool yZero, zZero;
bool xNeg;
float cy, cz;

Read(xNeg);

Read(yZero);
if (yZero)
y = 0.0;
else {
ReadCompressed(cy);
y = cy;
}

if (!Read(zZero))
return false;

if (zZero)
z = 0.0;
else {

if (!ReadCompressed(cz))
return false;
z = cz;
}

x = (templateType)(sqrtf((templateType)1.0 - y * y - z * z));
if (xNeg)
x = -x;
return true;
}

template <class templateType> 
bool NetworkBitStream::ReadVector(templateType& x, templateType& y, templateType& z)
{
float magnitude;
if (!Read(magnitude))
return false;
if (magnitude != 0.0) {
float cx, cy, cz;
ReadCompressed(cx);
ReadCompressed(cy);
if (!ReadCompressed(cz))
return false;
x = cx;
y = cy;
z = cz;
x *= magnitude;
y *= magnitude;
z *= magnitude;
} else {
x = 0.0;
y = 0.0;
z = 0.0;
}
return true;
}

template <class templateType> 
bool NetworkBitStream::ReadNormQuat(templateType& w, templateType& x, templateType& y, templateType& z)
{
bool cwNeg, cxNeg, cyNeg, czNeg;
unsigned short cx, cy, cz;
Read(cwNeg);
Read(cxNeg);
Read(cyNeg);
Read(czNeg);
Read(cx);
Read(cy);
if (!Read(cz))
return false;

x = (templateType)(cx / 65535.0);
y = (templateType)(cy / 65535.0);
z = (templateType)(cz / 65535.0);
if (cxNeg)
x = -x;
if (cyNeg)
y = -y;
if (czNeg)
z = -z;
float difference = 1.0f - x * x - y * y - z * z;
if (difference < 0.0f)
difference = 0.0f;
w = (templateType)(sqrt(difference));
if (cwNeg)
w = -w;
return true;
}

template <class templateType> 
bool NetworkBitStream::ReadOrthMatrix(
templateType& m00, templateType& m01, templateType& m02,
templateType& m10, templateType& m11, templateType& m12,
templateType& m20, templateType& m21, templateType& m22)
{
float qw, qx, qy, qz;
if (!ReadNormQuat(qw, qx, qy, qz))
return false;

double sqw = (double)qw * (double)qw;
double sqx = (double)qx * (double)qx;
double sqy = (double)qy * (double)qy;
double sqz = (double)qz * (double)qz;
m00 = (templateType)(sqx - sqy - sqz + sqw); 
m11 = (templateType)(-sqx + sqy - sqz + sqw);
m22 = (templateType)(-sqx - sqy + sqz + sqw);

double tmp1 = (double)qx * (double)qy;
double tmp2 = (double)qz * (double)qw;
m10 = (templateType)(2.0 * (tmp1 + tmp2));
m01 = (templateType)(2.0 * (tmp1 - tmp2));

tmp1 = (double)qx * (double)qz;
tmp2 = (double)qy * (double)qw;
m20 = (templateType)(2.0 * (tmp1 - tmp2));
m02 = (templateType)(2.0 * (tmp1 + tmp2));
tmp1 = (double)qy * (double)qz;
tmp2 = (double)qx * (double)qw;
m21 = (templateType)(2.0 * (tmp1 + tmp2));
m12 = (templateType)(2.0 * (tmp1 - tmp2));

return true;
}

inline void NetworkBitStream::writeCompressedVEC3(Vector3 data)
{
float magnitude = glm::length(data);
Write(magnitude);
if (magnitude > CompressedVecMagnitudeEpsilon) {
data /= magnitude;
WriteCompressed(data.x);
WriteCompressed(data.y);
WriteCompressed(data.z);
}
}

inline void NetworkBitStream::writeBIT(bool data)
{
Write<bool>(data);
}

inline bool NetworkBitStream::readBIT(bool& data)
{
return Read<bool>(data);
}
