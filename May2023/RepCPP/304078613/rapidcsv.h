

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#ifdef HAS_CODECVT
#include <codecvt>
#endif
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace rapidcsv
{
#if defined(_MSC_VER)
static const bool sPlatformHasCR = true;
#else
static const bool sPlatformHasCR = false;
#endif


struct ConverterParams
{

explicit ConverterParams(const bool pHasDefaultConverter = false,
const long double pDefaultFloat = std::numeric_limits<long double>::signaling_NaN(),
const long long pDefaultInteger = 0)
: mHasDefaultConverter(pHasDefaultConverter)
, mDefaultFloat(pDefaultFloat)
, mDefaultInteger(pDefaultInteger)
{
}


bool mHasDefaultConverter;


long double mDefaultFloat;


long long mDefaultInteger;
};


class no_converter : public std::exception
{

virtual const char* what() const throw()
{
return "unsupported conversion datatype";
}
};


template<typename T>
class Converter
{
public:

Converter(const ConverterParams& pConverterParams)
: mConverterParams(pConverterParams)
{
}


void ToStr(const T& pVal, std::string& pStr) const
{
if (typeid(T) == typeid(int) ||
typeid(T) == typeid(long) ||
typeid(T) == typeid(long long) ||
typeid(T) == typeid(unsigned) ||
typeid(T) == typeid(unsigned long) ||
typeid(T) == typeid(unsigned long long) ||
typeid(T) == typeid(float) ||
typeid(T) == typeid(double) ||
typeid(T) == typeid(long double) ||
typeid(T) == typeid(char))
{
std::ostringstream out;
out << pVal;
pStr = out.str();
}
else
{
throw no_converter();
}
}


void ToVal(const std::string& pStr, T& pVal) const
{
try
{
if (typeid(T) == typeid(int))
{
pVal = static_cast<T>(std::stoi(pStr));
return;
}
else if (typeid(T) == typeid(long))
{
pVal = static_cast<T>(std::stol(pStr));
return;
}
else if (typeid(T) == typeid(long long))
{
pVal = static_cast<T>(std::stoll(pStr));
return;
}
else if (typeid(T) == typeid(unsigned))
{
pVal = static_cast<T>(std::stoul(pStr));
return;
}
else if (typeid(T) == typeid(unsigned long))
{
pVal = static_cast<T>(std::stoul(pStr));
return;
}
else if (typeid(T) == typeid(unsigned long long))
{
pVal = static_cast<T>(std::stoull(pStr));
return;
}
}
catch (...)
{
if (!mConverterParams.mHasDefaultConverter)
{
throw;
}
else
{
pVal = static_cast<T>(mConverterParams.mDefaultInteger);
return;
}
}

try
{
if (typeid(T) == typeid(float))
{
pVal = static_cast<T>(std::stof(pStr));
return;
}
else if (typeid(T) == typeid(double))
{
pVal = static_cast<T>(std::stod(pStr));
return;
}
else if (typeid(T) == typeid(long double))
{
pVal = static_cast<T>(std::stold(pStr));
return;
}
}
catch (...)
{
if (!mConverterParams.mHasDefaultConverter)
{
throw;
}
else
{
pVal = static_cast<T>(mConverterParams.mDefaultFloat);
return;
}
}

if (typeid(T) == typeid(char))
{
pVal = static_cast<T>(pStr[0]);
return;
}
else
{
throw no_converter();
}
}

private:
const ConverterParams& mConverterParams;
};


template<>
inline void Converter<std::string>::ToStr(const std::string& pVal, std::string& pStr) const
{
pStr = pVal;
}


template<>
inline void Converter<std::string>::ToVal(const std::string& pStr, std::string& pVal) const
{
pVal = pStr;
}

template<typename T>
using ConvFunc = std::function<void (const std::string & pStr, T & pVal)>;


struct LabelParams
{

explicit LabelParams(const int pColumnNameIdx = 0, const int pRowNameIdx = -1)
: mColumnNameIdx(pColumnNameIdx)
, mRowNameIdx(pRowNameIdx)
{
}


int mColumnNameIdx;


int mRowNameIdx;
};


struct SeparatorParams
{

explicit SeparatorParams(const char pSeparator = ',', const bool pTrim = false,
const bool pHasCR = sPlatformHasCR, const bool pQuotedLinebreaks = false,
const bool pAutoQuote = true)
: mSeparator(pSeparator)
, mTrim(pTrim)
, mHasCR(pHasCR)
, mQuotedLinebreaks(pQuotedLinebreaks)
, mAutoQuote(pAutoQuote)
{
}


char mSeparator;


bool mTrim;


bool mHasCR;


bool mQuotedLinebreaks;


bool mAutoQuote;
};


class Document
{
public:

explicit Document(const std::string& pPath = std::string(),
const LabelParams& pLabelParams = LabelParams(),
const SeparatorParams& pSeparatorParams = SeparatorParams(),
const ConverterParams& pConverterParams = ConverterParams())
: mPath(pPath)
, mLabelParams(pLabelParams)
, mSeparatorParams(pSeparatorParams)
, mConverterParams(pConverterParams)
{
if (!mPath.empty())
{
ReadCsv();
}
}


explicit Document(std::istream& pStream,
const LabelParams& pLabelParams = LabelParams(),
const SeparatorParams& pSeparatorParams = SeparatorParams(),
const ConverterParams& pConverterParams = ConverterParams())
: mPath()
, mLabelParams(pLabelParams)
, mSeparatorParams(pSeparatorParams)
, mConverterParams(pConverterParams)
{
ReadCsv(pStream);
}


void Load(const std::string& pPath)
{
mPath = pPath;
ReadCsv();
}


void Load(std::istream& pStream)
{
mPath = "";
ReadCsv(pStream);
}


void Save(const std::string& pPath = std::string())
{
if (!pPath.empty())
{
mPath = pPath;
}
WriteCsv();
}


void Save(std::ostream& pStream)
{
WriteCsv(pStream);
}


ssize_t GetColumnIdx(const std::string& pColumnName) const
{
if (mLabelParams.mColumnNameIdx >= 0)
{
if (mColumnNames.find(pColumnName) != mColumnNames.end())
{
return mColumnNames.at(pColumnName) - (mLabelParams.mRowNameIdx + 1);
}
}
return -1;
}


template<typename T>
std::vector<T> GetColumn(const size_t pColumnIdx) const
{
const ssize_t columnIdx = pColumnIdx + (mLabelParams.mRowNameIdx + 1);
std::vector<T> column;
Converter<T> converter(mConverterParams);
for (auto itRow = mData.begin(); itRow != mData.end(); ++itRow)
{
if (std::distance(mData.begin(), itRow) > mLabelParams.mColumnNameIdx)
{
T val;
converter.ToVal(itRow->at(columnIdx), val);
column.push_back(val);
}
}
return column;
}


template<typename T>
std::vector<T> GetColumn(const size_t pColumnIdx, ConvFunc<T> pToVal) const
{
const ssize_t columnIdx = pColumnIdx + (mLabelParams.mRowNameIdx + 1);
std::vector<T> column;
for (auto itRow = mData.begin(); itRow != mData.end(); ++itRow)
{
if (std::distance(mData.begin(), itRow) > mLabelParams.mColumnNameIdx)
{
T val;
pToVal(itRow->at(columnIdx), val);
column.push_back(val);
}
}
return column;
}


template<typename T>
std::vector<T> GetColumn(const std::string& pColumnName) const
{
const ssize_t columnIdx = GetColumnIdx(pColumnName);
if (columnIdx < 0)
{
throw std::out_of_range("column not found: " + pColumnName);
}
return GetColumn<T>(columnIdx);
}


template<typename T>
std::vector<T> GetColumn(const std::string& pColumnName, ConvFunc<T> pToVal) const
{
const ssize_t columnIdx = GetColumnIdx(pColumnName);
if (columnIdx < 0)
{
throw std::out_of_range("column not found: " + pColumnName);
}
return GetColumn<T>(columnIdx, pToVal);
}


template<typename T>
void SetColumn(const size_t pColumnIdx, const std::vector<T>& pColumn)
{
const size_t columnIdx = pColumnIdx + (mLabelParams.mRowNameIdx + 1);

while (pColumn.size() + (mLabelParams.mColumnNameIdx + 1) > GetDataRowCount())
{
std::vector<std::string> row;
row.resize(GetDataColumnCount());
mData.push_back(row);
}

if ((columnIdx + 1) > GetDataColumnCount())
{
for (auto itRow = mData.begin(); itRow != mData.end(); ++itRow)
{
itRow->resize(columnIdx + 1 + (mLabelParams.mRowNameIdx + 1));
}
}

Converter<T> converter(mConverterParams);
for (auto itRow = pColumn.begin(); itRow != pColumn.end(); ++itRow)
{
std::string str;
converter.ToStr(*itRow, str);
mData.at(std::distance(pColumn.begin(), itRow) + (mLabelParams.mColumnNameIdx + 1)).at(columnIdx) = str;
}
}


template<typename T>
void SetColumn(const std::string& pColumnName, const std::vector<T>& pColumn)
{
const ssize_t columnIdx = GetColumnIdx(pColumnName);
if (columnIdx < 0)
{
throw std::out_of_range("column not found: " + pColumnName);
}
SetColumn<T>(columnIdx, pColumn);
}


void RemoveColumn(const size_t pColumnIdx)
{
const ssize_t columnIdx = pColumnIdx + (mLabelParams.mRowNameIdx + 1);
for (auto itRow = mData.begin(); itRow != mData.end(); ++itRow)
{
itRow->erase(itRow->begin() + columnIdx);
}
}


void RemoveColumn(const std::string& pColumnName)
{
ssize_t columnIdx = GetColumnIdx(pColumnName);
if (columnIdx < 0)
{
throw std::out_of_range("column not found: " + pColumnName);
}

RemoveColumn(columnIdx);
}


size_t GetColumnCount() const
{
const ssize_t count = static_cast<ssize_t>((mData.size() > 0) ? mData.at(0).size() : 0) -
(mLabelParams.mRowNameIdx + 1);
return (count >= 0) ? count : 0;
}


ssize_t GetRowIdx(const std::string& pRowName) const
{
if (mLabelParams.mRowNameIdx >= 0)
{
if (mRowNames.find(pRowName) != mRowNames.end())
{
return mRowNames.at(pRowName) - (mLabelParams.mColumnNameIdx + 1);
}
}
return -1;
}


template<typename T>
std::vector<T> GetRow(const size_t pRowIdx) const
{
const ssize_t rowIdx = pRowIdx + (mLabelParams.mColumnNameIdx + 1);
std::vector<T> row;
Converter<T> converter(mConverterParams);
for (auto itCol = mData.at(rowIdx).begin(); itCol != mData.at(rowIdx).end(); ++itCol)
{
if (std::distance(mData.at(rowIdx).begin(), itCol) > mLabelParams.mRowNameIdx)
{
T val;
converter.ToVal(*itCol, val);
row.push_back(val);
}
}
return row;
}


template<typename T>
std::vector<T> GetRow(const size_t pRowIdx, ConvFunc<T> pToVal) const
{
const ssize_t rowIdx = pRowIdx + (mLabelParams.mColumnNameIdx + 1);
std::vector<T> row;
Converter<T> converter(mConverterParams);
for (auto itCol = mData.at(rowIdx).begin(); itCol != mData.at(rowIdx).end(); ++itCol)
{
if (std::distance(mData.at(rowIdx).begin(), itCol) > mLabelParams.mRowNameIdx)
{
T val;
pToVal(*itCol, val);
row.push_back(val);
}
}
return row;
}


template<typename T>
std::vector<T> GetRow(const std::string& pRowName) const
{
ssize_t rowIdx = GetRowIdx(pRowName);
if (rowIdx < 0)
{
throw std::out_of_range("row not found: " + pRowName);
}
return GetRow<T>(rowIdx);
}


template<typename T>
std::vector<T> GetRow(const std::string& pRowName, ConvFunc<T> pToVal) const
{
ssize_t rowIdx = GetRowIdx(pRowName);
if (rowIdx < 0)
{
throw std::out_of_range("row not found: " + pRowName);
}
return GetRow<T>(rowIdx, pToVal);
}


template<typename T>
void SetRow(const size_t pRowIdx, const std::vector<T>& pRow)
{
const size_t rowIdx = pRowIdx + (mLabelParams.mColumnNameIdx + 1);

while ((rowIdx + 1) > GetDataRowCount())
{
std::vector<std::string> row;
row.resize(GetDataColumnCount());
mData.push_back(row);
}

if (pRow.size() > GetDataColumnCount())
{
for (auto itRow = mData.begin(); itRow != mData.end(); ++itRow)
{
itRow->resize(pRow.size() + (mLabelParams.mRowNameIdx + 1));
}
}

Converter<T> converter(mConverterParams);
for (auto itCol = pRow.begin(); itCol != pRow.end(); ++itCol)
{
std::string str;
converter.ToStr(*itCol, str);
mData.at(rowIdx).at(std::distance(pRow.begin(), itCol) + (mLabelParams.mRowNameIdx + 1)) = str;
}
}


template<typename T>
void SetRow(const std::string& pRowName, const std::vector<T>& pRow)
{
ssize_t rowIdx = GetRowIdx(pRowName);
if (rowIdx < 0)
{
throw std::out_of_range("row not found: " + pRowName);
}
return SetRow<T>(rowIdx, pRow);
}


void RemoveRow(const size_t pRowIdx)
{
const ssize_t rowIdx = pRowIdx + (mLabelParams.mColumnNameIdx + 1);
mData.erase(mData.begin() + rowIdx);
}


void RemoveRow(const std::string& pRowName)
{
ssize_t rowIdx = GetRowIdx(pRowName);
if (rowIdx < 0)
{
throw std::out_of_range("row not found: " + pRowName);
}

RemoveRow(rowIdx);
}


size_t GetRowCount() const
{
const ssize_t count = static_cast<ssize_t>(mData.size()) - (mLabelParams.mColumnNameIdx + 1);
return (count >= 0) ? count : 0;
}


template<typename T>
T GetCell(const size_t pColumnIdx, const size_t pRowIdx) const
{
const ssize_t columnIdx = pColumnIdx + (mLabelParams.mRowNameIdx + 1);
const ssize_t rowIdx = pRowIdx + (mLabelParams.mColumnNameIdx + 1);

T val;
Converter<T> converter(mConverterParams);
converter.ToVal(mData.at(rowIdx).at(columnIdx), val);
return val;
}


template<typename T>
T GetCell(const size_t pColumnIdx, const size_t pRowIdx, ConvFunc<T> pToVal) const
{
const ssize_t columnIdx = pColumnIdx + (mLabelParams.mRowNameIdx + 1);
const ssize_t rowIdx = pRowIdx + (mLabelParams.mColumnNameIdx + 1);

T val;
pToVal(mData.at(rowIdx).at(columnIdx), val);
return val;
}


template<typename T>
T GetCell(const std::string& pColumnName, const std::string& pRowName) const
{
const ssize_t columnIdx = GetColumnIdx(pColumnName);
if (columnIdx < 0)
{
throw std::out_of_range("column not found: " + pColumnName);
}

const ssize_t rowIdx = GetRowIdx(pRowName);
if (rowIdx < 0)
{
throw std::out_of_range("row not found: " + pRowName);
}

return GetCell<T>(columnIdx, rowIdx);
}


template<typename T>
T GetCell(const std::string& pColumnName, const std::string& pRowName, ConvFunc<T> pToVal) const
{
const ssize_t columnIdx = GetColumnIdx(pColumnName);
if (columnIdx < 0)
{
throw std::out_of_range("column not found: " + pColumnName);
}

const ssize_t rowIdx = GetRowIdx(pRowName);
if (rowIdx < 0)
{
throw std::out_of_range("row not found: " + pRowName);
}

return GetCell<T>(columnIdx, rowIdx, pToVal);
}


template<typename T>
T GetCell(const std::string& pColumnName, const size_t pRowIdx) const
{
const ssize_t columnIdx = GetColumnIdx(pColumnName);
if (columnIdx < 0)
{
throw std::out_of_range("column not found: " + pColumnName);
}

return GetCell<T>(columnIdx, pRowIdx);
}


template<typename T>
T GetCell(const std::string& pColumnName, const size_t pRowIdx, ConvFunc<T> pToVal) const
{
const ssize_t columnIdx = GetColumnIdx(pColumnName);
if (columnIdx < 0)
{
throw std::out_of_range("column not found: " + pColumnName);
}

return GetCell<T>(columnIdx, pRowIdx, pToVal);
}


template<typename T>
T GetCell(const size_t pColumnIdx, const std::string& pRowName) const
{
const ssize_t rowIdx = GetRowIdx(pRowName);
if (rowIdx < 0)
{
throw std::out_of_range("row not found: " + pRowName);
}

return GetCell<T>(pColumnIdx, rowIdx);
}


template<typename T>
T GetCell(const size_t pColumnIdx, const std::string& pRowName, ConvFunc<T> pToVal) const
{
const ssize_t rowIdx = GetRowIdx(pRowName);
if (rowIdx < 0)
{
throw std::out_of_range("row not found: " + pRowName);
}

return GetCell<T>(pColumnIdx, rowIdx, pToVal);
}


template<typename T>
void SetCell(const size_t pColumnIdx, const size_t pRowIdx, const T& pCell)
{
const size_t columnIdx = pColumnIdx + (mLabelParams.mRowNameIdx + 1);
const size_t rowIdx = pRowIdx + (mLabelParams.mColumnNameIdx + 1);

while ((rowIdx + 1) > GetDataRowCount())
{
std::vector<std::string> row;
row.resize(GetDataColumnCount());
mData.push_back(row);
}

if ((columnIdx + 1) > GetDataColumnCount())
{
for (auto itRow = mData.begin(); itRow != mData.end(); ++itRow)
{
itRow->resize(columnIdx + 1);
}
}

std::string str;
Converter<T> converter(mConverterParams);
converter.ToStr(pCell, str);
mData.at(rowIdx).at(columnIdx) = str;
}


template<typename T>
void SetCell(const std::string& pColumnName, const std::string& pRowName, const T& pCell)
{
const ssize_t columnIdx = GetColumnIdx(pColumnName);
if (columnIdx < 0)
{
throw std::out_of_range("column not found: " + pColumnName);
}

const ssize_t rowIdx = GetRowIdx(pRowName);
if (rowIdx < 0)
{
throw std::out_of_range("row not found: " + pRowName);
}

SetCell<T>(columnIdx, rowIdx, pCell);
}


std::string GetColumnName(const ssize_t pColumnIdx)
{
const ssize_t columnIdx = pColumnIdx + (mLabelParams.mRowNameIdx + 1);
if (mLabelParams.mColumnNameIdx < 0)
{
throw std::out_of_range("column name row index < 0: " + std::to_string(mLabelParams.mColumnNameIdx));
}

return mData.at(mLabelParams.mColumnNameIdx).at(columnIdx);
}


void SetColumnName(size_t pColumnIdx, const std::string& pColumnName)
{
const ssize_t columnIdx = pColumnIdx + (mLabelParams.mRowNameIdx + 1);
mColumnNames[pColumnName] = columnIdx;
if (mLabelParams.mColumnNameIdx < 0)
{
throw std::out_of_range("column name row index < 0: " + std::to_string(mLabelParams.mColumnNameIdx));
}

const int rowIdx = mLabelParams.mColumnNameIdx;
if (rowIdx >= (int) mData.size())
{
mData.resize(rowIdx + 1);
}
auto& row = mData[rowIdx];
if (columnIdx >= (int) row.size())
{
row.resize(columnIdx + 1);
}

mData.at(mLabelParams.mColumnNameIdx).at(columnIdx) = pColumnName;
}


std::vector<std::string> GetColumnNames()
{
if (mLabelParams.mColumnNameIdx >= 0)
{
return std::vector<std::string>(mData.at(mLabelParams.mColumnNameIdx).begin() +
(mLabelParams.mRowNameIdx + 1),
mData.at(mLabelParams.mColumnNameIdx).end());
}

return std::vector<std::string>();
}


std::string GetRowName(const ssize_t pRowIdx)
{
const ssize_t rowIdx = pRowIdx + (mLabelParams.mColumnNameIdx + 1);
if (mLabelParams.mRowNameIdx < 0)
{
throw std::out_of_range("row name column index < 0: " + std::to_string(mLabelParams.mRowNameIdx));
}

return mData.at(rowIdx).at(mLabelParams.mRowNameIdx);
}


void SetRowName(size_t pRowIdx, const std::string& pRowName)
{
const ssize_t rowIdx = pRowIdx + (mLabelParams.mColumnNameIdx + 1);
mRowNames[pRowName] = rowIdx;
if (mLabelParams.mRowNameIdx < 0)
{
throw std::out_of_range("row name column index < 0: " + std::to_string(mLabelParams.mRowNameIdx));
}

if (rowIdx >= (int) mData.size())
{
mData.resize(rowIdx + 1);
}
auto& row = mData[rowIdx];
if (mLabelParams.mRowNameIdx >= (int) row.size())
{
row.resize(mLabelParams.mRowNameIdx + 1);
}

mData.at(rowIdx).at(mLabelParams.mRowNameIdx) = pRowName;
}


std::vector<std::string> GetRowNames()
{
std::vector<std::string> rownames;
if (mLabelParams.mRowNameIdx >= 0)
{
for (auto itRow = mData.begin(); itRow != mData.end(); ++itRow)
{
if (std::distance(mData.begin(), itRow) > mLabelParams.mColumnNameIdx)
{
rownames.push_back(itRow->at(mLabelParams.mRowNameIdx));
}
}
}
return rownames;
}

private:
void ReadCsv()
{
std::ifstream stream;
stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
stream.open(mPath, std::ios::binary);
ReadCsv(stream);
}

void ReadCsv(std::istream& pStream)
{
pStream.seekg(0, std::ios::end);
std::streamsize length = pStream.tellg();
pStream.seekg(0, std::ios::beg);

#ifdef HAS_CODECVT
std::vector<char> bom2b(2, '\0');
if (length >= 2)
{
pStream.read(bom2b.data(), 2);
pStream.seekg(0, std::ios::beg);
}

static const std::vector<char> bomU16le = { '\xff', '\xfe' };
static const std::vector<char> bomU16be = { '\xfe', '\xff' };
if ((bom2b == bomU16le) || (bom2b == bomU16be))
{
mIsUtf16 = true;
mIsLE = (bom2b == bomU16le);

std::wifstream wstream;
wstream.exceptions(std::wifstream::failbit | std::wifstream::badbit);
wstream.open(mPath, std::ios::binary);
if (mIsLE)
{
wstream.imbue(std::locale(wstream.getloc(),
new std::codecvt_utf16<wchar_t, 0x10ffff,
static_cast<std::codecvt_mode>(std::consume_header |
std::little_endian)>));
}
else
{
wstream.imbue(std::locale(wstream.getloc(),
new std::codecvt_utf16<wchar_t, 0x10ffff,
std::consume_header>));
}
std::wstringstream wss;
wss << wstream.rdbuf();
std::string utf8 = ToString(wss.str());
std::stringstream ss(utf8);
ParseCsv(ss, utf8.size());
}
else
#endif
{
if (length >= 3)
{
std::vector<char> bom3b(3, '\0');
pStream.read(bom3b.data(), 3);
static const std::vector<char> bomU8 = { '\xef', '\xbb', '\xbf' };
if (bom3b != bomU8)
{
pStream.seekg(0, std::ios::beg);
}
else
{
length -= 3;
}
}

ParseCsv(pStream, length);
}
}

void ParseCsv(std::istream& pStream, std::streamsize p_FileLength)
{
const std::streamsize bufLength = 64 * 1024;
std::vector<char> buffer(bufLength);
std::vector<std::string> row;
std::string cell;
bool quoted = false;
int cr = 0;
int lf = 0;

while (p_FileLength > 0)
{
std::streamsize readLength = std::min(p_FileLength, bufLength);
pStream.read(buffer.data(), readLength);
for (int i = 0; i < readLength; ++i)
{
if (buffer[i] == '"')
{
if (cell.empty() || cell[0] == '"')
{
quoted = !quoted;
}
cell += buffer[i];
}
else if (buffer[i] == mSeparatorParams.mSeparator)
{
if (!quoted)
{
row.push_back(Unquote(Trim(cell)));
cell.clear();
}
else
{
cell += buffer[i];
}
}
else if (buffer[i] == '\r')
{
if (mSeparatorParams.mQuotedLinebreaks && quoted)
{
cell += buffer[i];
}
else
{
++cr;
}
}
else if (buffer[i] == '\n')
{
if (mSeparatorParams.mQuotedLinebreaks && quoted)
{
cell += buffer[i];
}
else
{
++lf;
row.push_back(Unquote(Trim(cell)));
cell.clear();
mData.push_back(row);
row.clear();
quoted = false;
}
}
else
{
cell += buffer[i];
}
}
p_FileLength -= readLength;
}

if (!cell.empty() || !row.empty())
{
row.push_back(Unquote(Trim(cell)));
cell.clear();
mData.push_back(row);
row.clear();
}

mSeparatorParams.mHasCR = (cr > (lf / 2));

if ((mLabelParams.mColumnNameIdx >= 0) &&
(static_cast<ssize_t>(mData.size()) > mLabelParams.mColumnNameIdx))
{
int i = 0;
for (auto& columnName : mData[mLabelParams.mColumnNameIdx])
{
mColumnNames[columnName] = i++;
}
}

if ((mLabelParams.mRowNameIdx >= 0) &&
(static_cast<ssize_t>(mData.size()) >
(mLabelParams.mColumnNameIdx + 1)))
{
int i = 0;
for (auto& dataRow : mData)
{
if (static_cast<ssize_t>(dataRow.size()) > mLabelParams.mRowNameIdx)
{
mRowNames[dataRow[mLabelParams.mRowNameIdx]] = i++;
}
}
}
}

void WriteCsv() const
{
#ifdef HAS_CODECVT
if (mIsUtf16)
{
std::stringstream ss;
WriteCsv(ss);
std::string utf8 = ss.str();
std::wstring wstr = ToWString(utf8);

std::wofstream wstream;
wstream.exceptions(std::wofstream::failbit | std::wofstream::badbit);
wstream.open(mPath, std::ios::binary | std::ios::trunc);

if (mIsLE)
{
wstream.imbue(std::locale(wstream.getloc(),
new std::codecvt_utf16<wchar_t, 0x10ffff,
static_cast<std::codecvt_mode>(std::little_endian)>));
}
else
{
wstream.imbue(std::locale(wstream.getloc(),
new std::codecvt_utf16<wchar_t, 0x10ffff>));
}

wstream << (wchar_t) 0xfeff;
wstream << wstr;
}
else
#endif
{
std::ofstream stream;
stream.exceptions(std::ofstream::failbit | std::ofstream::badbit);
stream.open(mPath, std::ios::binary | std::ios::trunc);
WriteCsv(stream);
}
}

void WriteCsv(std::ostream& pStream) const
{
for (auto itr = mData.begin(); itr != mData.end(); ++itr)
{
for (auto itc = itr->begin(); itc != itr->end(); ++itc)
{
if (mSeparatorParams.mAutoQuote &&
((itc->find(mSeparatorParams.mSeparator) != std::string::npos) ||
(itc->find(' ') != std::string::npos)))
{
std::string str = *itc;
ReplaceString(str, "\"", "\"\"");

pStream << "\"" << str << "\"";
}
else
{
pStream << *itc;
}

if (std::distance(itc, itr->end()) > 1)
{
pStream << mSeparatorParams.mSeparator;
}
}
pStream << (mSeparatorParams.mHasCR ? "\r\n" : "\n");
}
}

size_t GetDataRowCount() const
{
return mData.size();
}

size_t GetDataColumnCount() const
{
return (mData.size() > 0) ? mData.at(0).size() : 0;
}

std::string Trim(const std::string& pStr)
{
if (mSeparatorParams.mTrim)
{
std::string str = pStr;

str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](int ch) { return !isspace(ch); }));

str.erase(std::find_if(str.rbegin(), str.rend(), [](int ch) { return !isspace(ch); }).base(), str.end());

return str;
}
else
{
return pStr;
}
}

std::string Unquote(const std::string& pStr)
{
if (mSeparatorParams.mAutoQuote && (pStr.size() >= 2) && (pStr.front() == '"') && (pStr.back() == '"'))
{
std::string str = pStr.substr(1, pStr.size() - 2);

ReplaceString(str, "\"\"", "\"");

return str;
}
else
{
return pStr;
}
}

#ifdef HAS_CODECVT
#if defined(_MSC_VER)
#pragma warning (disable: 4996)
#endif
static std::string ToString(const std::wstring& pWStr)
{
size_t len = std::wcstombs(nullptr, pWStr.c_str(), 0) + 1;
char* cstr = new char[len];
std::wcstombs(cstr, pWStr.c_str(), len);
std::string str(cstr);
delete[] cstr;
return str;
}

static std::wstring ToWString(const std::string& pStr)
{
size_t len = 1 + mbstowcs(nullptr, pStr.c_str(), 0);
wchar_t* wcstr = new wchar_t[len];
std::mbstowcs(wcstr, pStr.c_str(), len);
std::wstring wstr(wcstr);
delete[] wcstr;
return wstr;
}
#if defined(_MSC_VER)
#pragma warning (default: 4996)
#endif
#endif

static void ReplaceString(std::string& pStr, const std::string& pSearch, const std::string& pReplace)
{
size_t pos = 0;

while ((pos = pStr.find(pSearch, pos)) != std::string::npos)
{
pStr.replace(pos, pSearch.size(), pReplace);
pos += pReplace.size();
}
}

private:
std::string mPath;
LabelParams mLabelParams;
SeparatorParams mSeparatorParams;
ConverterParams mConverterParams;
std::vector<std::vector<std::string>> mData;
std::map<std::string, size_t> mColumnNames;
std::map<std::string, size_t> mRowNames;
#ifdef HAS_CODECVT
bool mIsUtf16 = false;
bool mIsLE = false;
#endif
};
}
