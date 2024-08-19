

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>



namespace Aws
{
namespace Utils
{

class AWS_CORE_API StringUtils
{
public:
static void Replace(Aws::String& s, const char* search, const char* replace);



static Aws::String ToLower(const char* source);



static Aws::String ToUpper(const char* source);



static bool CaselessCompare(const char* value1, const char* value2);



static Aws::String URLEncode(const char* unsafe);


static Aws::String UTF8Escape(const char* unicodeString, const char* delimiter);


static Aws::String URLEncode(double unsafe);



static Aws::String URLDecode(const char* safe);



static Aws::Vector<Aws::String> Split(const Aws::String& toSplit, char splitOn);



static Aws::Vector<Aws::String> SplitOnLine(const Aws::String& toSplit);



static Aws::String LTrim(const char* source);



static Aws::String RTrim(const char* source);


static Aws::String Trim(const char* source);



static long long ConvertToInt64(const char* source);



static long ConvertToInt32(const char* source);



static bool ConvertToBool(const char* source);



static double ConvertToDouble(const char* source);


#ifdef _WIN32

static Aws::WString ToWString(const char* source);


static Aws::String FromWString(const wchar_t* source);
#endif


template< typename T >
static Aws::String to_string(T value)
{
Aws::OStringStream os;
os << value;
return os.str();
}

};


} 
} 


