

#pragma once

#ifdef __APPLE__

#ifdef __clang__
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif 

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif 

#endif 

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/Array.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace Utils
{
namespace Base64
{


class AWS_CORE_API Base64
{
public:
Base64(const char *encodingTable = nullptr);


Aws::String Encode(const ByteBuffer&) const;


ByteBuffer Decode(const Aws::String&) const;


static size_t CalculateBase64DecodedLength(const Aws::String& b64input);

static size_t CalculateBase64EncodedLength(const ByteBuffer& buffer);

private:
char m_mimeBase64EncodingTable[64];
uint8_t m_mimeBase64DecodingTable[256];

};

} 
} 
} 

