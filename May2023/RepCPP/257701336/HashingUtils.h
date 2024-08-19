

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/memory/stl/AWSStreamFwd.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/Array.h>

namespace Aws
{
namespace Utils
{


class AWS_CORE_API HashingUtils
{
public:

static Aws::String Base64Encode(const ByteBuffer& byteBuffer);


static ByteBuffer Base64Decode(const Aws::String&);


static Aws::String HexEncode(const ByteBuffer& byteBuffer);


static ByteBuffer HexDecode(const Aws::String& str);


static ByteBuffer CalculateSHA256HMAC(const ByteBuffer& toSign, const ByteBuffer& secret);


static ByteBuffer CalculateSHA256(const Aws::String& str);


static ByteBuffer CalculateSHA256(Aws::IOStream& stream);


static ByteBuffer CalculateSHA256TreeHash(const Aws::String& str);


static ByteBuffer CalculateSHA256TreeHash(Aws::IOStream& stream);


static ByteBuffer CalculateMD5(const Aws::String& str);


static ByteBuffer CalculateMD5(Aws::IOStream& stream);

static int HashString(const char* strToHash);

};

} 
} 

