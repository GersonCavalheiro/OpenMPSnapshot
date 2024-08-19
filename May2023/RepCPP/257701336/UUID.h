

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/Array.h>

namespace Aws
{
namespace Utils
{
static const size_t UUID_BINARY_SIZE = 0x10;
static const size_t UUID_STR_SIZE = 0x24;


class AWS_CORE_API UUID
{
public:

UUID(const Aws::String&);

UUID(const unsigned char uuid[UUID_BINARY_SIZE]);


operator Aws::String();

inline operator ByteBuffer() { return ByteBuffer(m_uuid, sizeof(m_uuid)); }


static UUID RandomUUID();

private:
unsigned char m_uuid[UUID_BINARY_SIZE];
};
}
}