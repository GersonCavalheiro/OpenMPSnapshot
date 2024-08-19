


#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/Array.h>
#include <streambuf>

namespace Aws
{
namespace Utils
{
namespace Stream
{

class AWS_CORE_API PreallocatedStreamBuf : public std::streambuf
{
public:

PreallocatedStreamBuf(Aws::Utils::Array<uint8_t>* buffer, std::size_t lengthToRead);

PreallocatedStreamBuf(const PreallocatedStreamBuf&) = delete;
PreallocatedStreamBuf& operator=(const PreallocatedStreamBuf&) = delete;

PreallocatedStreamBuf(PreallocatedStreamBuf&& toMove) = delete;
PreallocatedStreamBuf& operator=(PreallocatedStreamBuf&&) = delete;


Aws::Utils::Array<uint8_t>* GetBuffer() { return m_underlyingBuffer; }

protected:
pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out) override;
pos_type seekpos(pos_type pos, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out) override;

private:
Aws::Utils::Array<uint8_t>* m_underlyingBuffer;
std::size_t m_lengthToRead;
};
}
}
}
