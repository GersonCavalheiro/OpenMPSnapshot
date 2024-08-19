

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSStreamFwd.h>

namespace Aws
{
namespace Utils
{
namespace Stream
{

class AWS_CORE_API ResponseStream
{
public:

ResponseStream();

ResponseStream(ResponseStream&&);

ResponseStream(const Aws::IOStreamFactory& factory);

ResponseStream(IOStream* underlyingStreamToManage);
ResponseStream(const ResponseStream&) = delete;
~ResponseStream();


ResponseStream& operator=(ResponseStream&&);
ResponseStream& operator=(const ResponseStream&) = delete;


inline Aws::IOStream& GetUnderlyingStream() const { return *m_underlyingStream; }

private:
void ReleaseStream();

Aws::IOStream* m_underlyingStream;
};

class AWS_CORE_API DefaultUnderlyingStream : public Aws::IOStream
{
public:
using Base = Aws::IOStream;

DefaultUnderlyingStream();
virtual ~DefaultUnderlyingStream();
};

AWS_CORE_API Aws::IOStream* DefaultResponseStreamFactoryMethod();

} 
} 
} 
