

#pragma once

#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace Client
{

class AWS_CORE_API AsyncCallerContext
{
public:

AsyncCallerContext();


AsyncCallerContext(const Aws::String& uuid) : m_uuid(uuid) {}


AsyncCallerContext(const char* uuid) : m_uuid(uuid) {}

virtual ~AsyncCallerContext() {}


inline const Aws::String& GetUUID() const { return m_uuid; }


inline void SetUUID(const Aws::String& value) { m_uuid = value; }


inline void SetUUID(const char* value) { m_uuid.assign(value); }

private:
Aws::String m_uuid;
};
}
}

