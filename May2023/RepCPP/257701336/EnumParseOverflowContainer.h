

#pragma once

#include <aws/core/utils/memory/stl/AWSMap.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/Core_EXPORTS.h>
#include <mutex>

namespace Aws
{
namespace Utils
{

class AWS_CORE_API EnumParseOverflowContainer
{
public:
const Aws::String& RetrieveOverflow(int hashCode) const;
void StoreOverflow(int hashCode, const Aws::String& value);

private:
mutable std::mutex m_overflowLock;
Aws::Map<int, Aws::String> m_overflowMap;
Aws::String m_emptyString;
};
}
}


