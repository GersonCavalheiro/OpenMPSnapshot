

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSStack.h>
#include <functional>
#include <atomic>

namespace Aws
{
namespace Utils
{

class AWS_CORE_API GetTheLights
{
public:
GetTheLights();
void EnterRoom(std::function<void()>&&);
void LeaveRoom(std::function<void()>&&);
private:
std::atomic<int> m_value;
};
}
}