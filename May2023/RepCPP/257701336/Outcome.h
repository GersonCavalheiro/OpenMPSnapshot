

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <utility>

namespace Aws
{
namespace Utils
{


template<typename R, typename E> 
class Outcome
{
public:

Outcome() : success(false)
{
} 
Outcome(const R& r) : result(r), success(true)
{
} 
Outcome(const E& e) : error(e), success(false)
{
} 
Outcome(R&& r) : result(std::forward<R>(r)), success(true)
{
} 
Outcome(E&& e) : error(std::forward<E>(e)), success(false)
{
} 
Outcome(const Outcome& o) :
result(o.result),
error(o.error),
success(o.success)
{
}

Outcome& operator=(const Outcome& o)
{
if (this != &o)
{
result = o.result;
error = o.error;
success = o.success;
}

return *this;
}

Outcome(Outcome&& o) : 
result(std::move(o.result)),
error(std::move(o.error)),
success(o.success)
{
}

Outcome& operator=(Outcome&& o)
{
if (this != &o)
{
result = std::move(o.result);
error = std::move(o.error);
success = o.success;
}

return *this;
}

inline const R& GetResult() const
{
return result;
}

inline R& GetResult()
{
return result;
}


inline R&& GetResultWithOwnership()
{
return std::move(result);
}

inline const E& GetError() const
{
return error;
}

inline bool IsSuccess() const
{
return this->success;
}

private:
R result;
E error;
bool success;
};

} 
} 
