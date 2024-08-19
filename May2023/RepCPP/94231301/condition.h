
#pragma once

#include "mutex.h"

namespace embree
{
class ConditionSys
{
public:
ConditionSys();
~ConditionSys();
void wait( class MutexSys& mutex );
void notify_all();

template<typename Predicate>
__forceinline void wait( class MutexSys& mutex, const Predicate& pred )
{
while (!pred()) wait(mutex);
}

private:
ConditionSys (const ConditionSys& other) DELETED; 
ConditionSys& operator= (const ConditionSys& other) DELETED; 

protected:
void* cond;
};
}
