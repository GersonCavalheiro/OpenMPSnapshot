


#ifndef OMP_EXCEPTION_CATCHER_H
#define OMP_EXCEPTION_CATCHER_H

#include <exception>
#include <mutex>
#include <functional>
#include <cstdint>

namespace omp_exception_catcher {

enum class Strategy {DoNotTry, Continue, Abort, RethrowFirst};

namespace impl_ {

template<class _dummy=void>
class OMPExceptionCatcher
{
static Strategy GlobalDefaultStrategy; 
public:
static void setGlobalDefaultStrategy(Strategy s) { GlobalDefaultStrategy = s; }


OMPExceptionCatcher(): ex(nullptr), strategy(GlobalDefaultStrategy) {}


OMPExceptionCatcher(Strategy strategy_): ex(nullptr), strategy(strategy_) {}


void rethrow() const { if(strategy==Strategy::RethrowFirst && ex) std::rethrow_exception(ex); }


template<class Function, class... Parameters>
void run(Function func, Parameters... params) {
switch(strategy) {
case Strategy::DoNotTry:
func(params...);
break;
case Strategy::Continue:
try { func(params...); }
catch (...) { }
break;
case Strategy::Abort:
try { func(params...); }
catch (...) { std::abort(); }
break;
case Strategy::RethrowFirst:
try { func(params...); }
catch (...) { capture(); }
break;
}
}

private:
std::exception_ptr ex;
std::mutex lock;
Strategy strategy;

void capture() {
std::unique_lock<std::mutex> guard(lock);
if(!ex) ex = std::current_exception();
}
};

template<class IntType>
Strategy OMPExceptionCatcher<IntType>::GlobalDefaultStrategy = Strategy::RethrowFirst;

} 


using OMPExceptionCatcher = impl_::OMPExceptionCatcher<>;

} 

#endif
