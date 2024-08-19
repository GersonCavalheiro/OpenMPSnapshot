


















#include "smpdd.hpp"
#include "debug.hpp"
#include "system.hpp"
#include "smp_ult.hpp"
#include "instrumentation.hpp"
#include "taskexecutionexception.hpp"
#include "smpdevice.hpp"
#include "schedule.hpp"
#include <string>

#ifdef NANOS_DEBUG_ENABLED
#include <valgrind/valgrind.h>
#endif

using namespace nanos;
using namespace nanos::ext;


SMPDevice &nanos::ext::getSMPDevice() {
return sys._getSMPDevice();
}

size_t SMPDD::_stackSize = 256*1024;

inline SMPDD::~SMPDD() {
if ( _stack ) delete[] (char *) _stack;
#ifdef NANOS_DEBUG_ENABLED
VALGRIND_STACK_DEREGISTER( _valgrind_stack_id );
#endif
}

void SMPDD::prepareConfig ( Config &config )
{
size_t size = sys.getDeviceStackSize();
if (size > 0) _stackSize = size;

config.registerConfigOption ( "smp-stack-size", NEW Config::SizeVar( _stackSize ), "Defines SMP::task stack size" );
config.registerArgOption("smp-stack-size", "smp-stack-size");
}

void SMPDD::initStack ( WD *wd )
{
_state = ::initContext(_stack, _stackSize, &workWrapper, wd, (void *) Scheduler::exit, 0);
}

void SMPDD::workWrapper ( WD &wd )
{
SMPDD &dd = (SMPDD &) wd.getActiveDevice();
#ifdef NANOS_INSTRUMENTATION_ENABLED
NANOS_INSTRUMENT ( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("user-code") );
NANOS_INSTRUMENT ( nanos_event_value_t val = wd.getId() );
NANOS_INSTRUMENT ( if ( wd.isRuntimeTask() ) { );
NANOS_INSTRUMENT (    sys.getInstrumentation()->raiseOpenStateEvent ( NANOS_RUNTIME ) );
NANOS_INSTRUMENT ( } else { );
NANOS_INSTRUMENT (    sys.getInstrumentation()->raiseOpenStateAndBurst ( NANOS_RUNNING, key, val ) );
NANOS_INSTRUMENT ( } );
#endif

dd.execute(wd);

#ifdef NANOS_INSTRUMENTATION_ENABLED
NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseCloseStateAndBurst ( key, val ) );
NANOS_INSTRUMENT ( if ( wd.isRuntimeTask() ) { );
NANOS_INSTRUMENT (    sys.getInstrumentation()->raiseCloseStateEvent() );
NANOS_INSTRUMENT ( } else { );
NANOS_INSTRUMENT (    sys.getInstrumentation()->raiseCloseStateAndBurst ( key, val ) );
NANOS_INSTRUMENT ( } );
#endif
}

void SMPDD::lazyInit ( WD &wd, bool isUserLevelThread, WD *previous )
{
verbose0("Task " << wd.getId() << " initialization"); 
if (isUserLevelThread) {
if (previous == NULL) {
_stack = (void *) NEW char[_stackSize];
#ifdef NANOS_DEBUG_ENABLED
_valgrind_stack_id = VALGRIND_STACK_REGISTER(
_stack, (void*)(((uintptr_t)_stack)+_stackSize) );
#endif
verbose0("   new stack created: " << _stackSize << " bytes");
} else {
verbose0("   reusing stacks");
SMPDD &oldDD = (SMPDD &) previous->getActiveDevice();
std::swap(_stack, oldDD._stack);
}
initStack(&wd);
}
}

SMPDD * SMPDD::copyTo ( void *toAddr )
{
SMPDD *dd = new (toAddr) SMPDD(*this);
return dd;
}

void SMPDD::execute ( WD &wd ) throw ()
{
#ifdef NANOS_RESILIENCY_ENABLED
bool retry = false;
int num_tries = 0;
if (wd.isInvalid() || (wd.getParent() != NULL && wd.getParent()->isInvalid())) {

wd.setInvalid(true);
debug ( "Task " << wd.getId() << " is flagged as invalid.");
} else {
while (true) {
try {
getWorkFct()( wd.getData() );
} catch (TaskExecutionException& e) {

sigset_t sigs;
sigemptyset(&sigs);
sigaddset(&sigs, e.getSignal());
pthread_sigmask(SIG_UNBLOCK, &sigs, NULL);

if(!wd.setInvalid(true)) { 
message(e.what());
std::terminate();
} else { 
debug( e.what() );
}
} catch (std::exception& e) {
std::stringstream ss;
ss << "Uncaught exception "
<< typeid(e).name()
<< ". Thrown in task "
<< wd.getId()
<< ". \n"
<< e.what();
message(ss.str());
std::terminate();
} catch (...) {
message("Uncaught exception (unknown type). Thrown in task " << wd.getId() << ". ");
std::terminate();
}
retry = wd.isInvalid()
&& wd.isRecoverable()
&& (wd.getParent() == NULL || !wd.getParent()->isInvalid())
&& num_tries < sys.getTaskMaxRetries();

if (!retry)
break;

num_tries++;
recover(wd);
}
}
#else
getWorkFct()(wd.getData());
#endif
}

#ifdef NANOS_RESILIENCY_ENABLED
void SMPDD::recover( WD & wd ) {
wd.waitCompletion();


wd.setInvalid(false);
}
#endif
