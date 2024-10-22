


#ifndef fpedetection_hh_
#define fpedetection_hh_ 1

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#include <iostream>
#include <stdlib.h>  
#include <set>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <exception>
#include <stdexcept>

#   if defined(__MACH__)
#     ifndef FPE_INTDIV
#       define FPE_INTDIV FPE_NOOP
#     endif
#   endif

#ifndef CXX11
#   if __cplusplus > 199711L   
#       define CXX11
#   endif
#endif

#ifndef CXX14
#   if __cplusplus > 201103L   
#       define CXX14
#   endif
#endif


namespace mad
{

#ifdef CXX11
enum class fpe
{
divbyzero,
downward,
inexact,
invalid,
overflow,
tonearest,
towardzero,
underflow,
upward
};
#else
struct fpe
{
enum fpe_enum
{
divbyzero,
downward,
inexact,
invalid,
overflow,
tonearest,
towardzero,
underflow,
upward
};

void operator=(const fpe_enum& val) { fpe_val = val; }
operator fpe_enum() const { return fpe_val; }

private:
fpe_enum fpe_val;
};
#endif


class fpe_settings
{
public:
typedef std::set<fpe> fpe_set;

public:
static void enable(const fpe&);
static void disable(const fpe&);
static fpe_set enabled() { check_environment(); return fpe_enabled; }
static fpe_set disabled() { check_environment(); return fpe_disabled; }
static std::string str(const fpe&);
static std::string str();
static void check_environment();

private:
static fpe_set fpe_default;
static fpe_set fpe_enabled;
static fpe_set fpe_disabled;
};


} 

#if !(defined(__bgq__))

# if (defined(__GNUC__) && !defined(__clang__))

#   if defined(__linux__)
#     include <features.h>
#     include <fenv.h>
#     include <csignal>
#     include <execinfo.h> 
#     include <cxxabi.h>

namespace mad
{


static struct sigaction fpe_termaction, fpe_oldaction;


inline void StackBackTrace(std::stringstream& ss)
{
#   define BSIZE 50
void* buffer[ BSIZE ];
int nptrs = backtrace( buffer, BSIZE );
char** strings = backtrace_symbols( buffer, nptrs );
if(strings == NULL)
{
perror( "backtrace_symbols" );
return;
}

ss << std::endl << "Call Stack:" << std::endl;
for(int j = 0; j < nptrs; j++)
{
ss << std::setw(3) << nptrs-j-1 << " / "
<< std::setw(3) << nptrs << " : ";
char* mangled_start = strchr(strings[j],  '(' ) + 1;
if (mangled_start)
*(mangled_start-1) = '\0';
char* mangled_end   = strchr(mangled_start,'+' );
if (mangled_end)
*mangled_end = '\0';
int status = 0;
char* realname = 0;
if(mangled_end && strlen(mangled_start))
realname = abi::__cxa_demangle( mangled_start, 0, 0, &status);
if(realname)
{
ss << strings[j] << " : " << realname  << std::endl;
free(realname);
} else
{
ss << strings[j] << std::endl;
}
}

free(strings);
}


inline void TerminationSignalHandler(int sig, siginfo_t* sinfo,
void* )
{
std::stringstream message;
message << "### ERROR ### : " << sig << " @ address " << sinfo->si_addr
<< "\n - ";
if(sig == SIGSEGV)
message << "Segmentation Fault.";
else
message << "Floating-point exception (FPE).";

message << " : ";
if (sinfo)
{
if(sig == SIGSEGV)
{
switch (sinfo->si_code)
{
case SEGV_MAPERR:
message << "Address not mapped to object.";
break;
case SEGV_ACCERR:
message << "Invalid permissions for mapped object.";
break;
default:
message << "Unknown error: " << sinfo->si_code << ".";
break;
}
} else
{
switch (sinfo->si_code)
{
case FPE_INTDIV:
message << "Integer divide by zero.";
break;
case FPE_INTOVF:
message << "Integer overflow.";
break;
case FPE_FLTDIV:
message << "Floating point divide by zero.";
break;
case FPE_FLTOVF:
message << "Floating point overflow.";
break;
case FPE_FLTUND:
message << "Floating point underflow.";
break;
case FPE_FLTRES:
message << "Floating point inexact result.";
break;
case FPE_FLTINV:
message << "Floating point invalid operation.";
break;
case FPE_FLTSUB:
message << "Subscript out of range.";
break;
default:
message << "Unknown error.";
break;
}
}
}

message << std::endl;
StackBackTrace(message);
throw std::runtime_error(message.str());
}


inline bool EnableInvalidOperationDetection(fpe_settings::fpe_set operations
= fpe_settings::fpe_set())
{
static bool init_first = true;
if(init_first)
init_first = false;
else
return false;

typedef fpe_settings::fpe_set::const_iterator const_iterator;
if(operations.empty())
operations = fpe_settings::enabled();
for(const_iterator itr = operations.begin(); itr != operations.end(); ++itr)
{
switch(*itr)
{
case fpe::divbyzero:
(void) feenableexcept( FE_DIVBYZERO   );  
break;
case fpe::downward:
(void) feenableexcept( FE_DOWNWARD    );
break;
case fpe::inexact:
(void) feenableexcept( FE_INEXACT     );
break;
case fpe::invalid:
(void) feenableexcept( FE_INVALID     );  
break;
case fpe::overflow:
(void) feenableexcept( FE_OVERFLOW    );
break;
case fpe::tonearest:
(void) feenableexcept( FE_TONEAREST   );
break;
case fpe::towardzero:
(void) feenableexcept( FE_TOWARDZERO  );
break;
case fpe::underflow:
(void) feenableexcept( FE_UNDERFLOW   );
break;
case fpe::upward:
(void) feenableexcept( FE_UPWARD      );
break;
}
}

sigfillset(&fpe_termaction.sa_mask);
sigdelset(&fpe_termaction.sa_mask, SIGFPE);
sigdelset(&fpe_termaction.sa_mask, SIGSEGV);
fpe_termaction.sa_sigaction = TerminationSignalHandler;
fpe_termaction.sa_flags = SA_SIGINFO;
sigaction(SIGFPE,  &fpe_termaction, &fpe_oldaction);
sigaction(SIGSEGV, &fpe_termaction, &fpe_oldaction);

return true;
}


inline void DisableInvalidOperationDetection()
{
sigemptyset(&fpe_termaction.sa_mask);
fpe_termaction.sa_handler = SIG_DFL;
sigaction(SIGFPE,  &fpe_termaction, 0);
sigaction(SIGSEGV, &fpe_termaction, 0);
}


} 

#   elif defined(__MACH__)      
#     include <fenv.h>
#     include <signal.h>

#     if (defined(__ppc__) || defined(__ppc64__))

#       define FE_EXCEPT_SHIFT 22  
#       define FM_ALL_EXCEPT    FE_ALL_EXCEPT >> FE_EXCEPT_SHIFT


namespace mad
{


static inline int feenableexcept (unsigned int excepts)
{
static fenv_t fenv;
unsigned int new_excepts = (excepts & FE_ALL_EXCEPT) >> FE_EXCEPT_SHIFT,
old_excepts;  

if ( fegetenv (&fenv) )  { return -1; }
old_excepts = (fenv & FM_ALL_EXCEPT) << FE_EXCEPT_SHIFT;
fenv = (fenv & ~new_excepts) | new_excepts;

return ( fesetenv (&fenv) ? -1 : old_excepts );
}


static inline int fedisableexcept (unsigned int excepts)
{
static fenv_t fenv;
unsigned int still_on = ~((excepts & FE_ALL_EXCEPT) >> FE_EXCEPT_SHIFT),
old_excepts;  

if ( fegetenv (&fenv) )  { return -1; }
old_excepts = (fenv & FM_ALL_EXCEPT) << FE_EXCEPT_SHIFT;
fenv &= still_on;

return ( fesetenv (&fenv) ? -1 : old_excepts );
}


} 

#     elif  (defined(__i386__) || defined(__x86_64__))

namespace mad
{


static inline int feenableexcept (unsigned int excepts)
{
static fenv_t fenv;
unsigned int new_excepts = excepts & FE_ALL_EXCEPT,
old_excepts;  

if ( fegetenv (&fenv) )  { return -1; }
old_excepts = fenv.__control & FE_ALL_EXCEPT;

fenv.__control &= ~new_excepts;
fenv.__mxcsr   &= ~(new_excepts << 7);

return ( fesetenv (&fenv) ? -1 : old_excepts );
}


static inline int fedisableexcept (unsigned int excepts)
{
static fenv_t fenv;
unsigned int new_excepts = excepts & FE_ALL_EXCEPT,
old_excepts;  

if ( fegetenv (&fenv) )  { return -1; }
old_excepts = fenv.__control & FE_ALL_EXCEPT;

fenv.__control |= new_excepts;
fenv.__mxcsr   |= new_excepts << 7;

return ( fesetenv (&fenv) ? -1 : old_excepts );
}

#     endif  

static struct sigaction fpe_termaction, fpe_oldaction;


static void TerminationSignalHandler(int sig, siginfo_t* sinfo, void* )
{
std::cerr << "ERROR: " << sig;
std::string message = "Floating-point exception (FPE).";

if (sinfo) {
switch (sinfo->si_code) {
case FPE_INTDIV:
message = "Integer divide by zero.";
break;
case FPE_INTOVF:
message = "Integer overflow.";
break;
case FPE_FLTDIV:
message = "Floating point divide by zero.";
break;
case FPE_FLTOVF:
message = "Floating point overflow.";
break;
case FPE_FLTUND:
message = "Floating point underflow.";
break;
case FPE_FLTRES:
message = "Floating point inexact result.";
break;
case FPE_FLTINV:
message = "Floating point invalid operation.";
break;
case FPE_FLTSUB:
message = "Subscript out of range.";
break;
default:
message = "Unknown error.";
break;
}
}

std::cerr << " - " << message << std::endl;

::abort();
}


static bool EnableInvalidOperationDetection(fpe_settings::fpe_set operations
= fpe_settings::fpe_set())
{
static bool init_first = true;
if(init_first)
init_first = false;
else
return false;

typedef fpe_settings::fpe_set::const_iterator const_iterator;
if(operations.empty())
operations = fpe_settings::enabled();
for(const_iterator itr = operations.begin(); itr != operations.end(); ++itr)
{
switch(*itr)
{
case fpe::divbyzero:
feenableexcept( FE_DIVBYZERO   );  
break;
case fpe::downward:
feenableexcept( FE_DOWNWARD    );
break;
case fpe::inexact:
feenableexcept( FE_INEXACT     );
break;
case fpe::invalid:
feenableexcept( FE_INVALID     );  
break;
case fpe::overflow:
feenableexcept( FE_OVERFLOW    );
break;
case fpe::tonearest:
feenableexcept( FE_TONEAREST   );
break;
case fpe::towardzero:
feenableexcept( FE_TOWARDZERO  );
break;
case fpe::underflow:
feenableexcept( FE_UNDERFLOW   );
break;
case fpe::upward:
feenableexcept( FE_UPWARD      );
break;
}
}

sigfillset(&fpe_termaction.sa_mask);
sigdelset(&fpe_termaction.sa_mask, SIGFPE);
sigdelset(&fpe_termaction.sa_mask, SIGSEGV);
fpe_termaction.sa_sigaction = TerminationSignalHandler;
fpe_termaction.sa_flags = SA_SIGINFO;
sigaction(SIGFPE,  &fpe_termaction, &fpe_oldaction);
sigaction(SIGSEGV, &fpe_termaction, &fpe_oldaction);

return true;
}


static void DisableInvalidOperationDetection()
{
sigemptyset(&fpe_termaction.sa_mask);
fpe_termaction.sa_handler = SIG_DFL;
sigaction(SIGFPE,  &fpe_termaction, 0);
sigaction(SIGSEGV, &fpe_termaction, 0);
}


} 


#   else  


static bool EnableInvalidOperationDetection() { return false; }
static void DisableInvalidOperationDetection() { }


} 


#   endif 
# else  


namespace mad
{


static bool EnableInvalidOperationDetection() { return false; }
static void DisableInvalidOperationDetection() { }


} 


# endif
#else 


namespace mad
{


static bool EnableInvalidOperationDetection() { return false; }
static void DisableInvalidOperationDetection() { }


} 


#endif

#pragma GCC diagnostic pop

#endif 
