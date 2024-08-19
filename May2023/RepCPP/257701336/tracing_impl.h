

#ifndef TENSORFLOW_CORE_PLATFORM_DEFAULT_TRACING_IMPL_H_
#define TENSORFLOW_CORE_PLATFORM_DEFAULT_TRACING_IMPL_H_



#include "tensorflow/core/platform/tracing.h"

#define TRACELITERAL(a) \
do {                  \
} while (0)
#define TRACESTRING(s) \
do {                 \
} while (0)
#define TRACEPRINTF(format, ...) \
do {                           \
} while (0)

namespace tensorflow {
namespace tracing {

inline bool EventCollector::IsEnabled() { return false; }

}  
}  

#endif  
