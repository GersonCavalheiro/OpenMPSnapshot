
#ifndef RV_RESOLVERS_H
#define RV_RESOLVERS_H

#include "rv/config.h"
#include "rv/PlatformInfo.h"

namespace rv {
void addTLIResolver(const Config & config, PlatformInfo & platInfo);

void addSleefResolver(const Config & config, PlatformInfo & platInfo);

void addOpenMPResolver(const Config & config, PlatformInfo & platInfo);

void addRecursiveResolver(const Config & config, PlatformInfo & platInfo);
}

#endif
