#pragma once
#include <process.h>
#include "../utils/networking.h"
#include "../utils/macros.h"
#include "../config.h"
void PingTheReceiverAllNetworks();
void PingTheReceiverSingleNetwork(ZL_cstring start, ZL_ulong length);
void PingTheReceiver(ARGCNV);
void RunProcess();
