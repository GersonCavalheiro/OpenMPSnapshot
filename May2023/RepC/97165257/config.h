#pragma once
#include "utils/macros.h"
#include "utils/settings.h"
#include "utils/file.h"
#include "functions/string.h"
#include "default_config.h"
#define SETTINGS_MAX_VAR_LEN 1024
#define CFGVARS(o) \
o(ZL_cstring, VLC_EXE) \
o(ZL_uint,   BITRATE) \
o(ZL_uint,   RESOLUTION_H) \
o(ZL_uint,   FPS) \
o(ZL_uint,   MAX_PING) \
o(ZL_uint,   REMOTE_CONTROL_PORT) \
o(ZL_cstring, REMOTE_PASSWORD) \
o(ZL_uint,   LOCAL_CONTROL_PORT) \
o(ZL_uint,   VIDEO_PORT)
CPPVARS_TOSTRUCT(CFGVARS) PROGRAM_CONFIG;
void Configure();
