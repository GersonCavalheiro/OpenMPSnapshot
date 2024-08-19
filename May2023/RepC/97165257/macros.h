#pragma once
#include <stdio.h>
#include <Windows.h>
#define CRLF "\r\n"
typedef const char* ZL_cstring;
typedef unsigned long ZL_ulong;
typedef unsigned int ZL_uint;
typedef unsigned short ZL_ushort;
#define ARGCNV int argc, ZL_cstring* argv
#define argcnv argc, argv
#define ITOA_S(in) ITOA_SI(in)
#define ITOA_SI(in) #in
#define ITOA_R(in, out) char out[11]; ITOA_RI(in, out)
#define ITOA_RI(in, tmp) itoa(in, tmp, 10)
#define SFREE(ptr) do { if(ptr) free(ptr); } while (0)
#ifdef MESSAGEBOX_GUI
#define PRINT(msg) MessageBox(NULL, msg, "Info", MB_OK)
#else
#define PRINT(msg) printf("%s", msg)
#endif
#define LFATAL(st, nc) do { PRINT(st); exit(nc); } while(0)
#ifdef DEBUG
#define LDEBUG printf
#else
#define LDEBUG(a, ...) do { } while(0)
#endif
#define SWAP_UINT16(x) (((x) >> 8) | ((x) << 8))
#define SWAP_UINT32(x) (((x) >> 24) | (((x) & 0x00FF0000) >> 8) | (((x) & 0x0000FF00) << 8) | ((x) << 24))
