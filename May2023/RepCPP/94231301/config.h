
#pragma once

#ifndef DEFAULT_WIDTH
#  define DEFAULT_WIDTH 8
#endif

#ifdef _WIN32
#  define PSIMD_ALIGN(...) __declspec(align(__VA_ARGS__))
#else
#  define PSIMD_ALIGN(...) __attribute__((aligned(__VA_ARGS__)))
#endif