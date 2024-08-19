#define X  1
# undef X
#pragma pop_macro("X")
#ifdef X
#error X is defined
#endif
