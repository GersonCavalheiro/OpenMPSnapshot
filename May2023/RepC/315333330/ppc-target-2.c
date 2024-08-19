double normal1 (double a, double b) { return __builtin_copysign (a, b); }
#pragma GCC push_options
#pragma GCC target ("cpu=power5")
double power5 (double a, double b) { return __builtin_copysign (a, b); }
#pragma GCC pop_options
#pragma GCC target ("cpu=power6")
double power6 (double a, double b) { return __builtin_copysign (a, b); }
#pragma GCC reset_options
#pragma GCC target ("cpu=power6x")
double power6x (double a, double b) { return __builtin_copysign (a, b); }
#pragma GCC reset_options
#pragma GCC target ("cpu=power7")
double power7 (double a, double b) { return __builtin_copysign (a, b); }
#pragma GCC reset_options
#pragma GCC target ("cpu=power7,no-vsx")
double power7n (double a, double b) { return __builtin_copysign (a, b); }
#pragma GCC reset_options
double normal2 (double a, double b) { return __builtin_copysign (a, b); }
