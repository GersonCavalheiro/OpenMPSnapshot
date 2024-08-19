#pragma GCC warning "warn-a" 
#pragma GCC error "err-b" 
#define CONST1 _Pragma("GCC warning \"warn-c\"") 1
#define CONST2 _Pragma("GCC error \"err-d\"") 2
char a[CONST1]; 
char b[CONST2]; 
