#pragma unknown1 
#define COMMA ,
#define FOO(x) x
#define BAR(x) _Pragma("unknown_before") x
#define BAZ(x) x _Pragma("unknown_after")
int _Pragma("unknown2") bar1; 
FOO(int _Pragma("unknown3") bar2); 
int BAR(bar3); 
BAR(int bar4); 
int BAZ(bar5); 
int BAZ(bar6;) 
FOO(int bar7; _Pragma("unknown4")) 
#pragma unknown5 
