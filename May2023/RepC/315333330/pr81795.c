#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wc++-compat"
struct S { int f; };	
typedef int S;		
typedef int T;		
struct T { int f; };    
#pragma GCC diagnostic pop
