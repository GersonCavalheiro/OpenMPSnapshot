#pragma GCC poison foo
foo			
#pragma GCC poison foo2 foo3
foo2			
foo3			
#pragma GCC poison	foo4 	foo5
foo4			
foo5			
#pragma GCC poison +++	
#define foo6 123
#pragma GCC poison foo6	
#define foo6 345	
#define foo6 456	
#ifdef foo6		
#error hey! foo6 defined!
#endif
#if defined(foo6)	
#error foo6 still defined!
#else
foo6			
#endif
#pragma GCC poison
