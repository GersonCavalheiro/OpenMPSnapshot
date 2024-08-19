#if 1
#define foo bar		
# define foo bar	
#define foo bar	
# define foo bar	
#pragma whatever	
#pragma whatever	
#pragma whatever	
#pragma whatever	
#assert foo(bar)	
# assert bar(baz)	
#assert baz(quux)	
# assert quux(weeble)	
#else
#undef foo bar		
# undef foo bar		
#undef foo bar		
# undef foo bar	
#error whatever		
# error whatever	
#error whatever	
# error whatever	
#unassert foo(bar)		
# unassert bar(baz)		
#unassert baz(quux)		
# unassert quux(weeble)	
#endif
