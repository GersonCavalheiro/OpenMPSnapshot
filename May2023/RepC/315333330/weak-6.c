extern void * foo (void);
void * foo (void) { return (void *)foo; }
#pragma weak foo
