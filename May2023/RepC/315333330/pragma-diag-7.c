unsigned long ok = 0UL;
#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wtraditional"
unsigned long bad = 1UL; 
#pragma GCC diagnostic pop
unsigned long ok_again = 2UL; 
#pragma GCC diagnostic ignored "-Wtraditional"
#define UNW_DEC_PROLOGUE(fmt, body, rlen, arg) \
do {									\
unw_rlen = rlen;							\
*(int *)arg = body;						\
printf("    %s:%s(rlen=%lu)\n",					\
fmt, (body ? "body" : "prologue"), (unsigned long)rlen);	\
} while (0)
