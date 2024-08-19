#pragma GCC diagnostic error "-Wlarger-than=65536"
int a[131072];	
int b[1024];	
#pragma GCC diagnostic ignored "-Wlarger-than=65536"
int c[131072];	
int d[1024];	
#pragma GCC diagnostic warning "-Wlarger-than=65536"
int e[131072];	
int f[1024];	
