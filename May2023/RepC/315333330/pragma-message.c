#pragma message                  
#pragma message 0                
#pragma message id               
#pragma message (                
#pragma message (0               
#pragma message (id              
#pragma message ()               
#pragma message (0)              
#pragma message (id)             
#pragma message "
#pragma message "Bad 1
#pragma message ("Bad 2
#pragma message ("Bad 3"
#pragma message "" junk
#pragma message ("") junk
#pragma message ""               
#pragma message ("")
#pragma message "Okay 1"         
#pragma message ("Okay 2")       
#define THREE "3"
#pragma message ("Okay " THREE)  
#define DO_PRAGMA(x) _Pragma (#x)
#define TODO(x) DO_PRAGMA(message ("TODO - " #x))
TODO(Okay 4)                     
#if 0
#pragma message ("Not printed")
#endif
int unused;  
