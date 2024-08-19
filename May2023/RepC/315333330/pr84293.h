#pragma GCC system_header
struct typeobject { unsigned refs; };
typedef struct object { unsigned refs; } Object;
#define INCREF_TDEF(op) (((Object*)(op))->refs++)
#define INCREF_STAG(op) (((struct object*)(op))->refs++)
