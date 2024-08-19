#include "ort_prive.h"
static int check_cancel_for(void)
{
ort_eecb_t *me = __MYCB;
if (me->parent == NULL)
return ( me->cancel_for_me );
else
if (me->parent != NULL)
return ( TEAMINFO(me)->cancel_for_active );
return 0;
}
static int check_cancel_parallel(void)
{
return ( TEAMINFO(__MYCB)->cancel_par_active );
}
static int check_cancel_sections(void)
{
ort_eecb_t *me = __MYCB;
if (me->parent != NULL)
return ( TEAMINFO(me)->cancel_sec_active );
else
return ( me->cancel_sec_me );
return 0;
}
static int check_cancel_taskgroup(void)
{
return ( __CURRTASK(__MYCB)->taskgroup != NULL &&
__CURRTASK(__MYCB)->taskgroup->is_canceled );
}
int ort_check_cancel(int type)
{
switch (type)
{
case 0:
return check_cancel_parallel();
case 1:
return check_cancel_taskgroup();
case 2:
return check_cancel_for();
case 3:
return check_cancel_sections();
}
return 0;
}
static void enable_cancel_parallel(void)
{
ort_eecb_t *me = __MYCB;
if (CANCEL_ENABLED())
if (me->parent != NULL)
TEAMINFO(me)->cancel_par_active = true;
}
static void enable_cancel_sections(void)
{
ort_eecb_t *me = __MYCB;
if (CANCEL_ENABLED())
{
if (me->parent != NULL)
TEAMINFO(me)->cancel_sec_active = true;
else
me->cancel_sec_me = true;
}
}
static void enable_cancel_for(void)
{
ort_eecb_t  *me = __MYCB;
if (CANCEL_ENABLED())
{
if (me->parent != NULL)
TEAMINFO(me)->cancel_for_active = true;
else
me->cancel_for_me = true;
}
}
static void enable_cancel_taskgroup(void)
{
ort_eecb_t  *me = __MYCB;
if (__CURRTASK(me)->taskgroup != NULL && CANCEL_ENABLED())
__CURRTASK(me)->taskgroup->is_canceled = true;
}
int ort_enable_cancel(int type)
{
ort_eecb_t *me = __MYCB;
switch (type)
{
case 0:
enable_cancel_parallel();
return check_cancel_parallel();
case 1:
enable_cancel_taskgroup();
return check_cancel_taskgroup();
case 2:
enable_cancel_for();
if (me->parent == NULL)
{
if (check_cancel_for())
{
me->cancel_for_me = 0;
return 1;
}
else
return 0;
}
else
return ( check_cancel_for() );
case 3:
if (me->parent == NULL)
{
enable_cancel_sections();
if (check_cancel_sections())
{
me->cancel_sec_me = 0;
return 1;
}
else
return 0;
}
else
{
enable_cancel_sections();
return ( check_cancel_sections() );
}
}
}
