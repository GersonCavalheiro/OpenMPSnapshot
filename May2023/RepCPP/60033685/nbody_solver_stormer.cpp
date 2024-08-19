#include "nbody_solver_stormer.h"
#include "summation.h"

nbody_solver_stormer::nbody_solver_stormer() : nbody_solver()
{
}

const char* nbody_solver_stormer::type_name() const
{
return "nbody_solver_stormer";
}

void nbody_solver_stormer::advise(nbcoord_t dt)
{
Q_UNUSED(dt);

}

