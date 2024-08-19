#pragma once

#include "component.hpp"
#include "events.hpp"
#include "gtaquat.hpp"
#include "types.hpp"
#include <set>

struct IIDProvider
{
virtual int getID() const = 0;
};

struct IEntity : public IIDProvider
{
virtual Vector3 getPosition() const = 0;

virtual void setPosition(Vector3 position) = 0;

virtual GTAQuat getRotation() const = 0;

virtual void setRotation(GTAQuat rotation) = 0;

virtual int getVirtualWorld() const = 0;

virtual void setVirtualWorld(int vw) = 0;
};
