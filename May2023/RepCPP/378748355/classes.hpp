#pragma once

#include <sdk.hpp>

struct PlayerClass
{
int team; 
int skin; 
Vector3 spawn; 
float angle; 
WeaponSlots weapons; 

PlayerClass(int skin, int team, Vector3 spawn, float angle, const WeaponSlots& weapons)
: team(team)
, skin(skin)
, spawn(spawn)
, angle(angle)
, weapons(weapons)
{
}
};

struct IClass : public IExtensible, public IIDProvider
{
virtual const PlayerClass& getClass() = 0;

virtual void setClass(const PlayerClass& data) = 0;
};

static const UID PlayerClassData_UID = UID(0x185655ded843788b);
struct IPlayerClassData : public IExtension
{
PROVIDE_EXT_UID(PlayerClassData_UID)

virtual const PlayerClass& getClass() = 0;
virtual void setSpawnInfo(const PlayerClass& info) = 0;
virtual void spawnPlayer() = 0;
};

struct ClassEventHandler
{
virtual bool onPlayerRequestClass(IPlayer& player, unsigned int classId) { return true; }
};

static const UID ClassesComponent_UID = UID(0x8cfb3183976da208);
struct IClassesComponent : public IPoolComponent<IClass>
{
PROVIDE_UID(ClassesComponent_UID)

virtual IEventDispatcher<ClassEventHandler>& getEventDispatcher() = 0;

virtual IClass* create(int skin, int team, Vector3 spawn, float angle, const WeaponSlots& weapons) = 0;
};
