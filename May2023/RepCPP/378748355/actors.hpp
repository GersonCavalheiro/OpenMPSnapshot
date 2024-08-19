#pragma once

#include <anim.hpp>
#include <component.hpp>
#include <player.hpp>
#include <types.hpp>
#include <values.hpp>

struct ActorSpawnData
{
Vector3 position;
float facingAngle;
int skin;
};

struct IActor : public IExtensible, public IEntity
{
virtual void setSkin(int id) = 0;

virtual int getSkin() const = 0;

virtual void applyAnimation(const AnimationData& animation) = 0;

virtual const AnimationData& getAnimation() const = 0;

virtual void clearAnimations() = 0;

virtual void setHealth(float health) = 0;

virtual float getHealth() const = 0;

virtual void setInvulnerable(bool invuln) = 0;

virtual bool isInvulnerable() const = 0;

virtual bool isStreamedInForPlayer(const IPlayer& player) const = 0;

virtual void streamInForPlayer(IPlayer& player) = 0;

virtual void streamOutForPlayer(IPlayer& player) = 0;

virtual const ActorSpawnData& getSpawnData() = 0;
};

struct ActorEventHandler
{
virtual void onPlayerGiveDamageActor(IPlayer& player, IActor& actor, float amount, unsigned weapon, BodyPart part) { }
virtual void onActorStreamOut(IActor& actor, IPlayer& forPlayer) { }
virtual void onActorStreamIn(IActor& actor, IPlayer& forPlayer) { }
};

static const UID ActorsComponent_UID = UID(0xc81ca021eae2ad5c);
struct IActorsComponent : public IPoolComponent<IActor>
{
PROVIDE_UID(ActorsComponent_UID);

virtual IEventDispatcher<ActorEventHandler>& getEventDispatcher() = 0;

virtual IActor* create(int skin, Vector3 pos, float angle) = 0;
};
