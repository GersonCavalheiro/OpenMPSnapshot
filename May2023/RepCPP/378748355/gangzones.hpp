#pragma once

#include <anim.hpp>
#include <component.hpp>
#include <player.hpp>
#include <types.hpp>
#include <values.hpp>

struct GangZonePos
{
Vector2 min = { 0.0f, 0.0f };
Vector2 max = { 0.0f, 0.0f };
};

struct IGangZone : public IExtensible, public IIDProvider
{
virtual bool isShownForPlayer(const IPlayer& player) const = 0;

virtual bool isFlashingForPlayer(const IPlayer& player) const = 0;

virtual void showForPlayer(IPlayer& player, const Colour& colour) = 0;

virtual void hideForPlayer(IPlayer& player) = 0;

virtual void flashForPlayer(IPlayer& player, const Colour& colour) = 0;

virtual void stopFlashForPlayer(IPlayer& player) = 0;

virtual GangZonePos getPosition() const = 0;

virtual void setPosition(const GangZonePos& position) = 0;

virtual bool isPlayerInside(const IPlayer& player) const = 0;

virtual const FlatHashSet<IPlayer*>& getShownFor() = 0;

virtual const Colour getFlashingColourForPlayer(IPlayer& player) const = 0;

virtual const Colour getColourForPlayer(IPlayer& player) const = 0;

virtual void setLegacyPlayer(IPlayer* player) = 0;

virtual IPlayer* getLegacyPlayer() const = 0;
};

struct GangZoneEventHandler
{
virtual void onPlayerEnterGangZone(IPlayer& player, IGangZone& zone) { }
virtual void onPlayerLeaveGangZone(IPlayer& player, IGangZone& zone) { }
virtual void onPlayerClickGangZone(IPlayer& player, IGangZone& zone) { }
};

static const UID GangZoneComponent_UID = UID(0xb3351d11ee8d8056);

struct IGangZonesComponent : public IPoolComponent<IGangZone>
{
PROVIDE_UID(GangZoneComponent_UID);

virtual IEventDispatcher<GangZoneEventHandler>& getEventDispatcher() = 0;

virtual IGangZone* create(GangZonePos pos) = 0;

virtual const FlatHashSet<IGangZone*>& getCheckingGangZones() const = 0;

virtual void useGangZoneCheck(IGangZone& zone, bool enable) = 0;

virtual int toLegacyID(int real) const = 0;

virtual int fromLegacyID(int legacy) const = 0;

virtual void releaseLegacyID(int legacy) = 0;

virtual int reserveLegacyID() = 0;

virtual void setLegacyID(int legacy, int real) = 0;
};

static const UID GangZoneData_UID = UID(0xee8d8056b3351d11);
struct IPlayerGangZoneData : public IExtension
{
PROVIDE_EXT_UID(GangZoneData_UID);

virtual int toLegacyID(int real) const = 0;

virtual int fromLegacyID(int legacy) const = 0;

virtual void releaseLegacyID(int legacy) = 0;

virtual int reserveLegacyID() = 0;

virtual void setLegacyID(int legacy, int real) = 0;

virtual int toClientID(int real) const = 0;

virtual int fromClientID(int legacy) const = 0;

virtual void releaseClientID(int legacy) = 0;

virtual int reserveClientID() = 0;

virtual void setClientID(int legacy, int real) = 0;
};
