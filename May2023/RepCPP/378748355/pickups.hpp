#pragma once

#include <component.hpp>
#include <player.hpp>
#include <types.hpp>
#include <values.hpp>

typedef uint8_t PickupType;

struct IPickup : public IExtensible, public IEntity
{
virtual void setType(PickupType type, bool update = true) = 0;

virtual PickupType getType() const = 0;

virtual void setPositionNoUpdate(Vector3 position) = 0;

virtual void setModel(int id, bool update = true) = 0;

virtual int getModel() const = 0;

virtual bool isStreamedInForPlayer(const IPlayer& player) const = 0;

virtual void streamInForPlayer(IPlayer& player) = 0;

virtual void streamOutForPlayer(IPlayer& player) = 0;

virtual void setPickupHiddenForPlayer(IPlayer& player, bool hidden) = 0;

virtual bool isPickupHiddenForPlayer(IPlayer& player) const = 0;

virtual void setLegacyPlayer(IPlayer* player) = 0;

virtual IPlayer* getLegacyPlayer() const = 0;
};

struct PickupEventHandler
{
virtual void onPlayerPickUpPickup(IPlayer& player, IPickup& pickup) { }
};

static const UID PickupsComponent_UID = UID(0xcf304faa363dd971);
struct IPickupsComponent : public IPoolComponent<IPickup>
{
PROVIDE_UID(PickupsComponent_UID);

virtual IEventDispatcher<PickupEventHandler>& getEventDispatcher() = 0;

virtual IPickup* create(int modelId, PickupType type, Vector3 pos, uint32_t virtualWorld, bool isStatic) = 0;

virtual int toLegacyID(int real) const = 0;

virtual int fromLegacyID(int legacy) const = 0;

virtual void releaseLegacyID(int legacy) = 0;

virtual int reserveLegacyID() = 0;

virtual void setLegacyID(int legacy, int real) = 0;
};

static const UID PickupData_UID = UID(0x98376F4428D7B70B);
struct IPlayerPickupData : public IExtension
{
PROVIDE_EXT_UID(PickupData_UID);

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
