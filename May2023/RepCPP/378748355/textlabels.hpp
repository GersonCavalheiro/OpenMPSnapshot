#pragma once

#include <component.hpp>
#include <player.hpp>
#include <types.hpp>
#include <values.hpp>

struct IVehicle;

struct TextLabelAttachmentData
{
int playerID = INVALID_PLAYER_ID;
int vehicleID = INVALID_VEHICLE_ID;
};

struct ITextLabelBase : public IExtensible, public IEntity
{
virtual void setText(StringView text) = 0;

virtual StringView getText() const = 0;

virtual void setColour(Colour colour) = 0;

virtual Colour getColour() const = 0;

virtual void setDrawDistance(float dist) = 0;

virtual float getDrawDistance() = 0;

virtual void attachToPlayer(IPlayer& player, Vector3 offset) = 0;

virtual void attachToVehicle(IVehicle& vehicle, Vector3 offset) = 0;

virtual const TextLabelAttachmentData& getAttachmentData() const = 0;

virtual void detachFromPlayer(Vector3 position) = 0;

virtual void detachFromVehicle(Vector3 position) = 0;

virtual void setTestLOS(bool status) = 0;

virtual bool getTestLOS() const = 0;

virtual void setColourAndText(Colour colour, StringView text) = 0;
};

struct ITextLabel : public ITextLabelBase
{
virtual bool isStreamedInForPlayer(const IPlayer& player) const = 0;

virtual void streamInForPlayer(IPlayer& player) = 0;

virtual void streamOutForPlayer(IPlayer& player) = 0;
};

struct IPlayerTextLabel : public ITextLabelBase
{
};

static const UID TextLabelsComponent_UID = UID(0xa0c57ea80a009742);
struct ITextLabelsComponent : public IPoolComponent<ITextLabel>
{
PROVIDE_UID(TextLabelsComponent_UID);

virtual ITextLabel* create(StringView text, Colour colour, Vector3 pos, float drawDist, int vw, bool los) = 0;

virtual ITextLabel* create(StringView text, Colour colour, Vector3 pos, float drawDist, int vw, bool los, IPlayer& attach) = 0;

virtual ITextLabel* create(StringView text, Colour colour, Vector3 pos, float drawDist, int vw, bool los, IVehicle& attach) = 0;
};

static const UID PlayerTextLabelData_UID = UID(0xb9e2bd0dc5148c3c);
struct IPlayerTextLabelData : public IExtension, public IPool<IPlayerTextLabel>
{
PROVIDE_EXT_UID(PlayerTextLabelData_UID);

virtual IPlayerTextLabel* create(StringView text, Colour colour, Vector3 pos, float drawDist, bool los) = 0;

virtual IPlayerTextLabel* create(StringView text, Colour colour, Vector3 pos, float drawDist, bool los, IPlayer& attach) = 0;

virtual IPlayerTextLabel* create(StringView text, Colour colour, Vector3 pos, float drawDist, bool los, IVehicle& attach) = 0;
};
