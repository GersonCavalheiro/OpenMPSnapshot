

#pragma once

#include <sdk.hpp>

static const UID FixesData_UID = UID(0x672d5d6fbb094ef7);
struct IPlayerFixesData : public IExtension
{
PROVIDE_EXT_UID(FixesData_UID);

virtual bool sendGameText(StringView message, Milliseconds time, int style) = 0;

virtual bool hideGameText(int style) = 0;

virtual bool hasGameText(int style) = 0;

virtual bool getGameText(int style, StringView& message, Milliseconds& time, Milliseconds& remaining) = 0;

virtual void applyAnimation(IPlayer* player, IActor* actor, AnimationData const* animation) = 0;
};

static const UID FixesComponent_UID = UID(0xb5c615eff0329ff7);
struct IFixesComponent : public IComponent
{
PROVIDE_UID(FixesComponent_UID);

virtual bool sendGameTextToAll(StringView message, Milliseconds time, int style) = 0;

virtual bool hideGameTextForAll(int style) = 0;

virtual void clearAnimation(IPlayer* player, IActor* actor) = 0;
};
