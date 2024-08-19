#pragma once

#include <component.hpp>
#include <types.hpp>

static const UID LegacyConfigComponent_UID = UID(0x24ef6216838f9ffc);
struct ILegacyConfigComponent : public IComponent
{
PROVIDE_UID(LegacyConfigComponent_UID);

virtual StringView getConfig(StringView legacyName) = 0;

virtual StringView getLegacy(StringView configName) = 0;
};
