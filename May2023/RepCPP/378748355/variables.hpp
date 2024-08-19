#pragma once

#include <component.hpp>
#include <player.hpp>
#include <types.hpp>
#include <values.hpp>

enum VariableType
{
VariableType_None,
VariableType_Int,
VariableType_String,
VariableType_Float
};

struct IVariableStorageBase
{
virtual void setString(StringView key, StringView value) = 0;

virtual const StringView getString(StringView key) const = 0;

virtual void setInt(StringView key, int value) = 0;

virtual int getInt(StringView key) const = 0;

virtual void setFloat(StringView key, float value) = 0;

virtual float getFloat(StringView key) const = 0;

virtual VariableType getType(StringView key) const = 0;

virtual bool erase(StringView key) = 0;

virtual bool getKeyAtIndex(int index, StringView& key) const = 0;

virtual int size() const = 0;
};

static const UID VariablesComponent_UID = UID(0x75e121848bc01fa2);
struct IVariablesComponent : public IComponent, public IVariableStorageBase
{
PROVIDE_UID(VariablesComponent_UID);
};

static const UID PlayerVariableData_UID = UID(0x12debbc8a3bd23ad);
struct IPlayerVariableData : public IExtension, public IVariableStorageBase
{
PROVIDE_EXT_UID(PlayerVariableData_UID);
};
