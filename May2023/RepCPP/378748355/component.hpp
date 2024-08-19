#pragma once

#include "types.hpp"

#ifndef BUILD_NUMBER
#define BUILD_NUMBER 0
#endif

#define OMP_VERSION_SUPPORTED 1

#define PROVIDE_EXT_UID(uuid)                 \
static constexpr UID ExtensionIID = uuid; \
UID getExtensionID() override { return ExtensionIID; }

struct IExtension
{
virtual UID getExtensionID() = 0;

virtual void freeExtension() { }

virtual void reset() = 0;
};

struct IExtensible
{
virtual IExtension* getExtension(UID id) { return nullptr; }

template <class ExtensionT>
ExtensionT* _queryExtension()
{
static_assert(std::is_base_of<IExtension, ExtensionT>::value, "queryExtension parameter must inherit from IExtension");

auto it = miscExtensions.find(ExtensionT::ExtensionIID);
if (it != miscExtensions.end())
{
return static_cast<ExtensionT*>(it->second.first);
}

IExtension* ext = getExtension(ExtensionT::ExtensionIID);
if (ext)
{
return static_cast<ExtensionT*>(ext);
}
return nullptr;
}

virtual bool addExtension(IExtension* ext, bool autoDeleteExt)
{
return miscExtensions.emplace(robin_hood::pair<UID, Pair<IExtension*, bool>>(ext->getExtensionID(), std::make_pair(ext, autoDeleteExt))).second;
}

virtual bool removeExtension(IExtension* ext)
{
auto it = miscExtensions.find(ext->getExtensionID());
if (it == miscExtensions.end())
{
return false;
}
if (it->second.second)
{
it->second.first->freeExtension();
}
miscExtensions.erase(it);
return true;
}

virtual bool removeExtension(UID id)
{
auto it = miscExtensions.find(id);
if (it == miscExtensions.end())
{
return false;
}
if (it->second.second)
{
it->second.first->freeExtension();
}
miscExtensions.erase(it);
return true;
}

virtual ~IExtensible()
{
freeExtensions();
}

protected:
FlatHashMap<UID, Pair<IExtension*, bool>> miscExtensions;

void freeExtensions()
{
for (auto it = miscExtensions.begin(); it != miscExtensions.end(); ++it)
{
if (it->second.second)
{
it->second.first->freeExtension();
}
}
}

void resetExtensions()
{
for (auto it = miscExtensions.begin(); it != miscExtensions.end(); ++it)
{
if (it->second.second)
{
it->second.first->reset();
}
}
}
};

template <class ExtensionT>
ExtensionT* queryExtension(IExtensible* extensible)
{
return extensible->_queryExtension<ExtensionT>();
}

template <class ExtensionT>
ExtensionT* queryExtension(const IExtensible* extensible)
{
return extensible->_queryExtension<ExtensionT>();
}

template <class ExtensionT>
ExtensionT* queryExtension(IExtensible& extensible)
{
return extensible._queryExtension<ExtensionT>();
}

template <class ExtensionT>
ExtensionT* queryExtension(const IExtensible& extensible)
{
return extensible._queryExtension<ExtensionT>();
}

#define PROVIDE_UID(uuid)            \
static constexpr UID IID = uuid; \
UID getUID() override { return uuid; }

struct IUIDProvider
{
virtual UID getUID() = 0;
};

enum ComponentType
{
Other,
Network,
Pool,
};

struct ICore;
struct IComponentList;
struct ILogger;
struct IEarlyConfig;

struct IComponent : public IExtensible, public IUIDProvider
{
virtual int supportedVersion() const final
{
return OMP_VERSION_SUPPORTED;
}

virtual StringView componentName() const = 0;

virtual ComponentType componentType() const { return ComponentType::Other; }

virtual SemanticVersion componentVersion() const = 0;

virtual void onLoad(ICore* c) = 0;

virtual void onInit(IComponentList* components) { }

virtual void onReady() { }

virtual void onFree(IComponent* component) { }

virtual void provideConfiguration(ILogger& logger, IEarlyConfig& config, bool defaults) { }

virtual void free() = 0;

virtual void reset() = 0;
};

struct IComponentList : public IExtensible
{
virtual IComponent* queryComponent(UID id) = 0;

template <class ComponentT>
ComponentT* queryComponent()
{
static_assert(std::is_base_of<IComponent, ComponentT>::value, "queryComponent parameter must inherit from IComponent");
return static_cast<ComponentT*>(queryComponent(ComponentT::IID));
}
};
