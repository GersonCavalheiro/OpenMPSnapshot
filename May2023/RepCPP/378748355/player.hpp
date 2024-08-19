#pragma once

#include "anim.hpp"
#include "entity.hpp"
#include "network.hpp"
#include "pool.hpp"
#include "types.hpp"
#include "values.hpp"
#include <chrono>
#include <string>
#include <type_traits>
#include <utility>
struct IVehicle;
struct IObject;
struct IPlayerObject;
struct IActor;

enum PlayerFightingStyle
{
PlayerFightingStyle_Normal = 4,
PlayerFightingStyle_Boxing = 5,
PlayerFightingStyle_KungFu = 6,
PlayerFightingStyle_KneeHead = 7,
PlayerFightingStyle_GrabKick = 15,
PlayerFightingStyle_Elbow = 16
};

enum PlayerState
{
PlayerState_None = 0,
PlayerState_OnFoot = 1,
PlayerState_Driver = 2,
PlayerState_Passenger = 3,
PlayerState_ExitVehicle = 4,
PlayerState_EnterVehicleDriver = 5,
PlayerState_EnterVehiclePassenger = 6,
PlayerState_Wasted = 7,
PlayerState_Spawned = 8,
PlayerState_Spectating = 9
};

enum PlayerWeaponSkill
{
PlayerWeaponSkill_Pistol,
PlayerWeaponSkill_SilencedPistol,
PlayerWeaponSkill_DesertEagle,
PlayerWeaponSkill_Shotgun,
PlayerWeaponSkill_SawnOff,
PlayerWeaponSkill_SPAS12,
PlayerWeaponSkill_Uzi,
PlayerWeaponSkill_MP5,
PlayerWeaponSkill_AK47,
PlayerWeaponSkill_M4,
PlayerWeaponSkill_Sniper
};

enum PlayerSpecialAction
{
SpecialAction_None,
SpecialAction_Duck,
SpecialAction_Jetpack,
SpecialAction_EnterVehicle,
SpecialAction_ExitVehicle,
SpecialAction_Dance1,
SpecialAction_Dance2,
SpecialAction_Dance3,
SpecialAction_Dance4,
SpecialAction_HandsUp = 10,
SpecialAction_Cellphone,
SpecialAction_Sitting,
SpecialAction_StopCellphone,
SpecialAction_Beer = 20,
Specialaction_Smoke,
SpecialAction_Wine,
SpecialAction_Sprunk,
SpecialAction_Cuffed,
SpecialAction_Carry,
SpecialAction_Pissing = 68
};

enum PlayerAnimationSyncType
{
PlayerAnimationSyncType_NoSync,
PlayerAnimationSyncType_Sync,
PlayerAnimationSyncType_SyncOthers
};

enum PlayerBulletHitType : uint8_t
{
PlayerBulletHitType_None,
PlayerBulletHitType_Player = 1,
PlayerBulletHitType_Vehicle = 2,
PlayerBulletHitType_Object = 3,
PlayerBulletHitType_PlayerObject = 4,
};

enum BodyPart
{
BodyPart_Torso = 3,
BodyPart_Groin,
BodyPart_LeftArm,
BodyPart_RightArm,
BodyPart_LeftLeg,
BodyPart_RightLeg,
BodyPart_Head
};

enum MapIconStyle
{
MapIconStyle_Local,
MapIconStyle_Global,
MapIconStyle_LocalCheckpoint,
MapIconStyle_GlobalCheckpoint
};

enum PlayerClickSource
{
PlayerClickSource_Scoreboard
};

enum PlayerSpectateMode
{
PlayerSpectateMode_Normal = 1,
PlayerSpectateMode_Fixed,
PlayerSpectateMode_Side
};

enum PlayerCameraCutType
{
PlayerCameraCutType_Cut,
PlayerCameraCutType_Move
};

enum PlayerMarkerMode
{
PlayerMarkerMode_Off,
PlayerMarkerMode_Global,
PlayerMarkerMode_Streamed
};

enum LagCompMode
{
LagCompMode_Disabled = 0,
LagCompMode_PositionOnly = 2,
LagCompMode_Enabled = 1
};

enum PlayerWeapon
{
PlayerWeapon_Fist,
PlayerWeapon_BrassKnuckle,
PlayerWeapon_GolfClub,
PlayerWeapon_NiteStick,
PlayerWeapon_Knife,
PlayerWeapon_Bat,
PlayerWeapon_Shovel,
PlayerWeapon_PoolStick,
PlayerWeapon_Katana,
PlayerWeapon_Chainsaw,
PlayerWeapon_Dildo,
PlayerWeapon_Dildo2,
PlayerWeapon_Vibrator,
PlayerWeapon_Vibrator2,
PlayerWeapon_Flower,
PlayerWeapon_Cane,
PlayerWeapon_Grenade,
PlayerWeapon_Teargas,
PlayerWeapon_Moltov,
PlayerWeapon_Colt45 = 22,
PlayerWeapon_Silenced,
PlayerWeapon_Deagle,
PlayerWeapon_Shotgun,
PlayerWeapon_Sawedoff,
PlayerWeapon_Shotgspa,
PlayerWeapon_UZI,
PlayerWeapon_MP5,
PlayerWeapon_AK47,
PlayerWeapon_M4,
PlayerWeapon_TEC9,
PlayerWeapon_Rifle,
PlayerWeapon_Sniper,
PlayerWeapon_RocketLauncher,
PlayerWeapon_HeatSeeker,
PlayerWeapon_FlameThrower,
PlayerWeapon_Minigun,
PlayerWeapon_Satchel,
PlayerWeapon_Bomb,
PlayerWeapon_SprayCan,
PlayerWeapon_FireExtinguisher,
PlayerWeapon_Camera,
PlayerWeapon_Night_Vis_Goggles,
PlayerWeapon_Thermal_Goggles,
PlayerWeapon_Parachute,
PlayerWeapon_Vehicle = 49,
PlayerWeapon_Heliblades,
PlayerWeapon_Explosion,
PlayerWeapon_Drown = 53,
PlayerWeapon_Collision,
PlayerWeapon_End
};

static const StringView PlayerWeaponNames[] = {
"Fist",
"Brass Knuckles",
"Golf Club",
"Nite Stick",
"Knife",
"Baseball Bat",
"Shovel",
"Pool Cue",
"Katana",
"Chainsaw",
"Dildo",
"Dildo",
"Vibrator",
"Vibrator",
"Flowers",
"Cane",
"Grenade",
"Teargas",
"Molotov Cocktail", 
"Invalid",
"Invalid",
"Invalid",
"Colt 45", 
"Silenced Pistol",
"Desert Eagle",
"Shotgun",
"Sawn-off Shotgun",
"Combat Shotgun",
"UZI",
"MP5",
"AK47",
"M4",
"TEC9",
"Rifle",
"Sniper Rifle",
"Rocket Launcher",
"Heat Seaker",
"Flamethrower",
"Minigun",
"Satchel Explosives",
"Bomb",
"Spray Can",
"Fire Extinguisher",
"Camera",
"Night Vision Goggles",
"Thermal Goggles",
"Parachute", 
"Invalid",
"Invalid",
"Vehicle", 
"Helicopter Blades", 
"Explosion", 
"Invalid",
"Drowned", 
"Splat"
};

static const StringView BodyPartString[] = {
"invalid",
"invalid",
"invalid",
"torso",
"groin",
"left arm",
"right arm",
"left leg",
"right leg",
"head"
};

struct PlayerKeyData
{
uint32_t keys;
int16_t upDown;
int16_t leftRight;
};

struct PlayerAnimationData
{
uint16_t ID;
uint16_t flags;

inline Pair<StringView, StringView> name() const
{
return splitAnimationNames(ID);
}
};

struct PlayerSurfingData
{
enum class Type
{
None,
Vehicle,
Object,
PlayerObject
} type;
int ID;
Vector3 offset;
};

struct WeaponSlotData
{
uint8_t id;
uint32_t ammo;

WeaponSlotData()
: id(0)
, ammo(0)
{
}

WeaponSlotData(uint8_t id)
: id(id)
, ammo(0)
{
}

WeaponSlotData(uint8_t id, uint32_t ammo)
: id(id)
, ammo(ammo)
{
}

uint8_t slot()
{
static const uint8_t slots[] = { 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10, 8, 8, 8, INVALID_WEAPON_SLOT, INVALID_WEAPON_SLOT, INVALID_WEAPON_SLOT, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 4, 6, 6, 7, 7, 7, 7, 8, 12, 9, 9, 9, 11, 11, 11 };
if (id >= GLM_COUNTOF(slots))
{
return INVALID_WEAPON_SLOT;
}
return slots[id];
}

bool shootable()
{
return (id >= 22 && id <= 34) || id == 38;
}
};

typedef StaticArray<WeaponSlotData, MAX_WEAPON_SLOTS> WeaponSlots;

enum PlayerWeaponState : int8_t
{
PlayerWeaponState_Unknown = -1,
PlayerWeaponState_NoBullets,
PlayerWeaponState_LastBullet,
PlayerWeaponState_MoreBullets,
PlayerWeaponState_Reloading
};

struct PlayerAimData
{
Vector3 camFrontVector;
Vector3 camPos;
float aimZ;
float camZoom;
float aspectRatio;
PlayerWeaponState weaponState;
uint8_t camMode;
};

struct PlayerBulletData
{
Vector3 origin;
Vector3 hitPos;
Vector3 offset;
uint8_t weapon;
PlayerBulletHitType hitType;
uint16_t hitID;
};

struct PlayerSpectateData
{
enum ESpectateType
{
None,
Vehicle,
Player
};

bool spectating;
int spectateID;
ESpectateType type;
};

struct IPlayerPool;
struct IPlayer;

enum EPlayerNameStatus
{
Updated, 
Taken, 
Invalid 
};

struct IPlayer : public IExtensible, public IEntity
{
virtual void kick() = 0;

virtual void ban(StringView reason = StringView()) = 0;

virtual bool isBot() const = 0;

virtual const PeerNetworkData& getNetworkData() const = 0;

unsigned getPing() const
{
return getNetworkData().network->getPing(*this);
}

bool sendPacket(Span<uint8_t> data, int channel, bool dispatchEvents = true)
{
return getNetworkData().network->sendPacket(*this, data, channel, dispatchEvents);
}

bool sendRPC(int id, Span<uint8_t> data, int channel, bool dispatchEvents = true)
{
return getNetworkData().network->sendRPC(*this, id, data, channel, dispatchEvents);
}

virtual void broadcastRPCToStreamed(int id, Span<uint8_t> data, int channel, bool skipFrom = false) const = 0;

virtual void broadcastPacketToStreamed(Span<uint8_t> data, int channel, bool skipFrom = true) const = 0;

virtual void broadcastSyncPacket(Span<uint8_t> data, int channel) const = 0;

virtual void spawn() = 0;

virtual ClientVersion getClientVersion() const = 0;

virtual StringView getClientVersionName() const = 0;

virtual void setPositionFindZ(Vector3 pos) = 0;

virtual void setCameraPosition(Vector3 pos) = 0;

virtual Vector3 getCameraPosition() = 0;

virtual void setCameraLookAt(Vector3 pos, int cutType) = 0;

virtual Vector3 getCameraLookAt() = 0;

virtual void setCameraBehind() = 0;

virtual void interpolateCameraPosition(Vector3 from, Vector3 to, int time, PlayerCameraCutType cutType) = 0;

virtual void interpolateCameraLookAt(Vector3 from, Vector3 to, int time, PlayerCameraCutType cutType) = 0;

virtual void attachCameraToObject(IObject& object) = 0;

virtual void attachCameraToObject(IPlayerObject& object) = 0;

virtual EPlayerNameStatus setName(StringView name) = 0;

virtual StringView getName() const = 0;

virtual StringView getSerial() const = 0;

virtual void giveWeapon(WeaponSlotData weapon) = 0;

virtual void removeWeapon(uint8_t weapon) = 0;

virtual void setWeaponAmmo(WeaponSlotData data) = 0;

virtual const WeaponSlots& getWeapons() const = 0;

virtual WeaponSlotData getWeaponSlot(int slot) = 0;

virtual void resetWeapons() = 0;

virtual void setArmedWeapon(uint32_t weapon) = 0;

virtual uint32_t getArmedWeapon() const = 0;

virtual uint32_t getArmedWeaponAmmo() const = 0;

virtual void setShopName(StringView name) = 0;

virtual StringView getShopName() const = 0;

virtual void setDrunkLevel(int level) = 0;

virtual int getDrunkLevel() const = 0;

virtual void setColour(Colour colour) = 0;

virtual const Colour& getColour() const = 0;

virtual void setOtherColour(IPlayer& other, Colour colour) = 0;

virtual bool getOtherColour(IPlayer& other, Colour& colour) const = 0;

virtual void setControllable(bool controllable) = 0;

virtual bool getControllable() const = 0;

virtual void setSpectating(bool spectating) = 0;

virtual void setWantedLevel(unsigned level) = 0;

virtual unsigned getWantedLevel() const = 0;

virtual void playSound(uint32_t sound, Vector3 pos) = 0;

virtual uint32_t lastPlayedSound() const = 0;

virtual void playAudio(StringView url, bool usePos = false, Vector3 pos = Vector3(0.f), float distance = 0.f) = 0;

virtual bool playerCrimeReport(IPlayer& suspect, int crime) = 0;

virtual void stopAudio() = 0;

virtual StringView lastPlayedAudio() const = 0;

virtual void createExplosion(Vector3 vec, int type, float radius) = 0;

virtual void sendDeathMessage(IPlayer& player, IPlayer* killer, int weapon) = 0;

virtual void sendEmptyDeathMessage() = 0;

virtual void removeDefaultObjects(unsigned model, Vector3 pos, float radius) = 0;

virtual void forceClassSelection() = 0;

virtual void setMoney(int money) = 0;

virtual void giveMoney(int money) = 0;

virtual void resetMoney() = 0;

virtual int getMoney() = 0;

virtual void setMapIcon(int id, Vector3 pos, int type, Colour colour, MapIconStyle style) = 0;

virtual void unsetMapIcon(int id) = 0;

virtual void useStuntBonuses(bool enable) = 0;

virtual void toggleOtherNameTag(IPlayer& other, bool toggle) = 0;

virtual void setTime(Hours hr, Minutes min) = 0;

virtual Pair<Hours, Minutes> getTime() const = 0;

virtual void useClock(bool enable) = 0;

virtual bool hasClock() const = 0;

virtual void useWidescreen(bool enable) = 0;

virtual bool hasWidescreen() const = 0;

virtual void setTransform(GTAQuat tm) = 0;

virtual void setHealth(float health) = 0;

virtual float getHealth() const = 0;

virtual void setScore(int score) = 0;

virtual int getScore() const = 0;

virtual void setArmour(float armour) = 0;

virtual float getArmour() const = 0;

virtual void setGravity(float gravity) = 0;

virtual float getGravity() const = 0;

virtual void setWorldTime(Hours time) = 0;

virtual void applyAnimation(const AnimationData& animation, PlayerAnimationSyncType syncType) = 0;

virtual void clearAnimations(PlayerAnimationSyncType syncType) = 0;

virtual PlayerAnimationData getAnimationData() const = 0;

virtual PlayerSurfingData getSurfingData() const = 0;

virtual void streamInForPlayer(IPlayer& other) = 0;

virtual bool isStreamedInForPlayer(const IPlayer& other) const = 0;

virtual void streamOutForPlayer(IPlayer& other) = 0;

virtual const FlatPtrHashSet<IPlayer>& streamedForPlayers() const = 0;

virtual PlayerState getState() const = 0;

virtual void setTeam(int team) = 0;

virtual int getTeam() const = 0;

virtual void setSkin(int skin, bool send = true) = 0;

virtual int getSkin() const = 0;

virtual void setChatBubble(StringView text, const Colour& colour, float drawDist, Milliseconds expire) = 0;

virtual void sendClientMessage(const Colour& colour, StringView message) = 0;

virtual void sendChatMessage(IPlayer& sender, StringView message) = 0;

virtual void sendCommand(StringView message) = 0;

virtual void sendGameText(StringView message, Milliseconds time, int style) = 0;

virtual void hideGameText(int style) = 0;

virtual bool hasGameText(int style) = 0;

virtual bool getGameText(int style, StringView& message, Milliseconds& time, Milliseconds& remaining) = 0;

virtual void setWeather(int weatherID) = 0;

virtual int getWeather() const = 0;

virtual void setWorldBounds(Vector4 coords) = 0;

virtual Vector4 getWorldBounds() const = 0;

virtual void setFightingStyle(PlayerFightingStyle style) = 0;

virtual PlayerFightingStyle getFightingStyle() const = 0;

virtual void setSkillLevel(PlayerWeaponSkill skill, int level) = 0;

virtual void setAction(PlayerSpecialAction action) = 0;

virtual PlayerSpecialAction getAction() const = 0;

virtual void setVelocity(Vector3 velocity) = 0;

virtual Vector3 getVelocity() const = 0;

virtual void setInterior(unsigned interior) = 0;

virtual unsigned getInterior() const = 0;

virtual const PlayerKeyData& getKeyData() const = 0;

virtual const StaticArray<uint16_t, NUM_SKILL_LEVELS>& getSkillLevels() const = 0;

virtual const PlayerAimData& getAimData() const = 0;

virtual const PlayerBulletData& getBulletData() const = 0;

virtual void useCameraTargeting(bool enable) = 0;

virtual bool hasCameraTargeting() const = 0;

virtual void removeFromVehicle(bool force) = 0;

virtual IPlayer* getCameraTargetPlayer() = 0;

virtual IVehicle* getCameraTargetVehicle() = 0;

virtual IObject* getCameraTargetObject() = 0;

virtual IActor* getCameraTargetActor() = 0;

virtual IPlayer* getTargetPlayer() = 0;

virtual IActor* getTargetActor() = 0;

virtual void setRemoteVehicleCollisions(bool collide) = 0;

virtual void spectatePlayer(IPlayer& target, PlayerSpectateMode mode) = 0;

virtual void spectateVehicle(IVehicle& target, PlayerSpectateMode mode) = 0;

virtual const PlayerSpectateData& getSpectateData() const = 0;

virtual void sendClientCheck(int actionType, int address, int offset, int count) = 0;

virtual void toggleGhostMode(bool toggle) = 0;

virtual bool isGhostModeEnabled() const = 0;

virtual int getDefaultObjectsRemoved() const = 0;

virtual bool getKickStatus() const = 0;

virtual void clearTasks(PlayerAnimationSyncType syncType) = 0;

virtual void allowWeapons(bool allow) = 0;

virtual bool areWeaponsAllowed() const = 0;

virtual void allowTeleport(bool allow) = 0;

virtual bool isTeleportAllowed() const = 0;

virtual bool isUsingOfficialClient() const = 0;
};

struct PlayerSpawnEventHandler
{
virtual bool onPlayerRequestSpawn(IPlayer& player) { return true; }
virtual void onPlayerSpawn(IPlayer& player) { }
};

struct PlayerConnectEventHandler
{
virtual void onIncomingConnection(IPlayer& player, StringView ipAddress, unsigned short port) { }
virtual void onPlayerConnect(IPlayer& player) { }
virtual void onPlayerDisconnect(IPlayer& player, PeerDisconnectReason reason) { }
virtual void onPlayerClientInit(IPlayer& player) { }
};

struct PlayerStreamEventHandler
{
virtual void onPlayerStreamIn(IPlayer& player, IPlayer& forPlayer) { }
virtual void onPlayerStreamOut(IPlayer& player, IPlayer& forPlayer) { }
};

struct PlayerTextEventHandler
{
virtual bool onPlayerText(IPlayer& player, StringView message) { return true; }
virtual bool onPlayerCommandText(IPlayer& player, StringView message) { return false; }
};

struct PlayerShotEventHandler
{
virtual bool onPlayerShotMissed(IPlayer& player, const PlayerBulletData& bulletData) { return true; }
virtual bool onPlayerShotPlayer(IPlayer& player, IPlayer& target, const PlayerBulletData& bulletData) { return true; }
virtual bool onPlayerShotVehicle(IPlayer& player, IVehicle& target, const PlayerBulletData& bulletData) { return true; }
virtual bool onPlayerShotObject(IPlayer& player, IObject& target, const PlayerBulletData& bulletData) { return true; }
virtual bool onPlayerShotPlayerObject(IPlayer& player, IPlayerObject& target, const PlayerBulletData& bulletData) { return true; }
};

struct PlayerChangeEventHandler
{
virtual void onPlayerScoreChange(IPlayer& player, int score) { }
virtual void onPlayerNameChange(IPlayer& player, StringView oldName) { }
virtual void onPlayerInteriorChange(IPlayer& player, unsigned newInterior, unsigned oldInterior) { }
virtual void onPlayerStateChange(IPlayer& player, PlayerState newState, PlayerState oldState) { }
virtual void onPlayerKeyStateChange(IPlayer& player, uint32_t newKeys, uint32_t oldKeys) { }
};

struct PlayerDamageEventHandler
{
virtual void onPlayerDeath(IPlayer& player, IPlayer* killer, int reason) { }
virtual void onPlayerTakeDamage(IPlayer& player, IPlayer* from, float amount, unsigned weapon, BodyPart part) { }
virtual void onPlayerGiveDamage(IPlayer& player, IPlayer& to, float amount, unsigned weapon, BodyPart part) { }
};

struct PlayerClickEventHandler
{
virtual void onPlayerClickMap(IPlayer& player, Vector3 pos) { }
virtual void onPlayerClickPlayer(IPlayer& player, IPlayer& clicked, PlayerClickSource source) { }
};

struct PlayerCheckEventHandler
{
virtual void onClientCheckResponse(IPlayer& player, int actionType, int address, int results) { }
};

struct PlayerUpdateEventHandler
{
virtual bool onPlayerUpdate(IPlayer& player, TimePoint now) { return true; }
};

struct IPlayerPool : public IExtensible, public IReadOnlyPool<IPlayer>
{
virtual const FlatPtrHashSet<IPlayer>& entries() = 0;

virtual const FlatPtrHashSet<IPlayer>& players() = 0;

virtual const FlatPtrHashSet<IPlayer>& bots() = 0;

virtual IEventDispatcher<PlayerSpawnEventHandler>& getPlayerSpawnDispatcher() = 0;
virtual IEventDispatcher<PlayerConnectEventHandler>& getPlayerConnectDispatcher() = 0;
virtual IEventDispatcher<PlayerStreamEventHandler>& getPlayerStreamDispatcher() = 0;
virtual IEventDispatcher<PlayerTextEventHandler>& getPlayerTextDispatcher() = 0;
virtual IEventDispatcher<PlayerShotEventHandler>& getPlayerShotDispatcher() = 0;
virtual IEventDispatcher<PlayerChangeEventHandler>& getPlayerChangeDispatcher() = 0;
virtual IEventDispatcher<PlayerDamageEventHandler>& getPlayerDamageDispatcher() = 0;
virtual IEventDispatcher<PlayerClickEventHandler>& getPlayerClickDispatcher() = 0;
virtual IEventDispatcher<PlayerCheckEventHandler>& getPlayerCheckDispatcher() = 0;

virtual IEventDispatcher<PlayerUpdateEventHandler>& getPlayerUpdateDispatcher() = 0;

virtual IEventDispatcher<PoolEventHandler<IPlayer>>& getPoolEventDispatcher() = 0;

virtual bool isNameTaken(StringView name, const IPlayer* skip) = 0;

virtual void sendClientMessageToAll(const Colour& colour, StringView message) = 0;

virtual void sendChatMessageToAll(IPlayer& from, StringView message) = 0;

virtual void sendGameTextToAll(StringView message, Milliseconds time, int style) = 0;

virtual void hideGameTextForAll(int style) = 0;

virtual void sendDeathMessageToAll(IPlayer* killer, IPlayer& killee, int weapon) = 0;

virtual void sendEmptyDeathMessageToAll() = 0;

virtual void createExplosionForAll(Vector3 vec, int type, float radius) = 0;

virtual Pair<NewConnectionResult, IPlayer*> requestPlayer(const PeerNetworkData& netData, const PeerRequestParams& params) = 0;

virtual void broadcastPacket(Span<uint8_t> data, int channel, const IPlayer* skipFrom = nullptr, bool dispatchEvents = true) = 0;

virtual void broadcastRPC(int id, Span<uint8_t> data, int channel, const IPlayer* skipFrom = nullptr, bool dispatchEvents = true) = 0;

virtual bool isNameValid(StringView name) const = 0;

virtual void allowNickNameCharacter(char character, bool allow) = 0;

virtual bool isNickNameCharacterAllowed(char character) const = 0;
};
