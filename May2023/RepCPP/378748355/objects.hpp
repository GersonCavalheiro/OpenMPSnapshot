#pragma once

#include <sdk.hpp>

struct IVehicle;

enum ObjectMaterialSize
{
ObjectMaterialSize_32x32 = 10,
ObjectMaterialSize_64x32 = 20,
ObjectMaterialSize_64x64 = 30,
ObjectMaterialSize_128x32 = 40,
ObjectMaterialSize_128x64 = 50,
ObjectMaterialSize_128x128 = 60,
ObjectMaterialSize_256x32 = 70,
ObjectMaterialSize_256x64 = 80,
ObjectMaterialSize_256x128 = 90,
ObjectMaterialSize_256x256 = 100,
ObjectMaterialSize_512x64 = 110,
ObjectMaterialSize_512x128 = 120,
ObjectMaterialSize_512x256 = 130,
ObjectMaterialSize_512x512 = 140
};

enum ObjectMaterialTextAlign
{
ObjectMaterialTextAlign_Left,
ObjectMaterialTextAlign_Center,
ObjectMaterialTextAlign_Right
};

enum ObjectSelectType
{
ObjectSelectType_None,
ObjectSelectType_Global,
ObjectSelectType_Player
};

enum ObjectEditResponse
{
ObjectEditResponse_Cancel,
ObjectEditResponse_Final,
ObjectEditResponse_Update
};

enum PlayerBone
{
PlayerBone_None,
PlayerBone_Spine,
PlayerBone_Head,
PlayerBone_LeftUpperArm,
PlayerBone_RightUpperArm,
PlayerBone_LeftHand,
PlayerBone_RightHand,
PlayerBone_LeftThigh,
PlayerBone_RightThigh,
PlayerBone_LeftFoot,
PlayerBone_RightFoot,
PlayerBone_RightCalf,
PlayerBone_LeftCalf,
PlayerBone_LeftForearm,
PlayerBone_RightForearm,
PlayerBone_LeftShoulder,
PlayerBone_RightShoulder,
PlayerBone_Neck,
PlayerBone_Jaw
};



struct ObjectMaterialData
{
enum Type : uint8_t
{
None,
Default,
Text
};

union
{
int model; 
struct
{ 
uint8_t materialSize;
uint8_t fontSize;
uint8_t alignment;
bool bold;
};
};

union
{
Colour materialColour; 
Colour fontColour; 
};

Colour backgroundColour; 

HybridString<32> textOrTXD; 
HybridString<32> fontOrTexture; 

Type type; 
bool used; 

ObjectMaterialData()
: used(false)
{
}
};

struct ObjectAttachmentData
{
enum class Type : uint8_t
{
None,
Vehicle,
Object,
Player
} type;
bool syncRotation;
int ID;
Vector3 offset;
Vector3 rotation;
};

struct ObjectAttachmentSlotData
{
int model;
int bone;
Vector3 offset;
Vector3 rotation;
Vector3 scale;
Colour colour1;
Colour colour2;
};

struct ObjectMoveData
{
Vector3 targetPos;
Vector3 targetRot;
float speed;
};

struct IBaseObject : public IExtensible, public IEntity
{
virtual void setDrawDistance(float drawDistance) = 0;

virtual float getDrawDistance() const = 0;

virtual void setModel(int model) = 0;

virtual int getModel() const = 0;

virtual void setCameraCollision(bool collision) = 0;

virtual bool getCameraCollision() const = 0;

virtual void move(const ObjectMoveData& data) = 0;

virtual bool isMoving() const = 0;

virtual void stop() = 0;

virtual const ObjectMoveData& getMovingData() const = 0;

virtual void attachToVehicle(IVehicle& vehicle, Vector3 offset, Vector3 rotation) = 0;

virtual void resetAttachment() = 0;

virtual const ObjectAttachmentData& getAttachmentData() const = 0;

virtual bool getMaterialData(uint32_t materialIndex, const ObjectMaterialData*& out) const = 0;

virtual void setMaterial(uint32_t materialIndex, int model, StringView textureLibrary, StringView textureName, Colour colour) = 0;

virtual void setMaterialText(uint32_t materialIndex, StringView text, ObjectMaterialSize materialSize, StringView fontFace, int fontSize, bool bold, Colour fontColour, Colour backgroundColour, ObjectMaterialTextAlign align) = 0;
};

struct IObject : public IBaseObject
{
virtual void attachToPlayer(IPlayer& player, Vector3 offset, Vector3 rotation) = 0;

virtual void attachToObject(IObject& object, Vector3 offset, Vector3 rotation, bool syncRotation) = 0;
};

struct IPlayerObject : public IBaseObject
{
virtual void attachToObject(IPlayerObject& object, Vector3 offset, Vector3 rotation) = 0;

virtual void attachToPlayer(IPlayer& player, Vector3 offset, Vector3 rotation) = 0;
};

struct ObjectEventHandler;

static const UID ObjectsComponent_UID = UID(0x59f8415f72da6160);
struct IObjectsComponent : public IPoolComponent<IObject>
{
PROVIDE_UID(ObjectsComponent_UID)

virtual IEventDispatcher<ObjectEventHandler>& getEventDispatcher() = 0;

virtual void setDefaultCameraCollision(bool collision) = 0;

virtual bool getDefaultCameraCollision() const = 0;

virtual IObject* create(int modelID, Vector3 position, Vector3 rotation, float drawDist = 0.f) = 0;
};

struct ObjectEventHandler
{
virtual void onMoved(IObject& object) { }
virtual void onPlayerObjectMoved(IPlayer& player, IPlayerObject& object) { }
virtual void onObjectSelected(IPlayer& player, IObject& object, int model, Vector3 position) { }
virtual void onPlayerObjectSelected(IPlayer& player, IPlayerObject& object, int model, Vector3 position) { }
virtual void onObjectEdited(IPlayer& player, IObject& object, ObjectEditResponse response, Vector3 offset, Vector3 rotation) { }
virtual void onPlayerObjectEdited(IPlayer& player, IPlayerObject& object, ObjectEditResponse response, Vector3 offset, Vector3 rotation) { }
virtual void onPlayerAttachedObjectEdited(IPlayer& player, int index, bool saved, const ObjectAttachmentSlotData& data) { }
};

static const UID PlayerObjectData_UID = UID(0x93d4ed2344b07456);
struct IPlayerObjectData : public IExtension, public IPool<IPlayerObject>
{
PROVIDE_EXT_UID(PlayerObjectData_UID);

virtual IPlayerObject* create(int modelID, Vector3 position, Vector3 rotation, float drawDist = 0.f) = 0;

virtual void setAttachedObject(int index, const ObjectAttachmentSlotData& data) = 0;

virtual void removeAttachedObject(int index) = 0;

virtual bool hasAttachedObject(int index) const = 0;

virtual const ObjectAttachmentSlotData& getAttachedObject(int index) const = 0;

virtual void beginSelecting() = 0;

virtual bool selectingObject() const = 0;

virtual void endEditing() = 0;

virtual void beginEditing(IObject& object) = 0;

virtual void beginEditing(IPlayerObject& object) = 0;

virtual bool editingObject() const = 0;

virtual void editAttachedObject(int index) = 0;
};
