#pragma once

#include <component.hpp>
#include <player.hpp>
#include <types.hpp>
#include <values.hpp>

enum TextDrawAlignmentTypes
{
TextDrawAlignment_Default,
TextDrawAlignment_Left,
TextDrawAlignment_Center,
TextDrawAlignment_Right
};

enum TextDrawStyle
{
TextDrawStyle_0, 
TextDrawStyle_1, 
TextDrawStyle_2, 
TextDrawStyle_3, 
TextDrawStyle_4, 
TextDrawStyle_5, 
TextDrawStyle_FontBeckettRegular = 0, 
TextDrawStyle_FontAharoniBold, 
TextDrawStyle_FontBankGothic, 
TextDrawStyle_FontPricedown, 
TextDrawStyle_Sprite, 
TextDrawStyle_Preview 
};

struct ITextDrawBase : public IExtensible, public IIDProvider
{
virtual Vector2 getPosition() const = 0;

virtual ITextDrawBase& setPosition(Vector2 position) = 0;

virtual void setText(StringView text) = 0;

virtual StringView getText() const = 0;

virtual ITextDrawBase& setLetterSize(Vector2 size) = 0;

virtual Vector2 getLetterSize() const = 0;

virtual ITextDrawBase& setTextSize(Vector2 size) = 0;

virtual Vector2 getTextSize() const = 0;

virtual ITextDrawBase& setAlignment(TextDrawAlignmentTypes alignment) = 0;

virtual TextDrawAlignmentTypes getAlignment() const = 0;

virtual ITextDrawBase& setColour(Colour colour) = 0;

virtual Colour getLetterColour() const = 0;

virtual ITextDrawBase& useBox(bool use) = 0;

virtual bool hasBox() const = 0;

virtual ITextDrawBase& setBoxColour(Colour colour) = 0;

virtual Colour getBoxColour() const = 0;

virtual ITextDrawBase& setShadow(int shadow) = 0;

virtual int getShadow() const = 0;

virtual ITextDrawBase& setOutline(int outline) = 0;

virtual int getOutline() const = 0;

virtual ITextDrawBase& setBackgroundColour(Colour colour) = 0;

virtual Colour getBackgroundColour() const = 0;

virtual ITextDrawBase& setStyle(TextDrawStyle style) = 0;

virtual TextDrawStyle getStyle() const = 0;

virtual ITextDrawBase& setProportional(bool proportional) = 0;

virtual bool isProportional() const = 0;

virtual ITextDrawBase& setSelectable(bool selectable) = 0;

virtual bool isSelectable() const = 0;

virtual ITextDrawBase& setPreviewModel(int model) = 0;

virtual int getPreviewModel() const = 0;

virtual ITextDrawBase& setPreviewRotation(Vector3 rotation) = 0;

virtual Vector3 getPreviewRotation() const = 0;

virtual ITextDrawBase& setPreviewVehicleColour(int colour1, int colour2) = 0;

virtual Pair<int, int> getPreviewVehicleColour() const = 0;

virtual ITextDrawBase& setPreviewZoom(float zoom) = 0;

virtual float getPreviewZoom() const = 0;

virtual void restream() = 0;
};

struct ITextDraw : public ITextDrawBase
{
virtual void showForPlayer(IPlayer& player) = 0;

virtual void hideForPlayer(IPlayer& player) = 0;

virtual bool isShownForPlayer(const IPlayer& player) const = 0;

virtual void setTextForPlayer(IPlayer& player, StringView text) = 0;
};

struct IPlayerTextDraw : public ITextDrawBase
{
virtual void show() = 0;

virtual void hide() = 0;

virtual bool isShown() const = 0;
};

struct TextDrawEventHandler
{
virtual void onPlayerClickTextDraw(IPlayer& player, ITextDraw& td) { }
virtual void onPlayerClickPlayerTextDraw(IPlayer& player, IPlayerTextDraw& td) { }
virtual bool onPlayerCancelTextDrawSelection(IPlayer& player) { return false; }
virtual bool onPlayerCancelPlayerTextDrawSelection(IPlayer& player) { return false; }
};

static const UID TextDrawsComponent_UID = UID(0x9b5dc2b1d15c992a);
struct ITextDrawsComponent : public IPoolComponent<ITextDraw>
{
PROVIDE_UID(TextDrawsComponent_UID);

virtual IEventDispatcher<TextDrawEventHandler>& getEventDispatcher() = 0;

virtual ITextDraw* create(Vector2 position, StringView text) = 0;

virtual ITextDraw* create(Vector2 position, int model) = 0;
};

static const UID PlayerTextDrawData_UID = UID(0xbf08495682312400);
struct IPlayerTextDrawData : public IExtension, public IPool<IPlayerTextDraw>
{
PROVIDE_EXT_UID(PlayerTextDrawData_UID);

virtual void beginSelection(Colour highlight) = 0;

virtual bool isSelecting() const = 0;

virtual void endSelection() = 0;

virtual IPlayerTextDraw* create(Vector2 position, StringView text) = 0;

virtual IPlayerTextDraw* create(Vector2 position, int model) = 0;
};
