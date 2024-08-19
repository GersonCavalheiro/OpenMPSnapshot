

#pragma once


#include "../Manager/Manager.hpp"
#include "Impl.hpp"
#include "sdk.hpp"
#include <Server/Components/Pawn/pawn_natives.hpp>

namespace pawn_natives
{
template <>
class ParamCast<PawnScript&>
{
public:
ParamCast(AMX* amx, cell* params, int idx)
: value_(PawnManager::Get()->amxToScript_.find(amx)->second)
{
}

~ParamCast()
{
}

ParamCast(ParamCast<IDatabaseConnection*> const&) = delete;
ParamCast(ParamCast<IDatabaseConnection*>&&) = delete;

operator PawnScript&()
{
return *value_;
}

static constexpr int Size = 0;

private:
PawnScript* value_;
};
}
