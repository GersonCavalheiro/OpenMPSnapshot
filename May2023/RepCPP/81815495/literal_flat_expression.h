
#pragma once

#include <vector>
#include <string>

#include "includes/define.h"
#include "includes/ublas_interface.h"
#include "containers/container_expression/expressions/expression.h"

namespace Kratos {



template<class TRawDataType = double>
class KRATOS_API(KRATOS_CORE) LiteralFlatExpression : public Expression {
public:

using Pointer = Kratos::intrusive_ptr<LiteralFlatExpression<TRawDataType>>;

using value_type = TRawDataType;

using size_type = std::size_t;

using iterator = TRawDataType*;

using const_iterator = TRawDataType const*;


LiteralFlatExpression(
const IndexType NumberOfEntities,
const std::vector<IndexType>& rShape);

LiteralFlatExpression(
TRawDataType* pDataBegin,
const IndexType NumberOfEntities,
const std::vector<IndexType>& rShape);



static LiteralFlatExpression<TRawDataType>::Pointer Create(
const IndexType NumberOfEntities,
const std::vector<IndexType>& rShape);

static LiteralFlatExpression<TRawDataType>::Pointer Create(
TRawDataType* pDataBegin,
const IndexType NumberOfEntities,
const std::vector<IndexType>& rShape);

void SetData(
const IndexType EntityDataBeginIndex,
const IndexType ComponentIndex,
const TRawDataType Value);

const std::vector<IndexType> GetItemShape() const override;

IndexType size() const noexcept { return mData.size(); }

iterator begin() noexcept { return mData.begin(); }

iterator end() noexcept { return mData.end(); }

const_iterator begin() const noexcept { return mData.begin(); }

const_iterator end() const noexcept { return mData.end(); }

const_iterator cbegin() const noexcept { return mData.begin(); }

const_iterator cend() const noexcept { return mData.end(); }

std::string Info() const override;

protected:


class Data
{
public:


Data(const IndexType Size): mpBegin(new TRawDataType[Size]), mIsManaged(true), mSize(Size) {}


Data(TRawDataType* pBegin, const IndexType Size): mpBegin(pBegin), mIsManaged(false), mSize(Size) {}

~Data() { if (mIsManaged) { delete[] mpBegin; } }


inline iterator begin() noexcept { return mpBegin; }

inline iterator end() noexcept { return mpBegin + mSize; }

inline const_iterator begin() const noexcept { return mpBegin; }

inline const_iterator end() const noexcept { return mpBegin + mSize; }

inline const_iterator cbegin() const noexcept { return mpBegin; }

inline const_iterator cend() const noexcept { return mpBegin + mSize; }

inline IndexType size() const noexcept { return mSize; }

private:

TRawDataType* mpBegin;

const bool mIsManaged;

const IndexType mSize;

};


const std::vector<IndexType> mShape;

Data mData;

};

template<class TRawDataType = double>
class LiteralScalarFlatExpression : public LiteralFlatExpression<TRawDataType>
{
public:

using IndexType = std::size_t;

using LiteralFlatExpression<TRawDataType>::LiteralFlatExpression;

double Evaluate(
const IndexType EntityIndex,
const IndexType EntityDataBeginIndex,
const IndexType ComponentIndex) const override;

};

template<class TRawDataType = double>
class LiteralNonScalarFlatExpression : public LiteralFlatExpression<TRawDataType>
{
public:

using IndexType = std::size_t;

using LiteralFlatExpression<TRawDataType>::LiteralFlatExpression;

double Evaluate(
const IndexType EntityIndex,
const IndexType EntityDataBeginIndex,
const IndexType ComponentIndex) const override;

};

} 