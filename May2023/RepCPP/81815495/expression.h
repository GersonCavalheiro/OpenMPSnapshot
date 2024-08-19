
#pragma once

#include <atomic>
#include <ostream>
#include <string>
#include <vector>

#include "includes/define.h"

namespace Kratos {



class KRATOS_API(KRATOS_CORE) Expression {
public:


using Pointer = Kratos::intrusive_ptr<const Expression>;

using IndexType = std::size_t;


class ExpressionIterator {
public:

KRATOS_CLASS_POINTER_DEFINITION(ExpressionIterator);


ExpressionIterator();

ExpressionIterator(Expression::Pointer pExpression);


ExpressionIterator(const ExpressionIterator& rOther);


Expression::Pointer GetExpression() const;


double operator*() const;

bool operator==(const ExpressionIterator& rOther) const;

bool operator!=(const ExpressionIterator& rOther) const;

ExpressionIterator& operator=(const ExpressionIterator& rOther);

ExpressionIterator& operator++();

ExpressionIterator operator++(int);


private:

Expression::Pointer mpExpression;

IndexType mEntityIndex;

IndexType mEntityDataBeginIndex;

IndexType mItemComponentIndex;

IndexType mItemComponentCount;


friend class Expression;

};


using value_type = double;

using size_type = IndexType;

using const_iterator = ExpressionIterator;


Expression(const IndexType NumberOfEntities) : mNumberOfEntities(NumberOfEntities) {}

virtual ~Expression() = default;



virtual double Evaluate(
const IndexType EntityIndex,
const IndexType EntityDataBeginIndex,
const IndexType ComponentIndex) const = 0;


virtual const std::vector<IndexType> GetItemShape() const = 0;


inline IndexType NumberOfEntities() const { return mNumberOfEntities; };


IndexType GetItemComponentCount() const;


IndexType size() const;

const_iterator begin() const;

const_iterator end() const;

const_iterator cbegin() const;

const_iterator cend() const;

virtual std::string Info() const = 0;


private:

const IndexType mNumberOfEntities;


mutable std::atomic<int> mReferenceCounter{0};

friend void intrusive_ptr_add_ref(const Expression* x)
{
x->mReferenceCounter.fetch_add(1, std::memory_order_relaxed);
}

friend void intrusive_ptr_release(const Expression* x)
{
if (x->mReferenceCounter.fetch_sub(1, std::memory_order_release) == 1) {
std::atomic_thread_fence(std::memory_order_acquire);
delete x;
}
}


};

inline std::ostream& operator<<(
std::ostream& rOStream,
const Expression& rThis)
{
return rOStream << rThis.Info();
}

} 