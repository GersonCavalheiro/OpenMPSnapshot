#pragma once

#include <utility>

#include "algo/interfaces/serial/SerialInstrumental.h"
#include "algo/interfaces/AbstractSweepMethod.h"


class SerialSweepMethod : public SerialInstrumental, public AbstractSweepMethod {
protected:
vec Phi;
pairs kappa, mu, gamma;

public:
SerialSweepMethod() = default;

explicit SerialSweepMethod(size_t N) :
SerialSweepMethod(N,vec(N - 1, 1),std::make_pair(0., 0.), std::make_pair(1., 1.), std::make_pair(1., 1.)) {}

SerialSweepMethod(size_t N, vec phi, pairs kappa_, pairs mu_, pairs gamma_) :
SerialInstrumental(N),
Phi(std::move(phi)),
kappa(std::move(kappa_)), mu(std::move(mu_)), gamma(std::move(gamma_)) {}

SerialSweepMethod(const vec& a, vec c, vec b, vec phi, pairs kappa_, pairs mu_, pairs gamma_) :
SerialInstrumental(a, std::move(c), std::move(b)),
Phi(std::move(phi)),
kappa(std::move(kappa_)), mu(std::move(mu_)), gamma(std::move(gamma_)) {}

std::tuple<vec, vec, size_t, size_t, double, vec,
vec, vec, vec,
vec, pairs, pairs, pairs> getFields() const;

void setAllFields(const vec& v, const vec& u, size_t N, size_t node, double h, const vec& x,
const vec& A, const vec& C, const vec& B,
const vec& Phi_, pairs kappa_, pairs mu_, pairs gamma_);


vec run() override;
};