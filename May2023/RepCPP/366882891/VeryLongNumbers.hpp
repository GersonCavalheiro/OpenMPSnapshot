#pragma once

class VeryLongNumber {
private:
int point = 1010;
public:
int length = 1011;
int array[1011] = { 0 };

VeryLongNumber();
VeryLongNumber(int* numberm, int size);

int GetLength() const;
int GetPoint() const;

void operator=(const VeryLongNumber& VLN);
VeryLongNumber operator+=(const VeryLongNumber& VLN);
VeryLongNumber operator+(const VeryLongNumber& VLN);
VeryLongNumber operator-=(const VeryLongNumber& VLN);
VeryLongNumber operator-(const VeryLongNumber& VLN);
VeryLongNumber operator*=(const int n);
VeryLongNumber operator*(const int n);
VeryLongNumber operator*=(const VeryLongNumber& VLN);
VeryLongNumber operator*(const VeryLongNumber& VLN);
VeryLongNumber operator/=(const int n);
VeryLongNumber operator/(const int n);
};

VeryLongNumber::VeryLongNumber() {};
VeryLongNumber::VeryLongNumber(int* number, int size) {
for (int i = 0; i < size; ++i) {
array[i] = number[i];
}
};

int VeryLongNumber::GetLength() const { return length; }
int VeryLongNumber::GetPoint() const { return point; }

class::VeryLongNumber VeryLongNumber::operator+=(const VeryLongNumber& VLN) {
int c = 0;

for (int i = this->length - 1; i >= 0; --i) {
c = this->array[i] + VLN.array[i] + c;
this->array[i] = c % 10;
c /= 10;
}
return *this;
}
class::VeryLongNumber VeryLongNumber::operator+(const VeryLongNumber& VLN) {
VeryLongNumber a(*this);
a += VLN;
return a;
}
class::VeryLongNumber VeryLongNumber::operator-=(const VeryLongNumber& VLN) {
int c = 0;

for (int i = this->length - 1; i >= 0; --i) {
c = this->array[i] - VLN.array[i] + c + 10;
this->array[i] = c % 10;
if (c < 10) { c = -1; }
else { c = 0; }
}
return *this;
}
class::VeryLongNumber VeryLongNumber::operator-(const VeryLongNumber& VLN) {
VeryLongNumber a(*this);
a -= VLN;
return a;
}
class::VeryLongNumber VeryLongNumber::operator*=(const int n) {
int c = 0;

for (int i = this->length - 1; i >= 0; --i) {
c = c + (array[i] * n);
array[i] = c % 10;
c /= 10;
}
return *this;
}
class::VeryLongNumber VeryLongNumber::operator*(const int n) {
VeryLongNumber a(*this);
a *= n;
return a;
}
class::VeryLongNumber VeryLongNumber::operator*=(const VeryLongNumber& VLN) {
*this = *this * VLN;
return *this;
}
class::VeryLongNumber VeryLongNumber::operator*(const VeryLongNumber& VLN) {
int c = 0;
VeryLongNumber a;

for (int i = 0; i < length; ++i) {
for (int j = 0; j < length - i; ++j) {
a.array[i + j] += this->array[i] * VLN.array[j];
}
}

for (int i = this->length - 1; i > 0; --i) {
a.array[i - 1] += a.array[i] / 10;
a.array[i] %= 10;
}

return a;
}
class::VeryLongNumber VeryLongNumber::operator/=(const int n) {
int c = 0;

for (int i = 0; i < this->length; ++i) {
c = c + array[i];
array[i] = c / n;
c = 10 * (c % n);
}
return *this;
}
class::VeryLongNumber VeryLongNumber::operator/(const int n) {
VeryLongNumber a(*this);
a /= n;
return a;
}
std::ostream& operator<<(std::ostream& out, VeryLongNumber VLN) {
for (int i = 0; i < VLN.length; ++i) {
out << VLN.array[i];
if (VLN.GetPoint() + i + 1 == VLN.GetLength()) { out << "."; }
}
return out;
}

void VeryLongNumber::operator=(const VeryLongNumber& VLN) {
this->point = VLN.GetPoint();
this->length = VLN.GetLength();

for (int i = 0; i < this->length; ++i) {
this->array[i] = VLN.array[i];
}
}
