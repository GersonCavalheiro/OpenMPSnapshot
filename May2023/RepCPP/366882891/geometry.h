#pragma once


struct vector3D {
float x, y, z;

vector3D();
vector3D(float sx, float sy, float sz);



vector3D operator+(const vector3D& v) const;
vector3D operator-(const vector3D& v) const;
vector3D operator*(float c) const;
vector3D operator/(float c) const;

vector3D cross(const vector3D& v) const; 
float dot(const vector3D& v) const; 
float norm() const; 
};

class tryangle {
private:
vector3D tr1, tr2, tr3;

public:

tryangle(vector3D ntr1, vector3D ntr2, vector3D ntr3);
vector3D Get1() const;
vector3D Get2() const;
vector3D Get3() const;
};

class sphere {
private:
vector3D sph;
float radius;
public:

};




vector3D::vector3D() {
x = y = z = 0;
}

vector3D::vector3D(float sx, float sy, float sz) {
x = sx;
y = sy;
z = sz;
}

class vector3D vector3D::operator+(const vector3D& v)  const {
return vector3D(x + v.x, y + v.y, z + v.z);
}
class vector3D vector3D::operator-(const vector3D& v)  const {
return vector3D(x - v.x, y - v.y, z - v.z);
}
class vector3D vector3D::operator*(float c) const {
return vector3D(c*x, c*y, c*z);
}
class vector3D vector3D::operator/(float c) const {
return vector3D(x / c, y / c, z / c);
}

tryangle::tryangle(vector3D ntr1, vector3D ntr2, vector3D ntr3) {
tr1 = ntr1;
tr2 = ntr2;
tr3 = ntr3;
}
vector3D tryangle::Get1()  const { return tr1; }
vector3D tryangle::Get2()  const { return tr2; }
vector3D tryangle::Get3()  const { return tr3; }

vector3D vector3D::cross(const vector3D& v)  const {
return (vector3D(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x));
}
float vector3D::dot(const vector3D& v)  const {
return float(x*v.x+y*v.y+z*v.z);
}
float vector3D::norm() const {
return sqrt(this->dot(vector3D(x, y, z)));
}
