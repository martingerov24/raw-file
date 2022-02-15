#pragma once
#define OPENCV_ABI_COMPATIBILITY   300
#include "../dep/Dep.h"
#include <cassert>

template<typename _Tp> 
class Size_
{
public:
	typedef _Tp value_type;

	//! default constructor
	Size_();
	Size_(_Tp _width, _Tp _height);
#if OPENCV_ABI_COMPATIBILITY < 500
	Size_(const Size_& sz) = default;
	Size_(Size_&& sz) noexcept = default;
#endif
	Size_(const Point2<_Tp>& pt);

#if OPENCV_ABI_COMPATIBILITY < 500
	Size_& operator = (const Size_& sz) = default;
	Size_& operator = (Size_&& sz) noexcept = default;
#endif
	//! the area (width*height)
	_Tp area() const;
	//! aspect ratio (width/height)
	float aspectRatio() const;
	//! true if empty
	bool empty() const;

	//! conversion of another data type.
	template<typename _Tp2> operator Size_<_Tp2>() const;

	_Tp width; //!< the width
	_Tp height; //!< the height
};

typedef Size_<int> Size2i;
typedef Size_<int64_t> Size2l;
typedef Size_<float> Size2f;
typedef Size2i Size;


template<typename _Tp> 
class Rect_
{
public:
	typedef _Tp value_type;

	//! default constructor
	Rect_();
	Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
#if OPENCV_ABI_COMPATIBILITY < 500
	Rect_(const Rect_& r) = default;
	Rect_(Rect_&& r) noexcept = default;
#endif
	Rect_(const Point2<_Tp>& org, const Size_<_Tp>& sz);
	Rect_(const Point2<_Tp>& pt1, const Point2<_Tp>& pt2);

#if OPENCV_ABI_COMPATIBILITY < 500
	Rect_& operator = (const Rect_& r) = default;
	Rect_& operator = (Rect_&& r) noexcept = default;
#endif
	//! the top-left corner
	Point2<_Tp> tl() const;
	//! the bottom-right corner
	Point2<_Tp> br() const;

	//! size (width, height) of the rectangle
	Size_<_Tp> size() const;
	//! area (width*height) of the rectangle
	_Tp area() const;
	//! true if empty
	bool empty() const;

	//! conversion to another data type
	template<typename _Tp2> operator Rect_<_Tp2>() const;

	//! checks whether the rectangle contains the point
	bool contains(const Point2<_Tp>& pt) const;

	_Tp x; //!< x coordinate of the top-left corner
	_Tp y; //!< y coordinate of the top-left corner
	_Tp width; //!< width of the rectangle
	_Tp height; //!< height of the rectangle
};

typedef Rect_<int> Rect2i;
typedef Rect_<float> Rect2f;
typedef Rect2i Rect;


/// <summary>
/// Size implementation
/// </summary>
/// <typeparam name="_Tp"></typeparam>
template<typename _Tp> inline
Size_<_Tp>::Size_()
	: width(0), height(0) {}

template<typename _Tp> inline
Size_<_Tp>::Size_(_Tp _width, _Tp _height)
	: width(_width), height(_height) {}

template<typename _Tp> inline
Size_<_Tp>::Size_(const Point2<_Tp>& pt)
	: width(pt.x), height(pt.y) {}

template<typename _Tp> template<typename _Tp2> inline
Size_<_Tp>::operator Size_<_Tp2>() const
{
	return Size_<_Tp2>(saturate_cast<_Tp2>(width), saturate_cast<_Tp2>(height));
}

template<typename _Tp> inline
_Tp Size_<_Tp>::area() const
{
	const _Tp result = width * height;
	CV_DbgAssert(!std::numeric_limits<_Tp>::is_integer
		|| width == 0 || result / width == height); // make sure the result fits in the return value
	return result;
}

template<typename _Tp> inline
float Size_<_Tp>::aspectRatio() const
{
	return width / static_cast<float>(height);
}

template<typename _Tp> inline
bool Size_<_Tp>::empty() const
{
	return width <= 0 || height <= 0;
}


template<typename _Tp> static inline
Size_<_Tp>& operator *= (Size_<_Tp>& a, _Tp b)
{
	a.width *= b;
	a.height *= b;
	return a;
}

template<typename _Tp> static inline
Size_<_Tp> operator * (const Size_<_Tp>& a, _Tp b)
{
	Size_<_Tp> tmp(a);
	tmp *= b;
	return tmp;
}

template<typename _Tp> static inline
Size_<_Tp>& operator /= (Size_<_Tp>& a, _Tp b)
{
	a.width /= b;
	a.height /= b;
	return a;
}

template<typename _Tp> static inline
Size_<_Tp> operator / (const Size_<_Tp>& a, _Tp b)
{
	Size_<_Tp> tmp(a);
	tmp /= b;
	return tmp;
}

template<typename _Tp> static inline
Size_<_Tp>& operator += (Size_<_Tp>& a, const Size_<_Tp>& b)
{
	a.width += b.width;
	a.height += b.height;
	return a;
}

template<typename _Tp> static inline
Size_<_Tp> operator + (const Size_<_Tp>& a, const Size_<_Tp>& b)
{
	Size_<_Tp> tmp(a);
	tmp += b;
	return tmp;
}

template<typename _Tp> static inline
Size_<_Tp>& operator -= (Size_<_Tp>& a, const Size_<_Tp>& b)
{
	a.width -= b.width;
	a.height -= b.height;
	return a;
}

template<typename _Tp> static inline
Size_<_Tp> operator - (const Size_<_Tp>& a, const Size_<_Tp>& b)
{
	Size_<_Tp> tmp(a);
	tmp -= b;
	return tmp;
}

template<typename _Tp> static inline
bool operator == (const Size_<_Tp>& a, const Size_<_Tp>& b)
{
	return a.width == b.width && a.height == b.height;
}

template<typename _Tp> static inline
bool operator != (const Size_<_Tp>& a, const Size_<_Tp>& b)
{
	return !(a == b);
}


/// <summary>
/// Rect
/// </summary>
/// <typeparam name="_Tp"></typeparam>
template<typename _Tp> inline
Rect_<_Tp>::Rect_()
	: x(0), y(0), width(0), height(0) {}

template<typename _Tp> inline
Rect_<_Tp>::Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height)
	: x(_x), y(_y), width(_width), height(_height) {}

template<typename _Tp> inline
Rect_<_Tp>::Rect_(const Point2<_Tp>& org, const Size_<_Tp>& sz)
	: x(org.x), y(org.y), width(sz.width), height(sz.height) {}

template<typename _Tp> inline
Rect_<_Tp>::Rect_(const Point2<_Tp>& pt1, const Point2<_Tp>& pt2)
{
	x = std::min(pt1.x, pt2.x);
	y = std::min(pt1.y, pt2.y);
	width = std::max(pt1.x, pt2.x) - x;
	height = std::max(pt1.y, pt2.y) - y;
}

template<typename _Tp> inline
Point2<_Tp> Rect_<_Tp>::tl() const
{
	return Point2<_Tp>(x, y);
}

template<typename _Tp> inline
Point2<_Tp> Rect_<_Tp>::br() const
{
	return Point2<_Tp>(x + width, y + height);
}

template<typename _Tp> inline
Size_<_Tp> Rect_<_Tp>::size() const
{
	return Size_<_Tp>(width, height);
}

template<typename _Tp> inline
_Tp Rect_<_Tp>::area() const
{
	const _Tp result = width * height;
	assert(!std::numeric_limits<_Tp>::is_integer
		|| width == 0 || result / width == height); // make sure the result fits in the return value
	return result;
}

template<typename _Tp> inline
bool Rect_<_Tp>::empty() const
{
	return width <= 0 || height <= 0;
}

template<typename _Tp> template<typename _Tp2> inline
Rect_<_Tp>::operator Rect_<_Tp2>() const
{
	return Rect_<_Tp2>(static_cast<_Tp2>(x), static_cast<_Tp2>(y), static_cast<_Tp2>(width), static_cast<_Tp2>(height));
}

template<typename _Tp> inline
bool Rect_<_Tp>::contains(const Point2<_Tp>& pt) const
{
	return x <= pt.x && pt.x < x + width && y <= pt.y && pt.y < y + height;
}


template<typename _Tp> static inline
Rect_<_Tp>& operator += (Rect_<_Tp>& a, const Point2<_Tp>& b)
{
	a.x += b.x;
	a.y += b.y;
	return a;
}

template<typename _Tp> static inline
Rect_<_Tp>& operator -= (Rect_<_Tp>& a, const Point2<_Tp>& b)
{
	a.x -= b.x;
	a.y -= b.y;
	return a;
}

template<typename _Tp> static inline
Rect_<_Tp>& operator += (Rect_<_Tp>& a, const Size_<_Tp>& b)
{
	a.width += b.width;
	a.height += b.height;
	return a;
}

template<typename _Tp> static inline
Rect_<_Tp>& operator -= (Rect_<_Tp>& a, const Size_<_Tp>& b)
{
	const _Tp width = a.width - b.width;
	const _Tp height = a.height - b.height;
	assert(width >= 0 && height >= 0);
	a.width = width;
	a.height = height;
	return a;
}

template<typename _Tp> static inline
Rect_<_Tp>& operator &= (Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
	_Tp x1 = std::max(a.x, b.x);
	_Tp y1 = std::max(a.y, b.y);
	a.width = std::min(a.x + a.width, b.x + b.width) - x1;
	a.height = std::min(a.y + a.height, b.y + b.height) - y1;
	a.x = x1;
	a.y = y1;
	if (a.width <= 0 || a.height <= 0)
		a = Rect();
	return a;
}

template<typename _Tp> static inline
Rect_<_Tp>& operator |= (Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
	if (a.empty()) {
		a = b;
	}
	else if (!b.empty()) {
		_Tp x1 = std::min(a.x, b.x);
		_Tp y1 = std::min(a.y, b.y);
		a.width = std::max(a.x + a.width, b.x + b.width) - x1;
		a.height = std::max(a.y + a.height, b.y + b.height) - y1;
		a.x = x1;
		a.y = y1;
	}
	return a;
}

template<typename _Tp> static inline
bool operator == (const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
	return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
}

template<typename _Tp> static inline
bool operator != (const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
	return a.x != b.x || a.y != b.y || a.width != b.width || a.height != b.height;
}

template<typename _Tp> static inline
Rect_<_Tp> operator + (const Rect_<_Tp>& a, const Point2<_Tp>& b)
{
	return Rect_<_Tp>(a.x + b.x, a.y + b.y, a.width, a.height);
}

template<typename _Tp> static inline
Rect_<_Tp> operator - (const Rect_<_Tp>& a, const Point2<_Tp>& b)
{
	return Rect_<_Tp>(a.x - b.x, a.y - b.y, a.width, a.height);
}

template<typename _Tp> static inline
Rect_<_Tp> operator + (const Rect_<_Tp>& a, const Size_<_Tp>& b)
{
	return Rect_<_Tp>(a.x, a.y, a.width + b.width, a.height + b.height);
}

template<typename _Tp> static inline
Rect_<_Tp> operator - (const Rect_<_Tp>& a, const Size_<_Tp>& b)
{
	const _Tp width = a.width - b.width;
	const _Tp height = a.height - b.height;
	assert(width >= 0 && height >= 0);
	return Rect_<_Tp>(a.x, a.y, width, height);
}

template<typename _Tp> static inline
Rect_<_Tp> operator & (const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
	Rect_<_Tp> c = a;
	return c &= b;
}

template<typename _Tp> static inline
Rect_<_Tp> operator | (const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
	Rect_<_Tp> c = a;
	return c |= b;
}

template<typename _Tp> static inline
float jaccardDistance(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
	_Tp Aa = a.area();
	_Tp Ab = b.area();

	if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
		// jaccard_index = 1 -> distance = 0
		return 0.0;
	}

	float  Aab = (a & b).area();
	// distance = 1 - jaccard_index
	return 1.0 - Aab / (Aa + Ab - Aab);
}