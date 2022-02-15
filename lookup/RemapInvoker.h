#pragma once
#include "../lookup/Rect.h"
#define INT_MIN     (-2147483647 - 1)
#define INT_MAX       2147483647

#define INTER_MAX 7
#define CV_CN_SHIFT   3
#define CV_DEPTH_MAX	(1 << CV_CN_SHIFT)
#define CV_MAT_DEPTH 	( flags	) 	   ((flags) & CV_MAT_DEPTH_MASK)
#define CV_MAT_DEPTH_MASK	(CV_DEPTH_MAX - 1)
#define CV_MAT_DEPTH_MASK       (CV_DEPTH_MAX - 1)
#define CV_MAT_DEPTH(flags)     ((flags) & CV_MAT_DEPTH_MASK)
class Range
{
public:
	Range();
	Range(int _start, int _end);
	int size() const;
	bool empty() const;
	static Range all();

	int start, end;
};
inline int
cvRound(float value)
{
//#if defined CV_INLINE_ROUND_DBL
//	//CV_INLINE_ROUND_DBL(value);
//#elif ((defined _MSC_VER && defined _M_X64) || (defined __GNUC__ && defined __x86_64__ \
//    && defined __SSE2__ && !defined __APPLE__) || CV_SSE2) \
//    && !defined(__CUDACC__)
//	__m128d t = _mm_set_sd(value);
//	return _mm_cvtsd_si32(t);
//#elif defined _MSC_VER && defined _M_IX86
//	int t;
//	__asm
//	{
//		fld value;
//		fistp t;
//	}
//	return t;
//#elif defined CV_ICC || defined __GNUC__
//	return (int)(lrint(value));
//#else
	/* it's ok if round does not comply with IEEE754 standard;
	   the tests should allow +/-1 difference when the tested functions use round */
	return (int)(value + (value >= 0 ? 0.5 : -0.5));
//#endif
}
template<typename T>
inline int depth(Mat<T> & thing) 
{
	return CV_MAT_DEPTH(T);
}

inline
Range::Range()
	: start(0), end(0) {}

inline
Range::Range(int _start, int _end)
	: start(_start), end(_end) {}

inline
int Range::size() const
{
	return end - start;
}

inline
bool Range::empty() const
{
	return start == end;
}

inline
Range Range::all()
{
	return Range(INT_MIN, INT_MAX);
}


enum  InterpolationMasks {
	INTER_BITS = 5,
	INTER_BITS2 = INTER_BITS * 2,
	INTER_TAB_SIZE = 1 << INTER_BITS,
	INTER_TAB_SIZE2 = INTER_TAB_SIZE * INTER_TAB_SIZE
};

typedef void (*RemapNNFunc)(const Matf& _src, Matf& _dst, const Matf& _xy,
	int borderType, const void* _borderValue);

typedef void (*RemapFunc)(const Matf& _src, Matf& _dst, const Mat16s& _xy,
	const Mat16s& _fxy, const void* _wtab,
	int borderType, const void* _borderValue);

class ParallelLoopBody
{
public:
	virtual ~ParallelLoopBody();
	virtual void operator() (const Range& range) const = 0;
};

class RemapInvoker :
	public ParallelLoopBody
{
public:
	RemapInvoker(Matf _src, Matf _dst, const Mat16s* _m1, const Mat16u* _m2, int _broderType, bool _planar_input, RemapNNFunc _nnfunc, RemapFunc _ifunc, const void* _ctab) :
		ParallelLoopBody(), src(&_src), dst(&_dst), m1(_m1), m2(_m2),
		borderType(_broderType),
		planar_input(_planar_input), nnfunc(_nnfunc), ifunc(_ifunc), ctab(_ctab)
	{
	}

	virtual void operator() (const Range& range) const override
	{
		int x, y, x1, y1;
		const int buf_size = 1 << 14;
		int brows0 = std::min(128, dst->rows()), map_depth = 3;  //checked for float
		int bcols0 = std::min(buf_size / brows0, dst->cols());
		brows0 = std::min(buf_size / bcols0, dst->rows());
		Mat16s _bufa;
		Mat16s _bufxy(brows0, bcols0, 2);
		if (!nnfunc)
		{
			 _bufa(brows0, bcols0);
		}

		for (y = range.start; y < range.end; y += brows0)
		{
			for (x = 0; x < dst->cols(); x += bcols0) // why is it -> instead of . // TODO:?
			{
				int brows = std::min(brows0, range.end - y);
				int bcols = std::min(bcols0, dst->cols() - x);
				Matf dpart(*dst, Rect(x, y, bcols, brows));
				Mat16s bufxy(_bufxy, Rect(0, 0, bcols, brows));

//				if (nnfunc) // nope
//				{
//					if (m2->hasData() == false) // the data is already in the right format
//						bufxy = (*m1)(Rect(x, y, bcols, brows));
//					else if (map_depth != CV_32F)
//					{
//						for (y1 = 0; y1 < brows; y1++)
//						{
//							short* XY = bufxy.ptr<short>(y1);
//							const short* sXY = m1->ptr<short>(y + y1) + x * 2;
//							const ushort* sA = m2->ptr<ushort>(y + y1) + x;
//
//							for (x1 = 0; x1 < bcols; x1++)
//							{
//								int a = sA[x1] & (INTER_TAB_SIZE2 - 1);
//								XY[x1 * 2] = sXY[x1 * 2] + NNDeltaTab_i[a][0];
//								XY[x1 * 2 + 1] = sXY[x1 * 2 + 1] + NNDeltaTab_i[a][1];
//							}
//						}
//					}
//					else if (!planar_input)
//						(*m1)(Rect(x, y, bcols, brows)).convertTo(bufxy, bufxy.depth());
//					else
//					{
//						for (y1 = 0; y1 < brows; y1++)
//						{
//							short* XY = bufxy.ptr<short>(y1);
//							const float* sX = m1->ptr<float>(y + y1) + x;
//							const float* sY = m2->ptr<float>(y + y1) + x;
//							x1 = 0;
//
//#if CV_SIMD128
//							{
//								int span = v_float32x4::nlanes;
//								for (; x1 <= bcols - span * 2; x1 += span * 2)
//								{
//									v_int32x4 ix0 = v_round(v_load(sX + x1));
//									v_int32x4 iy0 = v_round(v_load(sY + x1));
//									v_int32x4 ix1 = v_round(v_load(sX + x1 + span));
//									v_int32x4 iy1 = v_round(v_load(sY + x1 + span));
//
//									v_int16x8 dx, dy;
//									dx = v_pack(ix0, ix1);
//									dy = v_pack(iy0, iy1);
//									v_store_interleave(XY + x1 * 2, dx, dy);
//								}
//							}
//#endif
//							for (; x1 < bcols; x1++)
//							{
//								XY[x1 * 2] = saturate_cast<short>(sX[x1]);
//								XY[x1 * 2 + 1] = saturate_cast<short>(sY[x1]);
//							}
//						}
//					}
//					nnfunc(*src, dpart, bufxy, borderType, borderValue);
//					continue;
//				}

				Mat16s bufa(_bufa, Rect(0, 0, bcols, brows));
				for (y1 = 0; y1 < brows; y1++)
				{
					short* XY = (short*)bufxy.ptr(y1); // was casting uchar to short, so i replaced the ptr() to return T
					unsigned short* A = (unsigned short*)bufa.ptr(y1);
					//short* XY = bufxy.ptr(y1);
					//int16_t* A = bufa.ptr(y1); 

					if (true) //&& (m2->type() == CV_16UC1 || m2->type() == CV_16SC1)&& m1->type() == CV_16SC2  it is ->
					{
						bufxy = (*m1)(Rect(x, y, bcols, brows)); // TODO: i don't know what this does

						const int16_t* sA = (const int16_t*)m2->ptr(y + y1) + x;
						x1 = 0;

#if CV_SIMD128
						{
							v_uint16x8 v_scale = v_setall_u16(INTER_TAB_SIZE2 - 1);
							int span = v_uint16x8::nlanes;
							for (; x1 <= bcols - span; x1 += span)
								v_store((unsigned short*)(A + x1), v_load(sA + x1) & v_scale);
						}
#endif
						for (; x1 < bcols; x1++)
							A[x1] = (unsigned short)(sA[x1] & (INTER_TAB_SIZE2 - 1));// has to be cast to either ushort or int16_t
					}
					//else if is commented here
//					else if (planar_input)
//					{
//						const float* sX = m1->ptr<float>(y + y1) + x;
//						const float* sY = m2->ptr<float>(y + y1) + x;
//
//						x1 = 0;
//#if CV_SIMD128
//						{
//							v_float32x4 v_scale = v_setall_f32((float)INTER_TAB_SIZE);
//							v_int32x4 v_scale2 = v_setall_s32(INTER_TAB_SIZE - 1);
//							int span = v_float32x4::nlanes;
//							for (; x1 <= bcols - span * 2; x1 += span * 2)
//							{
//								v_int32x4 v_sx0 = v_round(v_scale * v_load(sX + x1));
//								v_int32x4 v_sy0 = v_round(v_scale * v_load(sY + x1));
//								v_int32x4 v_sx1 = v_round(v_scale * v_load(sX + x1 + span));
//								v_int32x4 v_sy1 = v_round(v_scale * v_load(sY + x1 + span));
//								v_uint16x8 v_sx8 = v_reinterpret_as_u16(v_pack(v_sx0 & v_scale2, v_sx1 & v_scale2));
//								v_uint16x8 v_sy8 = v_reinterpret_as_u16(v_pack(v_sy0 & v_scale2, v_sy1 & v_scale2));
//								v_uint16x8 v_v = v_shl<INTER_BITS>(v_sy8) | (v_sx8);
//								v_store(A + x1, v_v);
//
//								v_int16x8 v_d0 = v_pack(v_shr<INTER_BITS>(v_sx0), v_shr<INTER_BITS>(v_sx1));
//								v_int16x8 v_d1 = v_pack(v_shr<INTER_BITS>(v_sy0), v_shr<INTER_BITS>(v_sy1));
//								v_store_interleave(XY + (x1 << 1), v_d0, v_d1);
//							}
//						}
//#endif
//						for (; x1 < bcols; x1++)
//						{
//							int sx = cvRound(sX[x1] * INTER_TAB_SIZE);
//							int sy = cvRound(sY[x1] * INTER_TAB_SIZE);
//							int v = (sy & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE - 1));
//							XY[x1 * 2] = static_cast<short>(sx >> INTER_BITS);// here was saturate_cast
//							XY[x1 * 2 + 1] = static_cast<short>(sy >> INTER_BITS);
//							A[x1] = (unsigned short)v;
//						}
//					}
					else
					{
						const float* sXY = (float*)m1->ptr(y + y1) + x * 2; // if i remove the tempalte part it may work, because i did not thought of solution TODO:
						x1 = 0;

#if CV_SIMD128
						{
							v_float32x4 v_scale = v_setall_f32((float)INTER_TAB_SIZE);
							v_int32x4 v_scale2 = v_setall_s32(INTER_TAB_SIZE - 1), v_scale3 = v_setall_s32(INTER_TAB_SIZE);
							int span = v_float32x4::nlanes;
							for (; x1 <= bcols - span * 2; x1 += span * 2)
							{
								v_float32x4 v_fx, v_fy;
								v_load_deinterleave(sXY + (x1 << 1), v_fx, v_fy);
								v_int32x4 v_sx0 = v_round(v_fx * v_scale);
								v_int32x4 v_sy0 = v_round(v_fy * v_scale);
								v_load_deinterleave(sXY + ((x1 + span) << 1), v_fx, v_fy);
								v_int32x4 v_sx1 = v_round(v_fx * v_scale);
								v_int32x4 v_sy1 = v_round(v_fy * v_scale);
								v_int32x4 v_v0 = v_muladd(v_scale3, (v_sy0 & v_scale2), (v_sx0 & v_scale2));
								v_int32x4 v_v1 = v_muladd(v_scale3, (v_sy1 & v_scale2), (v_sx1 & v_scale2));
								v_uint16x8 v_v8 = v_reinterpret_as_u16(v_pack(v_v0, v_v1));
								v_store(A + x1, v_v8);
								v_int16x8 v_dx = v_pack(v_shr<INTER_BITS>(v_sx0), v_shr<INTER_BITS>(v_sx1));
								v_int16x8 v_dy = v_pack(v_shr<INTER_BITS>(v_sy0), v_shr<INTER_BITS>(v_sy1));
								v_store_interleave(XY + (x1 << 1), v_dx, v_dy);
							}
						}
#endif

						for (; x1 < bcols; x1++)
						{
							int sx = cvRound(sXY[x1 * 2] * INTER_TAB_SIZE);
							int sy = cvRound(sXY[x1 * 2 + 1] * INTER_TAB_SIZE);
							int v = (sy & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE - 1));
							XY[x1 * 2] = static_cast<short>(sx >> INTER_BITS);
							XY[x1 * 2 + 1] = static_cast<short>(sy >> INTER_BITS);
							A[x1] = (unsigned short)v;
						}
					}
				}
				ifunc(*src, dpart, bufxy, bufa, ctab, borderType, borderValue);
			}
		}
	}

private:
	const Matf* src;
	Matf* dst;
	const Mat16s* m1;
	const Mat16u* m2;
	int borderType;
	void* borderValue;
	int planar_input;
	RemapNNFunc nnfunc;
	RemapFunc ifunc;
	const void* ctab;
};