class RemapInvoker :
	public ParallelLoopBody
{
public:
	RemapInvoker(const Mat& _src, Mat& _dst, const Mat* _m1,
		const Mat* _m2, int _borderType, const Scalar& _borderValue,
		int _planar_input, RemapNNFunc _nnfunc, RemapFunc _ifunc, const void* _ctab) :
		ParallelLoopBody(), src(&_src), dst(&_dst), m1(_m1), m2(_m2),
		borderType(_borderType), borderValue(_borderValue),
		planar_input(_planar_input), nnfunc(_nnfunc), ifunc(_ifunc), ctab(_ctab)
	{
	}

	virtual void operator() (const Range& range) const CV_OVERRIDE
	{
		int x, y, x1, y1;
		const int buf_size = 1 << 14;
		int brows0 = std::min(128, dst->rows), map_depth = m1->depth();
		int bcols0 = std::min(buf_size / brows0, dst->cols);
		brows0 = std::min(buf_size / bcols0, dst->rows);

		Mat _bufxy(brows0, bcols0, CV_16SC2), _bufa;
		if (!nnfunc)
			_bufa.create(brows0, bcols0, CV_16UC1);

		for (y = range.start; y < range.end; y += brows0)
		{
			for (x = 0; x < dst->cols; x += bcols0)
			{
				int brows = std::min(brows0, range.end - y);
				int bcols = std::min(bcols0, dst->cols - x);
				Mat dpart(*dst, Rect(x, y, bcols, brows));
				Mat bufxy(_bufxy, Rect(0, 0, bcols, brows));

				if (nnfunc)
				{
					if (m1->type() == CV_16SC2 && m2->empty()) // the data is already in the right format
						bufxy = (*m1)(Rect(x, y, bcols, brows));
					else if (map_depth != CV_32F)
					{
						for (y1 = 0; y1 < brows; y1++)
						{
							short* XY = bufxy.ptr<short>(y1);
							const short* sXY = m1->ptr<short>(y + y1) + x * 2;
							const ushort* sA = m2->ptr<ushort>(y + y1) + x;

							for (x1 = 0; x1 < bcols; x1++)
							{
								int a = sA[x1] & (INTER_TAB_SIZE2 - 1);
								XY[x1 * 2] = sXY[x1 * 2] + NNDeltaTab_i[a][0];
								XY[x1 * 2 + 1] = sXY[x1 * 2 + 1] + NNDeltaTab_i[a][1];
							}
						}
					}
					else if (!planar_input)
						(*m1)(Rect(x, y, bcols, brows)).convertTo(bufxy, bufxy.depth());
					else
					{
						for (y1 = 0; y1 < brows; y1++)
						{
							short* XY = bufxy.ptr<short>(y1);
							const float* sX = m1->ptr<float>(y + y1) + x;
							const float* sY = m2->ptr<float>(y + y1) + x;
							x1 = 0;

#if CV_SIMD128
							{
								int span = v_float32x4::nlanes;
								for (; x1 <= bcols - span * 2; x1 += span * 2)
								{
									v_int32x4 ix0 = v_round(v_load(sX + x1));
									v_int32x4 iy0 = v_round(v_load(sY + x1));
									v_int32x4 ix1 = v_round(v_load(sX + x1 + span));
									v_int32x4 iy1 = v_round(v_load(sY + x1 + span));

									v_int16x8 dx, dy;
									dx = v_pack(ix0, ix1);
									dy = v_pack(iy0, iy1);
									v_store_interleave(XY + x1 * 2, dx, dy);
								}
							}
#endif
							for (; x1 < bcols; x1++)
							{
								XY[x1 * 2] = saturate_cast<short>(sX[x1]);
								XY[x1 * 2 + 1] = saturate_cast<short>(sY[x1]);
							}
						}
					}
					nnfunc(*src, dpart, bufxy, borderType, borderValue);
					continue;
				}

				Mat bufa(_bufa, Rect(0, 0, bcols, brows));
				for (y1 = 0; y1 < brows; y1++)
				{
					short* XY = bufxy.ptr<short>(y1);
					ushort* A = bufa.ptr<ushort>(y1);

					if (m1->type() == CV_16SC2 && (m2->type() == CV_16UC1 || m2->type() == CV_16SC1))
					{
						bufxy = (*m1)(Rect(x, y, bcols, brows));

						const ushort* sA = m2->ptr<ushort>(y + y1) + x;
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
							A[x1] = (ushort)(sA[x1] & (INTER_TAB_SIZE2 - 1));
					}
					else if (planar_input)
					{
						const float* sX = m1->ptr<float>(y + y1) + x;
						const float* sY = m2->ptr<float>(y + y1) + x;

						x1 = 0;
#if CV_SIMD128
						{
							v_float32x4 v_scale = v_setall_f32((float)INTER_TAB_SIZE);
							v_int32x4 v_scale2 = v_setall_s32(INTER_TAB_SIZE - 1);
							int span = v_float32x4::nlanes;
							for (; x1 <= bcols - span * 2; x1 += span * 2)
							{
								v_int32x4 v_sx0 = v_round(v_scale * v_load(sX + x1));
								v_int32x4 v_sy0 = v_round(v_scale * v_load(sY + x1));
								v_int32x4 v_sx1 = v_round(v_scale * v_load(sX + x1 + span));
								v_int32x4 v_sy1 = v_round(v_scale * v_load(sY + x1 + span));
								v_uint16x8 v_sx8 = v_reinterpret_as_u16(v_pack(v_sx0 & v_scale2, v_sx1 & v_scale2));
								v_uint16x8 v_sy8 = v_reinterpret_as_u16(v_pack(v_sy0 & v_scale2, v_sy1 & v_scale2));
								v_uint16x8 v_v = v_shl<INTER_BITS>(v_sy8) | (v_sx8);
								v_store(A + x1, v_v);

								v_int16x8 v_d0 = v_pack(v_shr<INTER_BITS>(v_sx0), v_shr<INTER_BITS>(v_sx1));
								v_int16x8 v_d1 = v_pack(v_shr<INTER_BITS>(v_sy0), v_shr<INTER_BITS>(v_sy1));
								v_store_interleave(XY + (x1 << 1), v_d0, v_d1);
							}
						}
#endif
						for (; x1 < bcols; x1++)
						{
							int sx = cvRound(sX[x1] * INTER_TAB_SIZE);
							int sy = cvRound(sY[x1] * INTER_TAB_SIZE);
							int v = (sy & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE - 1));
							XY[x1 * 2] = saturate_cast<short>(sx >> INTER_BITS);
							XY[x1 * 2 + 1] = saturate_cast<short>(sy >> INTER_BITS);
							A[x1] = (ushort)v;
						}
					}
					else
					{
						const float* sXY = m1->ptr<float>(y + y1) + x * 2;
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
							XY[x1 * 2] = saturate_cast<short>(sx >> INTER_BITS);
							XY[x1 * 2 + 1] = saturate_cast<short>(sy >> INTER_BITS);
							A[x1] = (ushort)v;
						}
					}
				}
				ifunc(*src, dpart, bufxy, bufa, ctab, borderType, borderValue);
			}
		}
	}

private:
	const Mat* src;
	Mat* dst;
	const Mat* m1, * m2;
	int borderType;
	Scalar borderValue;
	int planar_input;
	RemapNNFunc nnfunc;
	RemapFunc ifunc;
	const void* ctab;
};