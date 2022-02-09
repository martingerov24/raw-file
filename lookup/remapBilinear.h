emplate<class CastOp, class VecOp, typename AT>
static void remapBilinear(const Mat& _src, Mat& _dst, const Mat& _xy,
	const Mat& _fxy, const void* _wtab,
	int borderType, const Scalar& _borderValue)
{
	typedef typename CastOp::rtype T;
	typedef typename CastOp::type1 WT;
	Size ssize = _src.size(), dsize = _dst.size();
	const int cn = _src.channels();
	const AT* wtab = (const AT*)_wtab;
	const T* S0 = _src.ptr<T>();
	size_t sstep = _src.step / sizeof(S0[0]);
	T cval[CV_CN_MAX];
	CastOp castOp;
	VecOp vecOp;

	for (int k = 0; k < cn; k++)
		cval[k] = saturate_cast<T>(_borderValue[k & 3]);

	unsigned width1 = std::max(ssize.width - 1, 0), height1 = std::max(ssize.height - 1, 0);
	CV_Assert(!ssize.empty());
#if CV_SIMD128
	if (_src.type() == CV_8UC3)
		width1 = std::max(ssize.width - 2, 0);
#endif

	for (int dy = 0; dy < dsize.height; dy++)
	{
		T* D = _dst.ptr<T>(dy);
		const short* XY = _xy.ptr<short>(dy);
		const ushort* FXY = _fxy.ptr<ushort>(dy);
		int X0 = 0;
		bool prevInlier = false;

		for (int dx = 0; dx <= dsize.width; dx++)
		{
			bool curInlier = dx < dsize.width ?
				(unsigned)XY[dx * 2] < width1 &&
				(unsigned)XY[dx * 2 + 1] < height1 : !prevInlier;
			if (curInlier == prevInlier)
				continue;

			int X1 = dx;
			dx = X0;
			X0 = X1;
			prevInlier = curInlier;

			if (!curInlier)
			{
				int len = vecOp(_src, D, XY + dx * 2, FXY + dx, wtab, X1 - dx);
				D += len * cn;
				dx += len;

				if (cn == 1)
				{
					for (; dx < X1; dx++, D++)
					{
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						const AT* w = wtab + FXY[dx] * 4;
						const T* S = S0 + sy * sstep + sx;
						*D = castOp(WT(S[0] * w[0] + S[1] * w[1] + S[sstep] * w[2] + S[sstep + 1] * w[3]));
					}
				}
				else if (cn == 2)
					for (; dx < X1; dx++, D += 2)
					{
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						const AT* w = wtab + FXY[dx] * 4;
						const T* S = S0 + sy * sstep + sx * 2;
						WT t0 = S[0] * w[0] + S[2] * w[1] + S[sstep] * w[2] + S[sstep + 2] * w[3];
						WT t1 = S[1] * w[0] + S[3] * w[1] + S[sstep + 1] * w[2] + S[sstep + 3] * w[3];
						D[0] = castOp(t0); D[1] = castOp(t1);
					}
				else if (cn == 3)
					for (; dx < X1; dx++, D += 3)
					{
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						const AT* w = wtab + FXY[dx] * 4;
						const T* S = S0 + sy * sstep + sx * 3;
						WT t0 = S[0] * w[0] + S[3] * w[1] + S[sstep] * w[2] + S[sstep + 3] * w[3];
						WT t1 = S[1] * w[0] + S[4] * w[1] + S[sstep + 1] * w[2] + S[sstep + 4] * w[3];
						WT t2 = S[2] * w[0] + S[5] * w[1] + S[sstep + 2] * w[2] + S[sstep + 5] * w[3];
						D[0] = castOp(t0); D[1] = castOp(t1); D[2] = castOp(t2);
					}
				else if (cn == 4)
					for (; dx < X1; dx++, D += 4)
					{
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						const AT* w = wtab + FXY[dx] * 4;
						const T* S = S0 + sy * sstep + sx * 4;
						WT t0 = S[0] * w[0] + S[4] * w[1] + S[sstep] * w[2] + S[sstep + 4] * w[3];
						WT t1 = S[1] * w[0] + S[5] * w[1] + S[sstep + 1] * w[2] + S[sstep + 5] * w[3];
						D[0] = castOp(t0); D[1] = castOp(t1);
						t0 = S[2] * w[0] + S[6] * w[1] + S[sstep + 2] * w[2] + S[sstep + 6] * w[3];
						t1 = S[3] * w[0] + S[7] * w[1] + S[sstep + 3] * w[2] + S[sstep + 7] * w[3];
						D[2] = castOp(t0); D[3] = castOp(t1);
					}
				else
					for (; dx < X1; dx++, D += cn)
					{
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						const AT* w = wtab + FXY[dx] * 4;
						const T* S = S0 + sy * sstep + sx * cn;
						for (int k = 0; k < cn; k++)
						{
							WT t0 = S[k] * w[0] + S[k + cn] * w[1] + S[sstep + k] * w[2] + S[sstep + k + cn] * w[3];
							D[k] = castOp(t0);
						}
					}
			}
			else
			{
				if (borderType == BORDER_TRANSPARENT && cn != 3)
				{
					D += (X1 - dx) * cn;
					dx = X1;
					continue;
				}

				if (cn == 1)
					for (; dx < X1; dx++, D++)
					{
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						if (borderType == BORDER_CONSTANT &&
							(sx >= ssize.width || sx + 1 < 0 ||
								sy >= ssize.height || sy + 1 < 0))
						{
							D[0] = cval[0];
						}
						else
						{
							int sx0, sx1, sy0, sy1;
							T v0, v1, v2, v3;
							const AT* w = wtab + FXY[dx] * 4;
							if (borderType == BORDER_REPLICATE)
							{
								sx0 = clip(sx, 0, ssize.width);
								sx1 = clip(sx + 1, 0, ssize.width);
								sy0 = clip(sy, 0, ssize.height);
								sy1 = clip(sy + 1, 0, ssize.height);
								v0 = S0[sy0 * sstep + sx0];
								v1 = S0[sy0 * sstep + sx1];
								v2 = S0[sy1 * sstep + sx0];
								v3 = S0[sy1 * sstep + sx1];
							}
							else
							{
								sx0 = borderInterpolate(sx, ssize.width, borderType);
								sx1 = borderInterpolate(sx + 1, ssize.width, borderType);
								sy0 = borderInterpolate(sy, ssize.height, borderType);
								sy1 = borderInterpolate(sy + 1, ssize.height, borderType);
								v0 = sx0 >= 0 && sy0 >= 0 ? S0[sy0 * sstep + sx0] : cval[0];
								v1 = sx1 >= 0 && sy0 >= 0 ? S0[sy0 * sstep + sx1] : cval[0];
								v2 = sx0 >= 0 && sy1 >= 0 ? S0[sy1 * sstep + sx0] : cval[0];
								v3 = sx1 >= 0 && sy1 >= 0 ? S0[sy1 * sstep + sx1] : cval[0];
							}
							D[0] = castOp(WT(v0 * w[0] + v1 * w[1] + v2 * w[2] + v3 * w[3]));
						}
					}
				else
					for (; dx < X1; dx++, D += cn)
					{
						int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
						if (borderType == BORDER_CONSTANT &&
							(sx >= ssize.width || sx + 1 < 0 ||
								sy >= ssize.height || sy + 1 < 0))
						{
							for (int k = 0; k < cn; k++)
								D[k] = cval[k];
						}
						else
						{
							int sx0, sx1, sy0, sy1;
							const T* v0, * v1, * v2, * v3;
							const AT* w = wtab + FXY[dx] * 4;
							if (borderType == BORDER_REPLICATE)
							{
								sx0 = clip(sx, 0, ssize.width);
								sx1 = clip(sx + 1, 0, ssize.width);
								sy0 = clip(sy, 0, ssize.height);
								sy1 = clip(sy + 1, 0, ssize.height);
								v0 = S0 + sy0 * sstep + sx0 * cn;
								v1 = S0 + sy0 * sstep + sx1 * cn;
								v2 = S0 + sy1 * sstep + sx0 * cn;
								v3 = S0 + sy1 * sstep + sx1 * cn;
							}
							else if (borderType == BORDER_TRANSPARENT &&
								((unsigned)sx >= (unsigned)(ssize.width - 1) ||
									(unsigned)sy >= (unsigned)(ssize.height - 1)))
								continue;
							else
							{
								sx0 = borderInterpolate(sx, ssize.width, borderType);
								sx1 = borderInterpolate(sx + 1, ssize.width, borderType);
								sy0 = borderInterpolate(sy, ssize.height, borderType);
								sy1 = borderInterpolate(sy + 1, ssize.height, borderType);
								v0 = sx0 >= 0 && sy0 >= 0 ? S0 + sy0 * sstep + sx0 * cn : &cval[0];
								v1 = sx1 >= 0 && sy0 >= 0 ? S0 + sy0 * sstep + sx1 * cn : &cval[0];
								v2 = sx0 >= 0 && sy1 >= 0 ? S0 + sy1 * sstep + sx0 * cn : &cval[0];
								v3 = sx1 >= 0 && sy1 >= 0 ? S0 + sy1 * sstep + sx1 * cn : &cval[0];
							}
							for (int k = 0; k < cn; k++)
								D[k] = castOp(WT(v0[k] * w[0] + v1[k] * w[1] + v2[k] * w[2] + v3[k] * w[3]));
						}
					}
			}
		}
	}
}