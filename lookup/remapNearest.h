#pragma once
static inline int clip(int x, int a, int b)
{
	return x >= a ? (x < b ? x : b - 1) : a;
}
template<typename T>
static void remapNearest(const Matf& _src, Matf& _dst, const Matf& _xy,
	int borderType, const void* _borderValue)
{
	Size ssize = _src.size(), dsize = _dst.size();
	const int cn = _src.channels();
	const T* S0 = (T*)_src.ptr();
	T cval[CV_CN_MAX];
	size_t sstep = _src.step() / sizeof(S0[0]);

	for (int k = 0; k < cn; k++)
		cval[k] = static_cast<T>(_borderValue[k & 3]);

	unsigned width1 = ssize.width, height1 = ssize.height;

	if (_dst.isContinuous() && _xy.isContinuous())
	{
		dsize.width *= dsize.height;
		dsize.height = 1;
	}

	for (int dy = 0; dy < dsize.height; dy++)
	{
		T* D = (T*)_dst.ptr(dy);
		const short* XY = (short *)_xy.ptr(dy);

		if (cn == 1)
		{
			for (int dx = 0; dx < dsize.width; dx++)
			{
				int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
				if ((unsigned)sx < width1 && (unsigned)sy < height1)
					D[dx] = S0[sy * sstep + sx];
				else
				{
					if (borderType == BORDER_REPLICATE)
					{
						sx = clip(sx, 0, ssize.width);
						sy = clip(sy, 0, ssize.height);
						D[dx] = S0[sy * sstep + sx];
					}
					else if (borderType == BORDER_CONSTANT)
						D[dx] = cval[0];
					/*else if (borderType != BORDER_TRANSPARENT)
					{
						sx = borderInterpolate(sx, ssize.width, borderType);
						sy = borderInterpolate(sy, ssize.height, borderType);
						D[dx] = S0[sy * sstep + sx];
					}*/
				}
			}
		}
		else
		{
			for (int dx = 0; dx < dsize.width; dx++, D += cn)
			{
				int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
				const T* S;
				if ((unsigned)sx < width1 && (unsigned)sy < height1)
				{
					if (cn == 3)
					{
						S = S0 + sy * sstep + sx * 3;
						D[0] = S[0], D[1] = S[1], D[2] = S[2];
					}
					else if (cn == 4)
					{
						S = S0 + sy * sstep + sx * 4;
						D[0] = S[0], D[1] = S[1], D[2] = S[2], D[3] = S[3];
					}
					else
					{
						S = S0 + sy * sstep + sx * cn;
						for (int k = 0; k < cn; k++)
							D[k] = S[k];
					}
				}
				else if (borderType != BORDER_TRANSPARENT)
				{
					if (borderType == BORDER_REPLICATE)
					{
						sx = clip(sx, 0, ssize.width);
						sy = clip(sy, 0, ssize.height);
						S = S0 + sy * sstep + sx * cn;
					}
					else if (borderType == BORDER_CONSTANT)
						S = &cval[0];
					//else
					//{
					//	sx = borderInterpolate(sx, ssize.width, borderType);
					//	sy = borderInterpolate(sy, ssize.height, borderType);
					//	S = S0 + sy * sstep + sx * cn;
					//}
					for (int k = 0; k < cn; k++)
						D[k] = S[k];
				}
			}
		}
	}
}
