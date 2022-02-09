atic const void* initInterTab2D(int method, bool fixpt)
{
	static bool inittab[INTER_MAX + 1] = { false };
	float* tab = 0;
	short* itab = 0;
	int ksize = 0;
	if (method == INTER_LINEAR)
		tab = BilinearTab_f[0][0], itab = BilinearTab_i[0][0], ksize = 2;
	else if (method == INTER_CUBIC)
		tab = BicubicTab_f[0][0], itab = BicubicTab_i[0][0], ksize = 4;
	else if (method == INTER_LANCZOS4)
		tab = Lanczos4Tab_f[0][0], itab = Lanczos4Tab_i[0][0], ksize = 8;
	else
		CV_Error(CV_StsBadArg, "Unknown/unsupported interpolation type");

	if (!inittab[method])
	{
		AutoBuffer<float> _tab(8 * INTER_TAB_SIZE);
		int i, j, k1, k2;
		initInterTab1D(method, _tab.data(), INTER_TAB_SIZE);
		for (i = 0; i < INTER_TAB_SIZE; i++)
			for (j = 0; j < INTER_TAB_SIZE; j++, tab += ksize * ksize, itab += ksize * ksize)
			{
				int isum = 0;
				NNDeltaTab_i[i * INTER_TAB_SIZE + j][0] = j < INTER_TAB_SIZE / 2;
				NNDeltaTab_i[i * INTER_TAB_SIZE + j][1] = i < INTER_TAB_SIZE / 2;

				for (k1 = 0; k1 < ksize; k1++)
				{
					float vy = _tab[i * ksize + k1];
					for (k2 = 0; k2 < ksize; k2++)
					{
						float v = vy * _tab[j * ksize + k2];
						tab[k1 * ksize + k2] = v;
						isum += itab[k1 * ksize + k2] = saturate_cast<short>(v * INTER_REMAP_COEF_SCALE);
					}
				}

				if (isum != INTER_REMAP_COEF_SCALE)
				{
					int diff = isum - INTER_REMAP_COEF_SCALE;
					int ksize2 = ksize / 2, Mk1 = ksize2, Mk2 = ksize2, mk1 = ksize2, mk2 = ksize2;
					for (k1 = ksize2; k1 < ksize2 + 2; k1++)
						for (k2 = ksize2; k2 < ksize2 + 2; k2++)
						{
							if (itab[k1 * ksize + k2] < itab[mk1 * ksize + mk2])
								mk1 = k1, mk2 = k2;
							else if (itab[k1 * ksize + k2] > itab[Mk1 * ksize + Mk2])
								Mk1 = k1, Mk2 = k2;
						}
					if (diff < 0)
						itab[Mk1 * ksize + Mk2] = (short)(itab[Mk1 * ksize + Mk2] - diff);
					else
						itab[mk1 * ksize + mk2] = (short)(itab[mk1 * ksize + mk2] - diff);
				}
			}
		tab -= INTER_TAB_SIZE2 * ksize * ksize;
		itab -= INTER_TAB_SIZE2 * ksize * ksize;
#if CV_SIMD128
		if (method == INTER_LINEAR)
		{
			for (i = 0; i < INTER_TAB_SIZE2; i++)
				for (j = 0; j < 4; j++)
				{
					BilinearTab_iC4[i][0][j * 2] = BilinearTab_i[i][0][0];
					BilinearTab_iC4[i][0][j * 2 + 1] = BilinearTab_i[i][0][1];
					BilinearTab_iC4[i][1][j * 2] = BilinearTab_i[i][1][0];
					BilinearTab_iC4[i][1][j * 2 + 1] = BilinearTab_i[i][1][1];
				}
		}
#endif
		inittab[method] = true;
	}
	return fixpt ? (const void*)itab : (const void*)tab;
}

#ifndef __MINGW32__
static bool initAllInterTab2D()
{
	return  initInterTab2D(INTER_LINEAR, false) &&
		initInterTab2D(INTER_LINEAR, true) &&
		initInterTab2D(INTER_CUBIC, false) &&
		initInterTab2D(INTER_CUBIC, true) &&
		initInterTab2D(INTER_LANCZOS4, false) &&
		initInterTab2D(INTER_LANCZOS4, true);
}