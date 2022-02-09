class ParallelLoopBodyLambdaWrapper : public ParallelLoopBody
{
private:
	std::function<void(const Range&)> m_functor;
public:
	inline
		ParallelLoopBodyLambdaWrapper(std::function<void(const Range&)> functor)
		: m_functor(functor)
	{
		// nothing
	}

	virtual void operator() (const cv::Range& range) const CV_OVERRIDE
	{
		m_functor(range);
	}
};

//! @ingroup core_parallel
static inline
void parallel_for_(const Range& range, std::function<void(const Range&)> functor, double nstripes = -1.)
{
	parallel_for_(range, ParallelLoopBodyLambdaWrapper(functor), nstripes);
}



class CV_EXPORTS ParallelLoopBody
{
public:
	virtual ~ParallelLoopBody();
	virtual void operator() (const Range& range) const = 0;
};