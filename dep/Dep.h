#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>

template<typename T>
struct Point2
{
	Point2() = default;
	Point2(T x, T y)
		:x(x), y(y) {};
	T x;
	T y;
};

typedef Point2<float>   Point2f;
typedef Point2<int32_t> Point2i;


template<typename T>
struct Point3
{
	Point3(T x, T y, T z)
		:x(x), y(y), z(z) {};
	T x;
	T y;
	T z;
};

typedef Point3<float> Point3f;
typedef Point3<int32_t> Point3i;
#pragma once

#include <cstdint>


template<class T>
class Mat
{
public:

	Mat() : m_cols(0), m_rows(0), m_step(0), m_matrix(nullptr), m_channels(0), m_owner(true) {}

	Mat(int x, int y, std::vector<T> &matrix)
	{
		m_cols = x;
		m_rows = y;
		m_channels = 1;
		std::swap(m_matrix, matrix);
		m_step = m_channels * m_cols * sizeof(T);
		m_owner = false;
	}

	Mat(int x, int y, int other_channels = 1)
	{
		create(x, y, other_channels);
	}

	Mat(const Mat& other)
	{
		*this = other;
		return *this;
	};

	Mat& operator=(const Mat& other)
	{
		*this = other;
		return *this;
	}

	Mat& operator=(Mat&& other)
	{
		std::swap(*this, other);
		return *this;
	}

	Mat(Mat&& other) 
	{
		m_cols = other.m_cols;
		m_rows = other.m_rows;
		m_step = other.m_step;
		m_channels = other.m_channels;
		m_owner = other.m_owner;
		std::swap(m_matrix, other.m_matrix);


		other.m_cols = 0;
		other.m_rows = 0;
		other.m_step = 0;
		other.m_channels = 0;
		other.m_owner = false;
	}

	void create(int x, int y, int other_channels = 1)
	{
		m_cols = x;
		m_rows = y;
		m_channels = other_channels;
		m_owner = true;
	
		//mm::deallocate(m_matrix);
		const uint64_t bytesPerLine = m_channels * m_cols * sizeof(T);
		m_step = bytesPerLine;
		m_matrix(m_cols * m_rows * m_channels, 0); // we create a matrix with size of the first, filled with 0
		//mm::allocate(reinterpret_cast<void**>(&m_matrix), uint64_t(m_step) * m_rows);
	}

	T& operator[](int where)
	{
		return m_matrix[where];
	}
	int size()
	{
		return m_cols * m_rows;
	}
	bool hasData()
	{
		return this->m_matrix.size() == 0 ? 1 : 0;
	}
	void createPitched(int x, int y) {
		m_cols = x;
		m_rows = y;
		m_channels = 1;
		m_owner = true;

		//mm::deallocate(m_matrix);

		// The value of texturePitchAlignment should match the value returned by cudaDeviceProp::textureAlignment
		static const uint64_t texturePitchAlignment = 512;
		const uint64_t bytesPerLine = m_channels * m_cols * sizeof(T);
		uint64_t pitch = bytesPerLine;
		if (bytesPerLine % texturePitchAlignment != 0) {
			pitch = (bytesPerLine + texturePitchAlignment - 1) / texturePitchAlignment;
			pitch *= texturePitchAlignment;
		}
		m_step = pitch;

		//mm::allocate(reinterpret_cast<void**>(&m_matrix), uint64_t(m_step) * m_rows);
	}

	~Mat() {
		if (m_owner) {
			//mm::deallocate(m_matrix);
			m_matrix = nullptr;
		}
	}

	int32_t cols() const { return m_cols; }
	int32_t rows() const { return m_rows; }
	int32_t step() const { return m_step; }
	int32_t channels() const { return m_channels; }

	T* ptr(int y = 0) { return reinterpret_cast<T*>(reinterpret_cast<char*>(m_matrix) + y * step()); }
	const T* ptr(int y = 0) const { return reinterpret_cast<const T*>(reinterpret_cast<const char*>(m_matrix) + y * step()); }

	T& operator ()(int y, int x) { return ptr(y)[x]; }
	const T& operator ()(int y, int x) const { return ptr(y)[x]; }

	Mat& operator*=(const float a) noexcept
	{
		for (int i = 0; i < m_rows; i++) {
			T* p = ptr(i);
			for (int j = 0; j < m_cols; j++) {
				p[j] *= a;
			}
		}
		return *this;
	}

private:
	std::vector<T> m_matrix;
	//T* m_matrix;
	int32_t m_cols;
	int32_t m_rows;
	int32_t m_step;
	int32_t m_channels;
	bool m_owner;
};
template<typename T>
inline bool isNull(const Mat<T>& mat)
{
	return mat.m_matrix.size() == 0 ? 1 : 0;
}
template<typename T>
Mat<T> inline eye(int m, int n) //TODO: if that does not work, try removing the inline
{
	Mat<T>M(m, n);
	for (int i = 0; i < std::min(m, n); i++)
	{
		M[n * i + i] = 1; // if i am not mistaken those are the maths
	}
	return std::move(M);
}
typedef Mat<float> Matf;
typedef Mat<int> Mati;
typedef Mat<int16_t> Mat16s;
typedef Mat<uint16_t> Mat16u;

template<class T>
class MatView
{
public:

	MatView(int x, int y, int step, const T* ptr) : m_cols(x), m_rows(y), m_step(step), m_matrix(const_cast<T*>(ptr))
	{
	}

	~MatView() = default;

	__host__ __device__ int32_t rows() const { return m_rows; }
	__host__ __device__ int32_t cols() const { return m_cols; }
	__host__ __device__ int32_t step() const { return m_step; }


	__host__ __device__ T* ptr(int y = 0) { return reinterpret_cast<T*>(reinterpret_cast<char*>(m_matrix) + y * step()); }
	__host__ __device__ const T* ptr(int y = 0) const { return reinterpret_cast<const T*>(reinterpret_cast<const char*>(m_matrix) + y * step()); }

	__host__ __device__ T& operator ()(int y, int x) { return ptr(y)[x]; }
	__host__ __device__ const T& operator ()(int y, int x) const { return ptr(y)[x]; }

private:
	int32_t m_cols;
	int32_t m_rows;
	int32_t m_step;
	T* m_matrix;
};