#include <opencv.hpp>
#include <cmath>

using namespace cv;

enum { XY_SHIFT = 16, XY_ONE = 1 << XY_SHIFT, DRAWING_STORAGE_BLOCK = (1 << 12) - 256 };

LineIterator::LineIterator(const Mat& img, Point pt1, Point pt2,
	int connectivity, bool left_to_right)
{
	count = -1;

	CV_Assert(connectivity == 8 || connectivity == 4);

	if ((unsigned)pt1.x >= (unsigned)(img.cols) ||
		(unsigned)pt2.x >= (unsigned)(img.cols) ||
		(unsigned)pt1.y >= (unsigned)(img.rows) ||
		(unsigned)pt2.y >= (unsigned)(img.rows))
	{
		if (!clipLine(img.size(), pt1, pt2))
		{
			ptr = img.data;
			err = plusDelta = minusDelta = plusStep = minusStep = count = 0;
			ptr0 = 0;
			step = 0;
			elemSize = 0;
			return;
		}
	}

	size_t bt_pix0 = img.elemSize(), bt_pix = bt_pix0;
	size_t istep = img.step;

	int dx = pt2.x - pt1.x;
	int dy = pt2.y - pt1.y;
	int s = dx < 0 ? -1 : 0;

	if (left_to_right)
	{
		dx = (dx ^ s) - s;
		dy = (dy ^ s) - s;
		pt1.x ^= (pt1.x ^ pt2.x) & s;
		pt1.y ^= (pt1.y ^ pt2.y) & s;
	}
	else
	{
		dx = (dx ^ s) - s;
		bt_pix = (bt_pix ^ s) - s;
	}

	ptr = (uchar*)(img.data + pt1.y * istep + pt1.x * bt_pix0);

	s = dy < 0 ? -1 : 0;
	dy = (dy ^ s) - s;
	istep = (istep ^ s) - s;

	s = dy > dx ? -1 : 0;

	/* conditional swaps */
	dx ^= dy & s;
	dy ^= dx & s;
	dx ^= dy & s;

	bt_pix ^= istep & s;
	istep ^= bt_pix & s;
	bt_pix ^= istep & s;

	if (connectivity == 8)
	{
		assert(dx >= 0 && dy >= 0);

		err = dx - (dy + dy);
		plusDelta = dx + dx;
		minusDelta = -(dy + dy);
		plusStep = (int)istep;
		minusStep = (int)bt_pix;
		count = dx + 1;
	}
	else /* connectivity == 4 */
	{
		assert(dx >= 0 && dy >= 0);

		err = 0;
		plusDelta = (dx + dx) + (dy + dy);
		minusDelta = -(dy + dy);
		plusStep = (int)(istep - bt_pix);
		minusStep = (int)bt_pix;
		count = dx + dy + 1;
	}

	this->ptr0 = img.ptr();
	this->step = (int)img.step;
	this->elemSize = (int)bt_pix0;
}

static void
Line(Mat& img, Point pt1, Point pt2,
	const void* _color, int connectivity = 8)
{
	if (connectivity == 0)
		connectivity = 8;
	else if (connectivity == 1)
		connectivity = 4;

	LineIterator iterator(img, pt1, pt2, connectivity, true);
	int i, count = iterator.count;
	int pix_size = (int)img.elemSize();
	const uchar* color = (const uchar*)_color;

	for (i = 0; i < count; i++, ++iterator)
	{
		uchar* ptr = *iterator;
		if (pix_size == 1)
			ptr[0] = std::max(color[0], ptr[0]);
		else if (pix_size == 3)
		{
			ptr[0] = std::max(color[0], ptr[0]);
			ptr[1] = std::max(color[1], ptr[1]);
			ptr[2] = std::max(color[2], ptr[2]);
		}
		else
			memcpy(*iterator, color, pix_size);
	}
}

static void
ThickLine(Mat& img, Point2l p0, Point2l p1, const void* color, int line_type)
{
	static const double INV_XY_ONE = 1. / XY_ONE;

	p0.x <<= XY_SHIFT;
	p0.y <<= XY_SHIFT;
	p1.x <<= XY_SHIFT;
	p1.y <<= XY_SHIFT;

	p0.x = (p0.x + (XY_ONE >> 1)) >> XY_SHIFT;
	p0.y = (p0.y + (XY_ONE >> 1)) >> XY_SHIFT;
	p1.x = (p1.x + (XY_ONE >> 1)) >> XY_SHIFT;
	p1.y = (p1.y + (XY_ONE >> 1)) >> XY_SHIFT;
	Line(img, p0, p1, color, line_type);
}

extern "C" __declspec(dllexport)
void aggerate_line(uchar* pimg, int r, int c, int* p1, int* p2, uchar* color, int line_type);

void aggerate_line(uchar* pimg, int r, int c, int* p1, int* p2, uchar* color, int line_type)
{
	Mat img = Mat(r, c, CV_8U, pimg);
	Point pt1 = Point(p1[0], p1[1]);
	Point pt2 = Point(p2[0], p2[1]);
	ThickLine(img, pt1, pt2, color, line_type);
}