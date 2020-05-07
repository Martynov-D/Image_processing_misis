#include <opencv2/opencv.hpp>
#include <vector>

void drawGrayScale(cv::Mat& img, int clStep, int brStep, int lvl, int width)
{
	for (int i{ 0 }; i < width; i += clStep)
	{
		// ������ ������� ������ � ���� �������������� HEIGHTxSTEP
		cv::rectangle(img, cv::Point(i, 0), cv::Point((i + clStep), img.rows), cv::Scalar(lvl, lvl, lvl), CV_FILLED);
		lvl += brStep;
	}
}

cv::Mat gammaCorrection(const cv::Mat& img)
{
	cv::Mat G;

	// �������������� � 64 ������ float
	img.convertTo(G, CV_64F);

	// ������������ � ���������� �����-���������
	G /= 255;
	cv::pow(G, 2.3, G);
	G *= 255;

	// ��������������� ������� � 8 ������ uchar
	G.convertTo(G, CV_8UC1);

	return G;
}

int main()
{
	// �������� ������ {0..255}
	int lvl{ 0 };
	int lvl2{ 5 };
	// ��� � ������� �������� �������� ������
	int brStep{ 1 };
	int brStep2{ 10 };
	// ������� � �������� �������� ���� �������� ������
	int step{ 3 };
	int step2{ 30 };

	// ������� ���� � �������� (768x60)
	int width{ 256 * step };
	int height{ 60 };

	cv::Mat img(height, width, CV_8UC1);
	cv::Mat img2(height, width, CV_8UC1);

	drawGrayScale(img, step, brStep, lvl, width);
	drawGrayScale(img2, step2, brStep2, lvl2, width);

	cv::Mat G1 = gammaCorrection(img);
	cv::Mat G2 = gammaCorrection(img2);

	// ���������� ���� �������� � ���� -----------------------------------------------

	cv::Mat allImages(height * 4, width, CV_8UC1);

	// ��������� �������� ���� ��� �������
	std::vector<cv::Mat>split;
	for (int i{ 0 }; i < 4; ++i)
	{
		split.push_back(cv::Mat(allImages, cv::Rect(0, i * height, width, height)));
	}

	// ������ �� ����� ��������
	std::vector<cv::Mat> temp;
	temp.push_back(img);
	temp.push_back(G1);
	temp.push_back(img2);
	temp.push_back(G2);

	// ����������� �������� � ���������� �������
	for (int i{ 0 }; i < 4; ++i)
	{
		temp[i].copyTo(split[i]);
	}

	// -------------------------------------------------------------------------------
	std::string writeTo{ "C:/Users/dimam/Documents/Admin/Programming/C++/OpenCV/OpenCVData/test_data/" };

	cv::imwrite(writeTo + "All images.png", allImages, { CV_IMWRITE_PNG_COMPRESSION, 0 });

	/*cv::namedWindow("All images", CV_WINDOW_AUTOSIZE);
	cv::imshow("All images", allImages);

	cv::waitKey(0);

	cv::destroyAllWindows;*/
}