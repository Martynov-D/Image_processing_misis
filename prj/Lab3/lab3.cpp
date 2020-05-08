#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

void drawGrayScale(cv::Mat& img, int clStep, int brStep, int lvl, int width)
{
	for (int i{ 0 }; i < width; i += clStep)
	{
		// Рисуем оттенок серого в виде прямоугольника HEIGHTxSTEP
		cv::rectangle(img, cv::Point(i, 0), cv::Point((i + clStep), img.rows), cv::Scalar(lvl, lvl, lvl), CV_FILLED);
		lvl += brStep;
	}
}

int main()
{
	// Загрузка изображения
	cv::Mat image{ cv::imread("C:/Admin/Programming/C++/OpenCV/data/test_data/AC_Valhalla.jpg", 1) };
	if (!image.data)
		return -1;

	// Повышение яркости изображения
	cv::Mat imageB = image + cv::Scalar(30, 30, 30);

	// Преобразование в изображение в оттенках серого
	cv::Mat imageGray;
	cv::cvtColor(image, imageGray, CV_BGR2GRAY);

	cv::Mat imageGrayB;
	cv::cvtColor(imageB, imageGrayB, CV_BGR2GRAY);

	int histSize{ 256 };
	float range[] = { 0,256 };
	const float* histRange = { range };

	bool uniform{ true };
	bool accumulate{ false };

	int histWidth{ 512 }; // 512
	int histHeight{ 400 }; // 400
	int bin_w{ cvRound((double)histWidth / histSize) };

	// Создание пустых изображений для гистограмм
	cv::Mat hist_window(histHeight, histWidth, CV_8UC1, 255); // cv::Scalar(255, 255, 255)
	cv::Mat histBright_window(histHeight, histWidth, CV_8UC1, 255);

	// Рассчет гистограмм
	cv::Mat hist;
	cv::Mat histBright;
	calcHist(&imageGray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&imageGrayB, 1, 0, cv::Mat(), histBright, 1, &histSize, &histRange, uniform, accumulate);

	// Нормализация гистограмм под размер окна
	cv::normalize(hist, hist, 0, hist_window.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(histBright, histBright, 0, histBright_window.rows, cv::NORM_MINMAX, -1, cv::Mat());

	// Построение гистограмм
	for (int i = 1; i < histSize; ++i)
	{
		line(hist_window, cv::Point(bin_w * (i - 1), histHeight - cvRound(hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), histHeight - cvRound(hist.at<float>(i))),
			cv::Scalar(0, 0, 0), 2, 8, 0);
		line(histBright_window, cv::Point(bin_w * (i - 1), histHeight - cvRound(histBright.at<float>(i - 1))),
			cv::Point(bin_w * (i), histHeight - cvRound(histBright.at<float>(i))),
			cv::Scalar(0, 0, 0), 2, 8, 0);
	}

	// Соединение гистограмм с полоской градаций серого
	cv::Mat grayScale(60, histWidth, CV_8UC1);
	drawGrayScale(grayScale, 2, 1, 0, histWidth);

	cv::Mat updHist(histHeight + 60, histWidth, CV_8UC1);
	cv::Mat updBrightHist(histHeight + 60, histWidth, CV_8UC1);

	std::vector<cv::Mat>split;
	split.push_back(cv::Mat(updHist, cv::Rect(0, 0, histWidth, histHeight)));
	split.push_back(cv::Mat(updHist, cv::Rect(0, histHeight, histWidth, 60)));
	split.push_back(cv::Mat(updBrightHist, cv::Rect(0, 0, histWidth, histHeight)));
	split.push_back(cv::Mat(updBrightHist, cv::Rect(0, histHeight, histWidth, 60)));

	std::vector<cv::Mat> temp;
	temp.push_back(hist_window);
	temp.push_back(grayScale);
	temp.push_back(histBright_window);
	temp.push_back(grayScale);

	for (int i{ 0 }; i < 2; ++i)
	{
		temp[i].copyTo(split[i]);
		temp[i + 2].copyTo(split[i + 2]);
	}

	// Отображение результатов

	cv::imshow("Original image in gray", imageGray);
	cv::imshow("Bright image", imageGrayB);
	cv::imshow("Hist", updHist);
	cv::imshow("Bright Hist", updBrightHist);

	cv::waitKey(0);

	cv::destroyAllWindows();

	return 0;
}