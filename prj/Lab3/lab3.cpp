#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>


void grayAndColorHist(cv::Mat image)
{
	// Преобразование в изображение в оттенках серого
	cv::Mat imageGray;
	cv::cvtColor(image, imageGray, CV_BGR2GRAY);

	// Вектор, который хранит разложение на каналы
	std::vector<cv::Mat> bgrPlanes;
	cv::split(image, bgrPlanes);

	int histSize{ 256 };
	float range[] = { 0,256 };
	const float* histRange = { range };

	bool uniform{ true };
	bool accumulate{ false };

	cv::Mat b_hist;
	cv::Mat g_hist;
	cv::Mat r_hist;

	cv::Mat gray_hist;

	// Рассчет гистограмм
	calcHist(&bgrPlanes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgrPlanes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgrPlanes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	calcHist(&imageGray, 1, 0, cv::Mat(), gray_hist, 1, &histSize, &histRange, uniform, accumulate);

	int hist_w{ 512 }; // 512
	int hist_h{ 400 }; // 400
	int bin_w{ cvRound((double)hist_w / histSize) };

	// Создание пустых изображений для гистограмм
	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

	cv::Mat grayHistImage(hist_h, hist_w, CV_8UC1, 255);

	// Нормализация гистограмм под размер окна
	cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

	cv::normalize(gray_hist, gray_hist, 0, grayHistImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

	// Построение гистограмм
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
			cv::Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
			cv::Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
			cv::Scalar(0, 0, 255), 2, 8, 0);

		line(grayHistImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(gray_hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(gray_hist.at<float>(i))),
			cv::Scalar(0, 0, 0), 2, 8, 0);
	}

	/// Отображение результатов
	cv::namedWindow("Original image", CV_WINDOW_AUTOSIZE);
	cv::imshow("Original image", image);

	cv::namedWindow("Hist", CV_WINDOW_AUTOSIZE);
	cv::imshow("Hist", histImage);

	cv::namedWindow("Original image in gray", CV_WINDOW_AUTOSIZE);
	cv::imshow("Original image in gray", imageGray);

	cv::namedWindow("Gray Hist", CV_WINDOW_AUTOSIZE);
	cv::imshow("Gray Hist", grayHistImage);

	cv::waitKey(0);
}

void lab2()
{
	cv::Mat image = cv::imread("C:/Admin/Programming/C++/OpenCV/data/test_data/parrot.jpg", CV_LOAD_IMAGE_COLOR);

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

	// Отображение результатов
	cv::namedWindow("Original image in gray", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("Hist", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Bright image", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Bright Hist", cv::WINDOW_AUTOSIZE);

	cv::imshow("Original image in gray", imageGray);
	cv::imshow("Hist", hist_window);
	cv::imshow("Bright image", imageGrayB);
	cv::imshow("Bright Hist", histBright_window);

	cv::waitKey(0);

	cv::destroyAllWindows();
}

int main()
{
	// Загрузка изображения
	cv::Mat image{ cv::imread("C:/Admin/Programming/C++/OpenCV/data/test_data/parrot.jpg", 1) };
	if (!image.data)
		return -1;

	//grayAndColorHist(image);
	lab2();

	return 0;
}