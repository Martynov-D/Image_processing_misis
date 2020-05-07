#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>


void first()
{
	IplImage* image = cvLoadImage("C:/Users/dimam/Documents/Admin/Programming/C++/OpenCV/OpenCVData/test_data/parrot.jpg", 0);

	int hist_size = 256;
	float range[] = { 0, 256 };
	float* histRange[] = { range };

	CvHistogram* hist;
	IplImage* hist_image = cvCreateImage(cvSize(1024, 720), 8, 1);
	hist = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, histRange, 1);
	cvCalcHist(&image, hist, 0, NULL);

	int bin_w;
	bin_w = cvRound((double)hist_image->width / hist_size);

	cvSet(hist_image, cvScalarAll(255), 0);
	for (int i{ 0 }; i < hist_size; ++i)
		cvRectangle
		(
			hist_image,
			cvPoint(i * bin_w, hist_image->height),
			cvPoint((i + 1) * bin_w, hist_image->height - cvRound(cvGetReal1D(hist->bins, i))),
			cvScalarAll(0),
			-1,
			8,
			0
		);

	// показываем результат
	cvNamedWindow("Gray", 1);
	cvShowImage("Gray", image);

	cvNamedWindow("Histogram", 1);
	cvShowImage("Histogram", hist_image);

	// ждём нажатия клавиши
	cvWaitKey(0);

	// освобождаем ресурсы
	cvReleaseImage(&image);
	cvReleaseImage(&hist_image);

	// удаляем окна
	cvDestroyAllWindows();
}

void second(cv::Mat image)
{
	// Преобразование в изображение в оттенках серого
	cv::Mat imageGray;
	cv::cvtColor(image, imageGray, CV_BGR2GRAY);

	// Вектор, который хранит монохроматичную картинку
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

	int hist_w{ 1280 }; // 512
	int hist_h{ 720 }; // 400
	int bin_w{ cvRound((double)hist_w / histSize) };

	// Создание пустых изображений для гистограмм
	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

	cv::Mat grayHistImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));

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

int main()
{
	// Загрузка изображения
	cv::Mat image{ cv::imread("C:/Users/dimam/Documents/Admin/Programming/C++/OpenCV/OpenCVData/test_data/parrot.jpg", 1) };
	if (!image.data)
		return -1;

	first();
	//second(image);

	return 0;
}