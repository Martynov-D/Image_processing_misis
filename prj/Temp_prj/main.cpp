#include <opencv2/opencv.hpp>
#include <vector>
#include <array>

void tempTask()
{
	// READ RGB color image and convert it to Lab
	cv::Mat bgr_image = cv::imread("C:/Users/dimam/Documents/Admin/Programming/C++/OpenCV/OpenCVData/test_data/parrot.jpg");
	cv::Mat lab_image;
	cv::cvtColor(bgr_image, lab_image, CV_BGR2Lab);

	// Extract the L channel
	std::vector<cv::Mat> lab_planes(3);
	cv::split(lab_image, lab_planes); // now we have the L image in lab_planes[0]

	// apply the CLAHE algorithm to the L channel
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(4);
	cv::Mat dst;
	clahe->apply(lab_planes[0], dst);

	// Merge the the color planes back into an Lab image
	dst.copyTo(lab_planes[0]);
	cv::merge(lab_planes, lab_image);

	// convert back to RGB
	cv::Mat image_clahe;
	cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);

	// display the results (you might also want to see lab_planes[0] before and after).
	cv::imshow("image original", bgr_image);
	cv::imshow("image CLAHE", image_clahe);
	cv::waitKey();
	cv::destroyAllWindows();
}

void second()
{
	cv::Mat image = cv::imread("C:/Users/dimam/Documents/Admin/Programming/C++/OpenCV/OpenCVData/test_data/parrot.jpg", CV_LOAD_IMAGE_COLOR);

	// Преобразование в изображение в оттенках серого
	cv::Mat imageGray;
	cv::cvtColor(image, imageGray, CV_BGR2GRAY);

	// Повышение яркости изображения
	cv::Mat imageB = image + cv::Scalar(30, 30, 30);

	cv::Mat imageGrayB;
	cv::cvtColor(imageB, imageGrayB, CV_BGR2GRAY);

	int histSize{ 256 };
	float range[] = { 0,256 };
	const float* histRange = { range };

	bool uniform{ true };
	bool accumulate{ false };

	int hist_w{ 512 }; // 512
	int hist_h{ 400 }; // 400
	int bin_w{ cvRound((double)hist_w / histSize) };

	// Создание пустых изображений для гистограмм
	cv::Mat hist_window(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat histBright_window(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));

	// Рассчет гистограмм
	cv::Mat hist;
	cv::Mat hist_bright;
	calcHist(&imageGray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&imageGrayB, 1, 0, cv::Mat(), hist_bright, 1, &histSize, &histRange, uniform, accumulate);

	// Нормализация гистограмм под размер окна
	cv::normalize(hist, hist, 0, hist_window.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(hist_bright, hist_bright, 0, histBright_window.rows, cv::NORM_MINMAX, -1, cv::Mat());

	// Построение гистограмм
	for (int i = 1; i < histSize; ++i)
	{
		line(hist_window, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
			cv::Scalar(0, 0, 0), 2, 8, 0);
		line(histBright_window, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist_bright.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(hist_bright.at<float>(i))),
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

std::vector<cv::Mat> absDifference(const std::vector<cv::Mat>& img, const std::vector<cv::Mat>& rgb)
{
	std::vector<cv::Mat> d;
	for (int i{ 0 }; i < 3; ++i)
	{
		cv::Mat temp;
		cv::absdiff(img[i], rgb[i], temp);
		d.push_back(temp);
	}
	return d;
}

void third()
{
	// Директория откуда читать
	const std::string readFrom{ "C:/Users/dimam/Documents/Admin/Programming/C++/OpenCV/OpenCVData/test_data/" };
	// Директория куда записывать
	const std::string writeTo{ "C:/Users/dimam/Documents/Admin/Programming/C++/OpenCV/OpenCVData/test_data/" };

	cv::Mat img = cv::imread(readFrom + "apple_256x256.png", CV_LOAD_IMAGE_COLOR);

	// Запись изображения в формате jpeg(95) и jpeg(65)
	//cv::imwrite(writeTo + "apple_256x256_95.jpg", img, { CV_IMWRITE_JPEG_QUALITY, 95 });
	//cv::imwrite(writeTo + "apple_256x256_65.jpg", img, { CV_IMWRITE_JPEG_QUALITY, 65 });

	cv::Mat compressed95 = cv::imread(readFrom + "apple_256x256_95.jpg", CV_LOAD_IMAGE_COLOR);
	cv::Mat compressed65 = cv::imread(readFrom + "apple_256x256_65.jpg", CV_LOAD_IMAGE_COLOR);

	// Создание окна под все картинки
	int cx{ img.cols };
	int cy{ img.rows };
	cv::Mat allImages(cy * 3, cx * 4, img.type());

	// Выделение областей окна под мозаику
	std::vector<cv::Mat>split;
	for (int i{ 0 }; i < 3; ++i)
	{
		for (int j{ 0 }; j < 4; ++j)
			split.push_back(cv::Mat(allImages, cv::Rect2d(j * cx, i * cy, cx, cy)));
	}

	// Разделение картинок на BGR каналы
	std::vector<cv::Mat> rgb;
	std::vector<cv::Mat> rgb_95;
	std::vector<cv::Mat> rgb_65;
	cv::split(img, rgb);
	cv::split(compressed95, rgb_95);
	cv::split(compressed65, rgb_65);

	// Создал вектор всех картинок, чтобы потом в цикле просто скопировать их в области мозаики
	cv::Mat zeros(cv::Mat::zeros(cx, cy, CV_8UC1));
	std::vector<std::vector<cv::Mat>> tempV;

	tempV.push_back({ rgb[0], rgb[1],rgb[2] });
	tempV.push_back({ rgb[0], zeros, zeros });
	tempV.push_back({ zeros, rgb[1], zeros });
	tempV.push_back({ zeros, zeros, rgb[2] });

	// Разница по каналам с jpeg (95)
	std::vector<cv::Mat> temp = absDifference(rgb, rgb_95);
	tempV.push_back({ rgb_95[0], rgb_95[1],rgb_95[2] });
	tempV.push_back({ temp[0], temp[0] , temp[0] });
	tempV.push_back({ temp[1], temp[1] , temp[1] });
	tempV.push_back({ temp[2], temp[2] , temp[2] });

	// Разница по каналам с jpeg (65)
	temp = absDifference(rgb, rgb_65);
	tempV.push_back({ rgb_65[0], rgb_65[1],  rgb_65[2] });
	tempV.push_back({ temp[0], temp[0] , temp[0] });
	tempV.push_back({ temp[1], temp[1] , temp[1] });
	tempV.push_back({ temp[2], temp[2] , temp[2] });

	// Копирование картинок в выделенные области
	for (int i{ 0 }; i < 12; ++i)
	{
		cv::merge(tempV[i], split[i]);
	}

	cv::imwrite(writeTo + "lab2_all_images.png", allImages, { CV_IMWRITE_PNG_COMPRESSION, 0 });

	/*cv::namedWindow("All Images", cv::WINDOW_AUTOSIZE);
	cv::imshow("All Images", allImages);

	cv::waitKey(0);

	cv::destroyAllWindows;*/
}

int main()
{
	//tempTask();
	//second();
	third();

	return 0;
}