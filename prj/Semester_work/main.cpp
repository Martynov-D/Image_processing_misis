#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>


struct sLine 
{
    double mx, my;
    double sx, sy;
};

struct sPoint 
{
    double x, y;
};

//double compute_distance(sLine& line, sPoint& x)
//{
//    return fabs((x.x - line.sx) * line.my - (x.y - line.sy) * line.mx) / sqrt(line.mx * line.mx + line.my * line.my);
//}
//
//double model_verification(sPoint* inliers, int* no_inliers, sLine& estimated_model, sPoint* data, int no_data, double distance_threshold)
//{
//    *no_inliers = 0;
//    double cost = 0.;
//
//    for (int i = 0; i < no_data; i++) {
//        // –ассчитать длину ремонтной линии по пр€мой.
//        double distance = compute_distance(estimated_model, data[i]);
//
//        // ≈сли данные действительны в прогнозируемой модели, они добавл€ютс€ в действительный набор данных.
//        if (distance < distance_threshold) {
//            cost += 1.;
//
//            inliers[*no_inliers] = data[i];
//            ++(*no_inliers);
//        }
//    }
//    return cost;
//}
//
//bool find_in_samples(sPoint* samples, int no_samples, sPoint* data)
//{
//    for (int i = 0; i < no_samples; ++i)
//    {
//        if (samples[i].x == data->x && samples[i].y == data->y) 
//        {
//            return true;
//        }
//    }
//    return false;
//}
//
//void get_samples(sPoint* samples, int no_samples, sPoint* data, int no_data)
//{
//    // N случайных выборок вз€ты из данных без наложени€.
//    for (int i = 0; i < no_samples;) 
//    {
//        int j = rand() % no_data;
//
//        if (!find_in_samples(samples, i, &data[j])) 
//        {
//            samples[i] = data[j];
//            ++i;
//        }
//    };
//}
//
//int compute_model_parameter(sPoint samples[], int no_samples, sLine& model)
//{
//    // PCA ѕредсказать параметры линейной модели.
//
//    double sx = 0, sy = 0;
//    double sxx = 0, syy = 0;
//    double sxy = 0, sw = 0;
//
//    for (int i = 0; i < no_samples; ++i)
//    {
//        double& x = samples[i].x;
//        double& y = samples[i].y;
//
//        sx += x;
//        sy += y;
//        sxx += x * x;
//        sxy += x * y;
//        syy += y * y;
//        sw += 1;
//    }
//
//    //variance;
//    double vxx = (sxx - sx * sx / sw) / sw;
//    double vxy = (sxy - sx * sy / sw) / sw;
//    double vyy = (syy - sy * sy / sw) / sw;
//
//    //principal axis
//    double theta = atan2(2 * vxy, vxx - vyy) / 2;
//
//    model.mx = cos(theta);
//    model.my = sin(theta);
//
//    //center of mass(xc, yc)
//    model.sx = sx / sw;
//    model.sy = sy / sw;
//
//    //Ћинейное уравнение: sin(theta)*(x - sx) = cos(theta)*(y - sy);
//    return 1;
//}
//
//double ransac_line_fitting(sPoint* data, int no_data, sLine& model, double distance_threshold)
//{
//    const int no_samples = 2;
//
//    if (no_data < no_samples) {
//        return 0;
//    }
//
//    sPoint* samples = new sPoint[no_samples];
//
//    int no_inliers = 0;
//    sPoint* inliers = new sPoint[no_data];
//
//    sLine estimated_model;
//    double max_cost = 0.;
//
//    int max_iteration = (int)(1 + log(1. - 0.99) / log(1. - pow(0.6, no_samples)));
//
//    for (int i = 0; i < max_iteration; i++) {
//        // 1. hypothesis
//        // ѕроизвольно выбрать N выборочных данных из исходных данных.
//        get_samples(samples, no_samples, data, no_data);
//
//        // ѕросмотр этих данных как нормальных данных и прогнозирование параметров модели.
//        compute_model_parameter(samples, no_samples, estimated_model);
//        // 2. Verification
//        // ѕроверьте, соответствуют ли исходные данные прогнозируемой модели.
//        double cost = model_verification(inliers, &no_inliers, estimated_model, data, no_data, distance_threshold);
//        // ≈сли прогнозируема€ модель хорошо подходит, нова€ модель получаетс€ из достоверных данных дл€ этой модели..
//        if (max_cost < cost)
//        {
//            max_cost = cost;
//            compute_model_parameter(inliers, no_inliers, model);
//        }
//    }
//    delete[] samples;
//    delete[] inliers;
//
//    return max_cost;
//}

cv::Mat getImage(std::string name)
{
    // ѕуть по умолчанию, откуда берутс€ картинки + название картинки
    cv::Mat img = cv::imread("C:/Admin/Programming/C++/OpenCV/data/test_data/" + name);

    // ѕеревод цветной картинки в оттенки серого
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, CV_BGR2GRAY);

    return imgGray;
}

cv::Mat stereoBm(const cv::Mat& left, const cv::Mat& right, int nDisparities, int SADWindowSize, bool algo)
{
    cv::Mat disp(left.rows, left.cols, CV_32F);

    // »спользуетс€ StereoBM
    if (algo)
    {
        cv::Ptr<cv::StereoBM> left_sbm = cv::StereoBM::create(nDisparities, SADWindowSize);
        left_sbm->compute(left, right, disp);
    }
    // »спользуетс€ StereoSGBM
    else
    {
        int 	minDisparity = 0;
        int 	numDisparities = nDisparities;
        int 	blockSize = SADWindowSize;
        int 	P1 = 8 * 3 * blockSize * blockSize;
        int 	P2 = 32 * 3 * blockSize * blockSize;
        int 	disp12MaxDiff = 1;
        int 	preFilterCap = 63;
        int 	uniquenessRatio = 10;
        int 	speckleWindowSize = 400;
        int 	speckleRange = 32;
        int 	mode = cv::StereoSGBM::MODE_HH;

        cv::Ptr<cv::StereoSGBM> left_sgbm = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize);
        left_sgbm->setP1(P1);
        left_sgbm->setP2(P2);
        left_sgbm->setPreFilterCap(preFilterCap);
        left_sgbm->setMode(mode);
        //
        left_sgbm->setDisp12MaxDiff(disp12MaxDiff);
        left_sgbm->setUniquenessRatio(uniquenessRatio);
        left_sgbm->setSpeckleWindowSize(speckleWindowSize);
        left_sgbm->setSpeckleRange(speckleRange);

        left_sgbm->compute(left, right, disp);
    }

    return disp;
}

int main()
{
    // Ќазвани€ изображений
    std::string left_im{ "194_l.png" };
    std::string right_im{ "194_r.png" };

    cv::Mat left = getImage(left_im);
    cv::Mat right = getImage(right_im);

    // –азница между максимальной и минимальной диспаратност€ми
    int numDisparities = 16 * 4;
    // –азмер окна поиска совпадений
    int SADWindowSize = 3;
    // »спользуемый метод: True = BM, False = SGBM
    bool mode = false;
    if (mode)
    {
        SADWindowSize = 9;
    }

    // ѕодсчет карты диспаратности и ее нормализаци€
    cv::Mat disp;
    cv::Mat disp8;
    disp = stereoBm(left, right, numDisparities, SADWindowSize, mode);
    cv::normalize(disp, disp8, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    //  depth = baseline * focal / disparity
    //  ƒл€ KITTI  baseline = 0.54m, фокусное рассто€ние ~721 pix.
    //   арта диспаратности должна иметь размер исходного изображени€ (left \ right)
    /*cv::Mat depth = 721 * 0.54 / disp8;
    cv::normalize(depth, depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);*/

    // =============== v-disparity ================

    unsigned int IMAGE_HEIGHT = disp8.rows;
    unsigned int IMAGE_WIDTH = disp8.cols;
    unsigned int MAX_DISP = 256;

    cv::Mat image = disp8;
    //cv::Mat uDisparity = cv::Mat::zeros(MAX_DISP, IMAGE_WIDTH, CV_32F);
    cv::Mat vDisparity = cv::Mat::zeros(IMAGE_HEIGHT, MAX_DISP, CV_32F);

    cv::Mat tmpImageMat;
    cv::Mat tmpHistMat;

    float value_ranges[] = { (float)0, (float)MAX_DISP };
    const float* hist_ranges[] = { value_ranges };
    int channels[] = { 0 };
    int histSize[] = { MAX_DISP };

    // ѕодсчет v-disparity
    for (int i = 0; i < IMAGE_HEIGHT; i++)
    {
        tmpImageMat = image.row(i);
        vDisparity.row(i).copyTo(tmpHistMat);

        cv::calcHist(&tmpImageMat, 1, channels, cv::Mat(), tmpHistMat, 1, histSize, hist_ranges, true, false);

        vDisparity.row(i) = tmpHistMat.t() / (float)IMAGE_HEIGHT;
    }

    // ѕодсчет u-disparity
    /*image = image.t();
    for (int i = 0; i < IMAGE_WIDTH; i++)
    {
        tmpImageMat = image.row(i);
        uDisparity.col(i).copyTo(tmpHistMat);

        cv::calcHist(&tmpImageMat, 1, channels, cv::Mat(), tmpHistMat, 1, histSize, hist_ranges, true, false);

        uDisparity.col(i) = tmpHistMat / (float)IMAGE_WIDTH;
    }
    image = image.t();*/

    // Ќормализаци€ гистограмм и изменение цветовой палитры
    //cv::normalize(uDisparity, uDisparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    //cv::applyColorMap(uhist, uhist, cv::COLORMAP_JET);

    cv::normalize(vDisparity, vDisparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    //cv::applyColorMap(vDisparity, vDisparity, cv::COLORMAP_JET);

    cv::Mat vhist;
    
    cv::threshold(vDisparity, vhist, 30, 255, CV_THRESH_TOZERO);
    cv::applyColorMap(vDisparity, vDisparity, cv::COLORMAP_JET);

    
    int no_data = 400;
    sPoint* data = new sPoint[no_data];
    int k = 0, cnt = 0;
    for (int j = 255; j > 10; j--) 
    {
        for (int i = (vDisparity.rows)-1; i > 170; i--)
        {
            int d = vhist.at<unsigned char>(i, j);
            if (d != 0 && d < 220 && k < no_data)
            {
                //std::cout<<"diap is :"<<d<<std::endl;
                data[k].x = j;
                data[k].y = i;
                k++;
                vhist.at<unsigned char>(i, j) = 255;
                cnt++;
            }
            if (k >= no_data)
                break;
            if (cnt == 3)
            {
                cnt = 0;
                break;
            }
        }
        if (k >= no_data)
            break;
    }

    cv::Mat vhist_threshold;
    cv::threshold(vhist, vhist_threshold, 254, 255, CV_THRESH_TOZERO);

    cv::Mat canny;
    cv::Mat hough_lines;
    cv::Mat detected_edges;
    cv::Canny(vhist_threshold, canny, 50, 200, 3);
    cv::cvtColor(canny, hough_lines, cv::COLOR_GRAY2BGR);

    // === Probabilistic Line Transform ===

    // will hold the results of the detection
    std::vector<cv::Vec4i> linesP;
    // runs the actual detection
    HoughLinesP(canny, linesP, 1, CV_PI / 180, 50, 50, 10); 

    // Draw the lines
    for (size_t i = 0; i < linesP.size(); i++)
    {
        cv::Vec4i l = linesP[i];
        cv::line(hough_lines, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
        std::cout << l[0] << '\n' << l[1] << '\n' << l[2] << '\n' << l[3] << '\n';
    }

    // Show results
    cv::imshow("vDisparity", vDisparity);
    cv::imshow("vhist", vhist);
    cv::imshow("Source", vhist_threshold);
    cv::imshow("Detected Lines (in red) - Probabilistic Line Transform", hough_lines);
    /*
    sLine ground;
    static int no_data = 400;
    double cost = ransac_line_fitting(data, no_data, ground, 10);
    double ylim = ((ground.my * (-ground.sx)) / ground.mx) + ground.sy;
    double slop = ground.my / ground.mx;
    double Ymx = 0, Ymy = 0, Ysx = 0, Ysy = 0;
    if (30. < cost) 
    {

        if (ylim >= 70 && ylim < 180 && slop < 1.3)
        {
            cv::line(vvvDisparity, cv::Point(0, ((ground.my * (-ground.sx)) / ground.mx) + ground.sy), cv::Point(255, ((ground.my * (255 - ground.sx)) / ground.mx) + ground.sy), cv::Scalar(0, 0, 255), 2);
            cv::line(vvDisparity, cv::Point(0, pt1y), cv::Point(255, pt2y), cv::Scalar(0, 0, 255), 1);
            //cout<<((ground.my*(-ground.sx))/ground.mx)+ground.sy<<endl;

            pt1y = ((ground.my * (-ground.sx)) / ground.mx) + ground.sy;
            pt2y = ((ground.my * (255 - ground.sx)) / ground.mx) + ground.sy;
            Ymx = ground.mx; Ymy = ground.my; Ysx = ground.sx; Ysy = ground.sy;
        }
        else
        {
            for (int k = no_data; k >= 25; k -= 30)
            {
                double cost = ransac_line_fitting(data, k, ground, 30);
                double ylim = ((ground.my * (-ground.sx)) / ground.mx) + ground.sy;
                if (ylim >= 80 && ylim < 160 && slop < 1.4) // ”меньшите количество выборок, когда перва€ пр€ма€ лини€ не обнаружена
                {
                    Ymx = ground.mx; Ymy = ground.my; Ysx = ground.sx; Ysy = ground.sy;
                    cv::line(vvvDisparity, cv::Point(0, ((ground.my * (-ground.sx)) / ground.mx) + ground.sy), cv::Point(255, ((ground.my * (255 - ground.sx)) / ground.mx) + ground.sy), cv::Scalar(0, 0, 255), 2);
                    cv::line(vvDisparity, cv::Point(0, pt1y), cv::Point(255, pt2y), cv::Scalar(0, 0, 255), 1);
                    break;
                }
                else if (k >= 250) //  огда пр€ма€ лини€ не обнаружена
                {
                   cv:: line(vvvDisparity, cv::Point(0, pt1y), cv::Point(255, pt2y), cv::Scalar(0, 0, 255), 2);
                    cv::line(vvDisparity, cv::Point(0, pt1y), cv::Point(255, pt2y), cv::Scalar(0, 0, 255), 1);
                    break;
                }
            }
        }
    }
    */
    
    cv::Mat groundMask(disp8.rows, disp8.cols, CV_8UC3, CV_RGB(0, 0, 0));
    cv::Vec4i l = linesP[0]; 
    double Ysx = l[0];
    double Ysy = l[1];
    double Ymx = l[2];
    double Ymy = l[3];
    
    //std::cout << "1 x: " << Ymx << "\n1 y: " << Ymy << '\n';
    //std::cout << "2 x: " << Ysx << "\n2 y: " << Ysy << '\n';
    for (int i = (disp8.rows)/2; i < disp8.rows; i++) 
    {
        for (int j = numDisparities; j < disp8.cols; j++) 
        {
            int d = disp8.at<unsigned char>(i, j);

            if (i >= (((Ymy * (d - Ysx)) / Ymx) + Ysy) - 12 && d != 0)
            {
                disp8.at<unsigned char>(i, j) = 255;
                groundMask.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 84, 75);
            }
        }
    }

    cv::imshow("ground mask", groundMask);
    cv::imshow("Left", left);
    //cv::imshow("Right", right);
    cv::imshow("Disparity map", disp8);
    //cv::imshow("Depth", depth);
    //cv::imshow("uhist", uhist);
    //cv::imshow("vhist", vhist);
    //cv::imshow("Canny", canny);

    // —охранение изображений на диск
    {
        if (mode)
        {
            cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/disp8_BM.png", disp8, { CV_IMWRITE_PNG_COMPRESSION, 0 });
            cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/v_hist_BM.png", vDisparity, { CV_IMWRITE_PNG_COMPRESSION, 0 });
            cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/ground_mask_BM.png", groundMask, { CV_IMWRITE_PNG_COMPRESSION, 0 });
            cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/detected_ground_line_BM.png", hough_lines, { CV_IMWRITE_PNG_COMPRESSION, 0 });
            //cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/u_hist_BM.png", uDisparity, { CV_IMWRITE_PNG_COMPRESSION, 0 });
            //cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/depth_BM.png", depth, { CV_IMWRITE_PNG_COMPRESSION, 0 });
        }
        else
        {
            cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/disp8_SGBM.png", disp8, { CV_IMWRITE_PNG_COMPRESSION, 0 });
            cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/v_hist_SGBM.png", vDisparity, { CV_IMWRITE_PNG_COMPRESSION, 0 });
            cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/ground_mask_SGBM.png", groundMask, { CV_IMWRITE_PNG_COMPRESSION, 0 });
            cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/detected_ground_line_SGBM.png", hough_lines, { CV_IMWRITE_PNG_COMPRESSION, 0 });
            //cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/u_hist_SGBM.png", uDisparity, { CV_IMWRITE_PNG_COMPRESSION, 0 });
            //cv::imwrite("C:/Admin/Programming/C++/OpenCV/data/lab_data/depth_SGBM.png", depth, { CV_IMWRITE_PNG_COMPRESSION, 0 });
        }
    }

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
