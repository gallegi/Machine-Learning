
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <fstream>
#include <set>
#include <vector>
#include "sign_classify.cpp"

using namespace std;
using namespace cv;

typedef std::pair<Mat, Rect> MatArea;

vector<MatArea> api_traffic_sign_detection_visual(Mat& src);

Mat blue_detector(Mat input);
Mat white_detector(Mat input);
Mat red_detector(Mat input);
Mat green_detector(Mat input);

bool check_blue(int x, int y, int width, int height, Mat input);
bool check_red(int x, int y, int width, int height, Mat input);
bool check_green(int x, int y, int width, int height, Mat input);

vector<MatArea> extract_candidate_img_visual(Mat src,Mat& raw, 
        Mat hsv, char color, vector<MatArea>& result);

Mat ROI(Mat src, int x, int y, int width, int height);

Mat test(Mat src);
Mat convert(Mat src);
Mat normalizeI(const Mat& src);

int rangeH, rangeS, rangeV;

int evaluate_frame_visual(Mat& frame);
void test_process();
void extract();

int mode;
int SENTIVITY = 10;
String VIDEO_DIR = "";
String extracted_folder = "extracted3/";
String extracted_prefix = "7";

int counter = 0;    
string pre_label = "";
int stack_times = 0;
double std_cnfdnce = 0.95;

Mat src;
int frame_index = 0;
int start_frame = 0;

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    mode = atoi(argv[1]);
    VIDEO_DIR = argv[2];
    start_frame = atoi(argv[3]);

    test_process();

    //extract();
    return 0;
}

void test_process(){
    VideoCapture cap;
    if(mode == 0){
        cap = VideoCapture(0);
    }else{
        cap = VideoCapture(VIDEO_DIR);
    }
    
    while(true){
        char key;
        frame_index++;
        
        
        Mat frame;
        cap >> frame;
        
        if(frame_index<start_frame)
            continue;
        
        Mat eval_frame;
        resize(frame, eval_frame, Size(320,240));
        
        int label = evaluate_frame_visual(eval_frame);
        key = waitKey(40);
        if(key == 's'){
            waitKey(0);
        }
        cout<<label<<endl;
    }
}

void extract(){
    VideoCapture cap;
    if(mode == 0){
        cap = VideoCapture(0);
    }else{
        cap = VideoCapture(VIDEO_DIR);
    }
    
    while(true){
        
        counter++;
        if(counter<start_frame)
            continue;
        
        Mat frame;
        cap >> frame;
        
        imshow("frame", frame);
        
        Mat eval_frame;
        resize(frame, eval_frame, Size(320,240));
        
        vector<MatArea> candidates = api_traffic_sign_detection_visual(eval_frame);
        
        //extract cropped images
        for(int i=0;i<candidates.size();i++){
            String file_name = extracted_folder+to_string(getTickCount())+ ".png";
            imwrite(file_name, candidates[i].first);
        }
        
        waitKey(1);
    }
}

int evaluate_frame_visual(Mat& frame) {
    int int_label = -1;

    vector<MatArea> detected_imgs = api_traffic_sign_detection_visual(frame);

    double start = getTickCount();
    //get max label
    double max_confidence = 0;
    Prediction p_max;
    MatArea mat_area;
    if (detected_imgs.size() > 0) {
        mat_area = detected_imgs[0];
    }

    for (int i = 0; i < detected_imgs.size(); i++) {
        Mat input;
        //cvtColor(detected_imgs[i].first, input, CV_BGR2GRAY);
        Prediction p = getLabel(detected_imgs[i].first);
        cout << p.first <<" "<<p.second<<endl;
        if (p.second > max_confidence) {
            p_max = p;
            max_confidence = p_max.second;
            mat_area = detected_imgs[i];
        }
    }

    double _end = getTickCount();

   // cout << "FPS: " << getTickFrequency() / (_end - start) << endl;

    //increase counter if pre_label = current label (p_max.first)
    if (p_max.first == pre_label) {
        counter++;
    } else {
        counter = 0;
    }

    pre_label = p_max.first;
    //cout <<counter <<"--------"<<endl;
    if (counter > stack_times) {
        if (p_max.first == "0 turn_left_ahead" && p_max.second >= std_cnfdnce) {
            int_label = 0;
            cout << p_max.first<<" "<<p_max.second << endl;
            rectangle(frame, mat_area.second, Scalar(0,0,255),10);
            putText(frame, p_max.first, Point(80, 20), FONT_HERSHEY_SIMPLEX, .7, Scalar(0, 0, 255), 2, 8, false);
        } else if (p_max.first == "1 turn_right_ahead" && p_max.second >= std_cnfdnce) {
            int_label = 1;
            cout << p_max.first<<" "<<p_max.second << endl;
            rectangle(frame, mat_area.second, Scalar(0,0,255),10);
            putText(frame, p_max.first, Point(80, 20), FONT_HERSHEY_SIMPLEX, .7, Scalar(0, 0, 255), 2, 8, false);
        }else if (p_max.first == "2 stop" && p_max.second >= std_cnfdnce) {
            int_label = 2;
            cout << p_max.first<<" "<<p_max.second << endl;
            rectangle(frame, mat_area.second, Scalar(0,0,255),10);
            putText(frame, p_max.first, Point(80, 20), FONT_HERSHEY_SIMPLEX, .7, Scalar(0, 0, 255), 2, 8, false);
        }else if (p_max.first == "3 o_red" && p_max.second >= std_cnfdnce) {
            int_label = 3;
            cout << p_max.first<<" "<<p_max.second << endl;
            rectangle(frame, mat_area.second, Scalar(0,0,255),10);
            putText(frame, p_max.first, Point(80, 20), FONT_HERSHEY_SIMPLEX, .7, Scalar(0, 0, 255), 2, 8, false);
        }else if (p_max.first == "4 o_green" && p_max.second >= std_cnfdnce) {
            int_label = 4;
            cout << p_max.first<<" "<<p_max.second << endl;
            rectangle(frame, mat_area.second, Scalar(0,0,255),10);
            putText(frame, p_max.first, Point(80, 20), FONT_HERSHEY_SIMPLEX, .7, Scalar(0, 0, 255), 2, 8, false);
        }


        int area = mat_area.second.width * mat_area.second.height;

        counter = 0;
        pre_label = "";
        
    }

   imshow("Traffic Sign", frame);
    return int_label;
}


vector<MatArea> api_traffic_sign_detection_visual(Mat& src) {
    Mat blue, red, green;
    Mat dst, hsv;
    vector<MatArea> result;

    //normalize each range of color in the image to 1 color
    hsv = normalizeI(src);

 //   imshow("L: ", convert(hsv));
    
    //create red, green, blue massk for detection by color
    red = red_detector(hsv);
    blue = blue_detector(hsv);
    green = green_detector(hsv);

    //blur red, green, blue mask to remove noise
    GaussianBlur(blue, blue, Size(3, 3), 0, 0);
    GaussianBlur(red, red, Size(5, 5), 0, 0);
    GaussianBlur(green, green, Size(3, 3), 0, 0);

    //extract candidate
    extract_candidate_img_visual(blue,src,hsv, 'B', result);
    extract_candidate_img_visual(red, src, hsv, 'R', result);
    extract_candidate_img_visual(green, src, hsv, 'G', result);

    imshow("norm_img: ", convert(hsv));

    cout << "size of vec candidates: " <<result.size()<<endl;
    return result;
}

Mat blue_detector(Mat input) {
    Mat dst;
    cv::inRange(input,Scalar(100,255,255),Scalar(100,255,255),dst);
    return dst;
}

Mat white_detector(Mat input) {
    Mat dst;
    cv::inRange(input, Scalar(0, 0, 255),  (0, 0, 255), dst);
    return dst;
}

Mat red_detector(Mat input) {
    Mat dst;
    cv::inRange(input, Scalar(179, 255, 255), Scalar(179, 255, 255), dst);
    return dst;
}

Mat green_detector(Mat input) {
    Mat dst;
    cv::inRange(input, Scalar(57, 255, 255), Scalar(57, 255, 255), dst);
    return dst;
}

//check if this cropped image contains blue and white pixel with some given ratio
bool check_blue(int x, int y, int width, int height, Mat input) {

    Mat checkMatrix = Mat::zeros(Size(width, height), CV_8U);
    Mat croped = ROI(input, x, y, width, height);
    double given_ratio = 0.2;

    croped = normalizeI(croped.clone());

    Mat filted = white_detector(croped);

    float sum = 0;
    bool haveBlue = false;
    bool haveWhite = false;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (filted.at<uchar>(i, j) == 255) {
                sum--;
                haveWhite = true;
            }
        }
    }

    filted = blue_detector(croped);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (filted.at<uchar>(i, j) == 255) {
                sum++;
                haveBlue = true;
            }
        }
    }

    sum = sum / (height * width);

    if (haveBlue && sum > given_ratio) {
        return true;
    }
    return false;
}

//check if this cropped image contains red and white pixel with some given ratio
bool check_red(int x, int y, int width, int height, Mat input) {

    Mat checkMatrix = Mat::zeros(Size(width, height), CV_8U);
    Mat croped = ROI(input, x, y, width, height);
    double given_ratio = 0.2;

    croped = normalizeI(croped.clone());

    Mat filted = white_detector(croped);

    float sum = 0;
    bool haveRed = false;
    bool haveWhite = false;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (filted.at<uchar>(i, j) == 255) {
                sum--;
                haveWhite = true;
            }
        }
    }

    filted = red_detector(croped);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (filted.at<uchar>(i, j) == 255) {
                sum++;
                haveRed = true;
            }
        }
    }

    sum = sum / (height * width);

    if (haveRed && sum > given_ratio) {
        return true;
    }
    return false;
}

//check if this cropped image contains green and white pixel with some given ratio
bool check_green(int x, int y, int width, int height, Mat input) {

    Mat checkMatrix = Mat::zeros(Size(width, height), CV_8U);
    Mat croped = ROI(input, x, y, width, height);
    double given_ratio = 0.5;

    croped = normalizeI(croped.clone());

    Mat filted = white_detector(croped);

    float sum = 0;
    bool haveGreen = false;
    bool haveWhite = false;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (filted.at<uchar>(i, j) == 255) {
                sum--;
                haveWhite = true;
            }
        }
    }

    filted = green_detector(croped);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (filted.at<uchar>(i, j) == 255) {
                sum++;
                haveGreen = true;
            }
        }
    }

    sum = sum / (height * width);

    if (haveGreen && sum > given_ratio) {
        return true;
    }
    return false;
}

vector<MatArea> extract_candidate_img_visual(Mat src,Mat& raw, Mat hsv, char color, vector<MatArea>& result) {
    vector<vector<Point> > cnts;
    vector<Vec4i> hierarchy;

    findContours(src, cnts, hierarchy, RETR_EXTERNAL, 1, Point(0, 0));
    for (int i = 0; i < cnts.size(); i++) {
        vector<Point> cnt = cnts[i];
        Rect rect = boundingRect(cnt);
        int height = rect.height;
        int width = rect.width;
        int area = width * height;
        if (area > 500 && abs(width - height) < 10) {
            int x = rect.x;
            int y = rect.y;

            if(color == 'R') {
                if(!check_red(x, y, width, height, raw)) {
                    continue;
                }
            } else if(color == 'B') {
                if (!check_blue(x, y, width, height, raw)) {
                    continue;
                }
            } else if(color == 'G') {
                if (!check_green(x, y, width, height, raw)) {
                    continue;
                }
            }

            rectangle(hsv, Point(x, y), Point(x + width, y + height), Scalar(0, 0, 0), 3);

            MatArea mat_area;
            mat_area.first = ROI(raw, x, y, width, height);
            mat_area.second = rect;

            result.push_back(mat_area);
        }
    }
    return result;
}

Mat ROI(Mat src, int x, int y, int width, int height) {
    return src(Rect(x,y,width,height));
}


Mat convert(Mat src) {
    Mat dst;
    cvtColor(src, dst, CV_HSV2BGR);
    return dst;
}

// normalize all values of a color in an interval to only 1 color
Mat normalizeI(const Mat& src) {
    int rangeH, rangeS, rangeV;
    Mat dst;
    //convert to hsv
    cvtColor(src,dst, CV_BGR2HSV);

    /*============DEFINE RANGE FOR HUE COLOR==============*/
    int max_hue = 179;
    int min_hue = 0;
    /*Lower thresh*/
    int l_yellow, l_green, l_blue, l_magneta, l_red;
    /*Upper thresh*/
    int h_yellow, h_green, h_blue, h_magneta, h_red;
    /*====================================================*/

    /*Specific range*/
    l_green = 35;
    h_green = 96;

    l_blue = 97;
    h_blue = 118;

    l_red = 169;
    h_red = 5;
    /*------------*/


    for (int y = 0; y < dst.rows; ++y) {
        for (int x = 0; x < dst.cols; ++x) {
            /*===================HUE PROCESSING==========================*/
            if (dst.at<Vec3b>(y, x)[0] >= l_green && dst.at<Vec3b>(y, x)[0] < h_green) {
                //Green
                rangeH = 57;
            } else if (dst.at<Vec3b>(y, x)[0] >= l_blue && dst.at<Vec3b>(y, x)[0] < h_blue) {
                //Blue
                rangeH = 100;
            } else if (dst.at<Vec3b>(y, x)[0] >= l_red || dst.at<Vec3b>(y, x)[0] < h_red) {
                //Red
                rangeH = 179;
            }
            else {
                dst.at<Vec3b>(y, x)[0] = 0;
                dst.at<Vec3b>(y, x)[1] = 0;
                dst.at<Vec3b>(y, x)[2] = 255;
            }
            dst.at<Vec3b>(y, x)[0] = rangeH;
            /*===================SATURATION + VALUE PROCESSING================*/
            int thresh = 100;
            if (dst.at<Vec3b>(y, x)[1] < thresh && dst.at<Vec3b>(y, x)[2] >= thresh) {
                //White case
                dst.at<Vec3b>(y, x)[0] = 0;
                rangeS = 0;
                rangeV = 255;
            } else if (dst.at<Vec3b>(y, x)[1] >= thresh && dst.at<Vec3b>(y, x)[2] >= thresh) {
                //Light Color case
                rangeS = 255;
                rangeV = 255;
            } else if (dst.at<Vec3b>(y, x)[1] >= thresh && dst.at<Vec3b>(y, x)[2] < thresh) {
                //Dark Color case
                rangeS = 255;
                rangeV = 255;
            } else {
                //Black case
                dst.at<Vec3b>(y, x)[0] = 0;
                rangeS = 0;
                rangeV = 255;
            }
            dst.at<Vec3b>(y, x)[1] = rangeS;
            dst.at<Vec3b>(y, x)[2] = rangeV;
            /*=================================================================*/

        }
    }

    return dst;
}