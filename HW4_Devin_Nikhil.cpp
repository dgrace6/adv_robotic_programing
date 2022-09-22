#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace cv;
using namespace cv::xfeatures2d;

using std::cout;
using std::endl;

// Pulls in the video
/*VideoCapture cap("Name of video");*/

// Test this
/*VideoCapture cap;
cap.open("Name of video");*/

int main(){

    VideoCapture cap("../required_files/IMG_6826.MOV");

    cout << "A" << endl;

//  Test this
/*  VideoCapture cap;
  cap.open("Name of video");*/

    // If cap is unable to open anything, terminate the program
    if(!cap.isOpened()){

        cout << "Error opening video stream or file" << endl;

    return -1;
    }

    cout << "B" << endl;

    while(1){
        
        Mat frame;

        cout << "C" << endl;
        
        // Capture frame-by-frame
        cap >> frame;

        cout << "D" << endl;

        // Break the loop if there is no frames left
        if (frame.empty())
            break;

        cout << "E" << endl;

        Mat grayscale;
        cvtColor(frame, grayscale, CV_RGB2GRAY); 
        
        // Show the frame pulled from the video
        imshow("Frame", grayscale);

        cout << "F" << endl;

        // Reading in any key presses during operation 
        char c=(char)waitKey(25);

        cout << "G" << endl;

        // The esc key is 27. So if pressed, breaks the loop
        if(c==27)
            break;

        cout << "H" << endl;
    }

// Releases the video so it can be opened again, less errors
cap.release();

// Closes all frames
destroyAllWindows();

return 0;
}