#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

const char *inputImagePath = "input-cpu.jpg";
const char *outputImagePath = "output-cpu.jpg";

/*
    Function that takes loaded image as array of type uint8_t and converts it to array of integers
    Since uint8_t has BGR format, 2 - k part is needed to convert it to RGB
*/
std::vector<int> imgToArray(uint8_t *pixelPtr, int sizeRows, int sizeCols, int sizeDepth)
{
    std::vector<int> pixels(sizeRows * sizeCols * sizeDepth);
    for (int i = 0; i < sizeRows; i++)
    {
        for (int j = 0; j < sizeCols; j++)
        {
            for (int k = 0; k < sizeDepth; k++)
            {
                pixels[i * sizeCols * sizeDepth + j * sizeDepth + k] =
                    (int)pixelPtr[i * sizeCols * sizeDepth + j * sizeDepth + 2 - k];
            }
        }
    }
    return pixels;
}

/*
    Function that takes output images as array of integers and converts it to array of uint8_t
*/

void arrayToImg(std::vector<int> &pixels, uint8_t *pixelPtr, int sizeRows, int sizeCols, int sizeDepth)
{
    for (int i = 0; i < sizeRows; i++)
    {
        for (int j = 0; j < sizeCols; j++)
        {
            for (int k = 0; k < sizeDepth; k++)
            {
                pixelPtr[i * sizeCols * sizeDepth + j * sizeDepth + k] =
                    (uint8_t)pixels[i * sizeCols * sizeDepth + j * sizeDepth + (sizeDepth - 1 - k)];
            }
        }
    }
    return;
}

/*
    Function that takes input image pixels and performs Gaussian blur on every single pixel
*/
std::vector<int> gaussianBlur(std::vector<int> &pixels, std::vector<std::vector<double>> &kernel, double kernelConst, int sizeRows, int sizeCols, int sizeDepth)
{
    std::vector<int> pixelsBlur(sizeRows * sizeCols * sizeDepth);
    for (int i = 0; i < sizeRows; i++)
    {
        for (int j = 0; j < sizeCols; j++)
        {
            for (int k = 0; k < sizeDepth; k++)
            {
                double sum = 0;
                double sumKernel = 0;
                for (int y = -2; y <= 2; y++)
                {
                    for (int x = -2; x <= 2; x++)
                    {
                        if ((i + x) >= 0 && (i + x) < sizeRows && (j + y) >= 0 && (j + y) < sizeCols)
                        {
                            double channel = (double)pixels[(i + x) * sizeCols * sizeDepth + (j + y) * sizeDepth + k];
                            sum += channel * kernelConst * kernel[x + 2][y + 2];
                            sumKernel += kernelConst * kernel[x + 2][y + 2];
                        }
                    }
                }
                pixelsBlur[i * sizeCols * sizeDepth + j * sizeDepth + k] = (int)(sum / sumKernel);
            }
        }
    }
    return pixelsBlur;
}

/*
    Function that converts given blurred image to grayscale
    Size of blurred image is reduced by 3 since every pixel is not present with one entry in the array
    Grayscale value is found by summing up RGB values and dividing it by 3
*/

std::vector<int> rgbToGrayscale(std::vector<int> &pixels, int sizeRows, int sizeCols, int sizeDepth)
{
    std::vector<int> pixelsGray(sizeRows * sizeCols);
    for (int i = 0; i < sizeRows; i++)
    {
        for (int j = 0; j < sizeCols; j++)
        {
            int sum = 0;
            for (int k = 0; k < sizeDepth; k++)
            {
                sum = sum + pixels[i * sizeCols * sizeDepth + j * sizeDepth + k];
            }
            pixelsGray[i * sizeCols + j] = (int)(sum / sizeDepth);
        }
    }
    return pixelsGray;
}

/*
    Function that performs non maximum suppresion on each pixel
    Actions are performed based on current pixel gradient direction

    Example:

    If current pixel has gradient direction 0 or 180 degrees, we have to check pixel that's bellow and above it
    If any of these pixels have greater gradient value than current one, current one's gradient value will be set to 0
*/

void performNonMaximumSuppresion(double *G, std::vector<int> &theta, int sizeCols, int sizeRows, std::vector<int> pixels, double largestG)
{
    for (int i = 1; i < sizeRows - 1; i++)
    {
        for (int j = 1; j < sizeCols - 1; j++)
        {
            if (theta[i * sizeCols + j] == 0 || theta[i * sizeCols + j] == 180)
            {
                if (G[i * sizeCols + j] < G[i * sizeCols + j - 1] || G[i * sizeCols + j] < G[i * sizeCols + j + 1])
                {
                    G[i * sizeCols + j] = 0;
                }
            }
            else if (theta[i * sizeCols + j] == 45 || theta[i * sizeCols + j] == 225)
            {
                if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j + 1] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j - 1])
                {
                    G[i * sizeCols + j] = 0;
                }
            }
            else if (theta[i * sizeCols + j] == 90 || theta[i * sizeCols + j] == 270)
            {
                if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j])
                {
                    G[i * sizeCols + j] = 0;
                }
            }
            else
            {
                if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j - 1] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j + 1])
                {
                    G[i * sizeCols + j] = 0;
                }
            }

            pixels[i * sizeCols + j] = (int)(G[i * sizeCols + j] * (255.0 / largestG));
        }
    }
};

std::vector<int> cannyFilter(std::vector<int> &pixels, int sizeRows, int sizeCols, double lowerThreshold, double higherThreshold)
{
    std::vector<int> pixelsCanny(sizeRows * sizeCols);
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    double *G = new double[sizeRows * sizeCols];
    std::vector<int> theta(sizeRows * sizeCols);
    double largestG = 0;

    // perform canny edge detection on everything but the edges
    for (int i = 1; i < sizeRows - 1; i++)
    {
        for (int j = 1; j < sizeCols - 1; j++)
        {
            // find gx and gy for each pixel
            double gxValue = 0;
            double gyValue = 0;
            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    gxValue = gxValue + (gx[1 - x][1 - y] * (double)(pixels[(i + x) * sizeCols + j + y]));
                    gyValue = gyValue + (gy[1 - x][1 - y] * (double)(pixels[(i + x) * sizeCols + j + y]));
                }
            }

            // Find gradient value of current pixel
            G[i * sizeCols + j] = std::sqrt(std::pow(gxValue, 2) + std::pow(gyValue, 2));
            double atanResult = atan2(gyValue, gxValue) * 180.0 / 3.14159265;
            theta[i * sizeCols + j] = (int)(180.0 + atanResult);

            if (G[i * sizeCols + j] > largestG)
            {
                largestG = G[i * sizeCols + j];
            }

            // setting the edges
            if (i == 1)
            {
                G[i * sizeCols + j - 1] = G[i * sizeCols + j];
                theta[i * sizeCols + j - 1] = theta[i * sizeCols + j];
            }
            else if (j == 1)
            {
                G[(i - 1) * sizeCols + j] = G[i * sizeCols + j];
                theta[(i - 1) * sizeCols + j] = theta[i * sizeCols + j];
            }
            else if (i == sizeRows - 1)
            {
                G[i * sizeCols + j + 1] = G[i * sizeCols + j];
                theta[i * sizeCols + j + 1] = theta[i * sizeCols + j];
            }
            else if (j == sizeCols - 1)
            {
                G[(i + 1) * sizeCols + j] = G[i * sizeCols + j];
                theta[(i + 1) * sizeCols + j] = theta[i * sizeCols + j];
            }

            // setting the corners
            if (i == 1 && j == 1)
            {
                G[(i - 1) * sizeCols + j - 1] = G[i * sizeCols + j];
                theta[(i - 1) * sizeCols + j - 1] = theta[i * sizeCols + j];
            }
            else if (i == 1 && j == sizeCols - 1)
            {
                G[(i - 1) * sizeCols + j + 1] = G[i * sizeCols + j];
                theta[(i - 1) * sizeCols + j + 1] = theta[i * sizeCols + j];
            }
            else if (i == sizeRows - 1 && j == 1)
            {
                G[(i + 1) * sizeCols + j - 1] = G[i * sizeCols + j];
                theta[(i + 1) * sizeCols + j - 1] = theta[i * sizeCols + j];
            }
            else if (i == sizeRows - 1 && j == sizeCols - 1)
            {
                G[(i + 1) * sizeCols + j + 1] = G[i * sizeCols + j];
                theta[(i + 1) * sizeCols + j + 1] = theta[i * sizeCols + j];
            }

            // round to the nearest 45 degrees
            theta[i * sizeCols + j] = round(theta[i * sizeCols + j] / 45) * 45;
        }
    }

    performNonMaximumSuppresion(G, theta, sizeCols, sizeRows, pixelsCanny, largestG);

    // double threshold
    bool changes;
    do
    {
        changes = false;
        for (int i = 1; i < sizeRows - 1; i++)
        {
            for (int j = 1; j < sizeCols - 1; j++)
            {
                if (G[i * sizeCols + j] < (lowerThreshold * largestG))
                {
                    G[i * sizeCols + j] = 0;
                }
                else if (G[i * sizeCols + j] >= (higherThreshold * largestG))
                {
                    continue;
                }
                else if (G[i * sizeCols + j] < (higherThreshold * largestG))
                {
                    int tempG = G[i * sizeCols + j];
                    G[i * sizeCols + j] = 0;
                    for (int x = -1; x <= 1; x++)
                    {
                        bool breakNestedLoop = false;
                        for (int y = -1; y <= 1; y++)
                        {
                            if (x == 0 && y == 0)
                            {
                                continue;
                            }
                            if (G[(i + x) * sizeCols + (j + y)] >= (higherThreshold * largestG))
                            {
                                G[i * sizeCols + j] = (higherThreshold * largestG);
                                changes = true;
                                breakNestedLoop = true;
                                break;
                            }
                        }
                        if (breakNestedLoop)
                        {
                            break;
                        }
                    }
                }
                pixelsCanny[i * sizeCols + j] = (int)(G[i * sizeCols + j] * (255.0 / largestG));
            }
        }
    } while (changes);
    return pixelsCanny;
};

void cannyEdgeDetection(uint8_t *inputImage, double lowerThreshold, double higherThreshold, int width, int height, int channels)
{
    std::vector<int> pixels = imgToArray(inputImage, height, width, channels);

    // GAUSSIAN_BLUR:

    std::vector<std::vector<double>> kernel = {{2.0, 4.0, 5.0, 4.0, 2.0},
                                               {4.0, 9.0, 12.0, 9.0, 4.0},
                                               {5.0, 12.0, 15.0, 12.0, 5.0},
                                               {4.0, 9.0, 12.0, 9.0, 4.0},
                                               {2.0, 4.0, 5.0, 4.0, 2.0}};
    double kernelConst = (1.0 / 159.0);
    std::vector<int> pixelsBlur = gaussianBlur(pixels, kernel, kernelConst, height, width, channels);

    // GRAYSCALE:

    std::vector<int> pixelsGray = rgbToGrayscale(pixelsBlur, height, width, channels);

    // CANNY_FILTER:

    std::vector<int> pixelsCanny = cannyFilter(pixelsGray, height, width, lowerThreshold, higherThreshold);

    uint8_t *outputImage = (unsigned char *)malloc(width * height);
    arrayToImg(pixelsCanny, outputImage, height, width, 1);

    stbi_write_jpg(outputImagePath, width, height, 1, outputImage, 100);
}

int main()
{
    int width, height, channels;

    // Load the input image
    uint8_t *inputImage = stbi_load(inputImagePath, &width, &height, &channels, STBI_rgb);
    if (inputImage == NULL)
    {
        printf("Failed to load image: %s\n", inputImagePath);
        return -1;
    }

    // if (width < 1400 || height < 1400)
    // {
    //     printf("Choose image with greater resolution\n");
    //     return -1;
    // }

    if (channels != 3)
    {
        printf("Images is not in RGB format\n");
        return -1;
    }

    double lowerThreshold = 0.03;
    double higherThreshold = 0.1;

    cannyEdgeDetection(inputImage, lowerThreshold, higherThreshold, width, height, channels);

    return 0;
}

// #include <iostream>
// #include <vector>
// #include <math.h>
// #include <stdio.h>
// #include <stdlib.h>

// #define STB_IMAGE_IMPLEMENTATION
// #include "stb_image.h"

// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "stb_image_write.h"

// using namespace std;

// const char *inputImagePath = "input-cpu.jpg";
// const char *outputImagePath = "output-cpu.jpg";

// /*
//     Function that takes loaded image as array of type uint8_t and converts it to array of integers
//     Since uint8_t has BGR format, 2 - k part is needed to convert it to RGB
// */
// std::vector<int> imgToArray(uint8_t *pixelPtr, int sizeRows, int sizeCols, int sizeDepth)
// {
//     std::vector<int> pixels(sizeRows * sizeCols * sizeDepth);
//     for (int i = 0; i < sizeRows; i++)
//     {
//         for (int j = 0; j < sizeCols; j++)
//         {
//             for (int k = 0; k < sizeDepth; k++)
//             {
//                 pixels[i * sizeCols * sizeDepth + j * sizeDepth + k] =
//                     (int)pixelPtr[i * sizeCols * sizeDepth + j * sizeDepth + 2 - k];
//             }
//         }
//     }
//     return pixels;
// }

// /*
//     Function that takes output images as array of integers and converts it to array of uint8_t
// */

// void arrayToImg(std::vector<int> &pixels, uint8_t *pixelPtr, int sizeRows, int sizeCols, int sizeDepth)
// {
//     for (int i = 0; i < sizeRows; i++)
//     {
//         for (int j = 0; j < sizeCols; j++)
//         {
//             for (int k = 0; k < sizeDepth; k++)
//             {
//                 pixelPtr[i * sizeCols * sizeDepth + j * sizeDepth + k] =
//                     (uint8_t)pixels[i * sizeCols * sizeDepth + j * sizeDepth + (sizeDepth - 1 - k)];
//             }
//         }
//     }
//     return;
// }
// /*
//     Function that takes input image pixels and performs Gaussian blur on every single pixel

// */

// std::vector<int> gaussianBlur(std::vector<int> &pixels, std::vector<std::vector<double>> &kernel, double kernelConst, int sizeRows, int sizeCols, int sizeDepth)
// {
//     std::vector<int> pixelsBlur(sizeRows * sizeCols * sizeDepth);
//     for (int i = 0; i < sizeRows; i++)
//     {
//         for (int j = 0; j < sizeCols; j++)
//         {
//             for (int k = 0; k < sizeDepth; k++)
//             {
//                 double sum = 0;
//                 double sumKernel = 0;
//                 for (int y = -2; y <= 2; y++)
//                 {
//                     for (int x = -2; x <= 2; x++)
//                     {
//                         if ((i + x) >= 0 && (i + x) < sizeRows && (j + y) >= 0 && (j + y) < sizeCols)
//                         {
//                             double channel = (double)pixels[(i + x) * sizeCols * sizeDepth + (j + y) * sizeDepth + k];
//                             sum += channel * kernelConst * kernel[x + 2][y + 2];
//                             sumKernel += kernelConst * kernel[x + 2][y + 2];
//                         }
//                     }
//                 }
//                 pixelsBlur[i * sizeCols * sizeDepth + j * sizeDepth + k] = (int)(sum / sumKernel);
//             }
//         }
//     }
//     return pixelsBlur;
// }

// /*
//     Function that converts given blurred image to grayscale
//     Size of blurred image is reduced by 3 since every pixel is not present with one entry in the array
//     Grayscale value is found by summing up RGB values and dividing it by 3
// */

// std::vector<int> rgbToGrayscale(std::vector<int> &pixels, int sizeRows, int sizeCols, int sizeDepth)
// {
//     std::vector<int> pixelsGray(sizeRows * sizeCols);
//     for (int i = 0; i < sizeRows; i++)
//     {
//         for (int j = 0; j < sizeCols; j++)
//         {
//             int sum = 0;
//             for (int k = 0; k < sizeDepth; k++)
//             {
//                 sum = sum + pixels[i * sizeCols * sizeDepth + j * sizeDepth + k];
//             }
//             pixelsGray[i * sizeCols + j] = (int)(sum / sizeDepth);
//         }
//     }
//     return pixelsGray;
// }

// /*
//     Function that performs non maximum suppresion on each pixel
//     Actions are performed based on current pixel gradient direction

//     Example:

//     If current pixel has gradient direction 0 or 180 degrees, we have to check pixel that's bellow and above it
//     If any of these pixels have greater gradient value than current one, current one's gradient value will be set to 0
// */

// void performNonMaximumSuppresion(double *G, std::vector<int> &theta, int sizeCols, int sizeRows, std::vector<int> pixels, double largestG)
// {
//     for (int i = 1; i < sizeRows - 1; i++)
//     {
//         for (int j = 1; j < sizeCols - 1; j++)
//         {
//             if (theta[i * sizeCols + j] == 0 || theta[i * sizeCols + j] == 180)
//             {
//                 if (G[i * sizeCols + j] < G[i * sizeCols + j - 1] || G[i * sizeCols + j] < G[i * sizeCols + j + 1])
//                 {
//                     G[i * sizeCols + j] = 0;
//                 }
//             }
//             else if (theta[i * sizeCols + j] == 45 || theta[i * sizeCols + j] == 225)
//             {
//                 if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j + 1] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j - 1])
//                 {
//                     G[i * sizeCols + j] = 0;
//                 }
//             }
//             else if (theta[i * sizeCols + j] == 90 || theta[i * sizeCols + j] == 270)
//             {
//                 if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j])
//                 {
//                     G[i * sizeCols + j] = 0;
//                 }
//             }
//             else
//             {
//                 if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j - 1] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j + 1])
//                 {
//                     G[i * sizeCols + j] = 0;
//                 }
//             }

//             pixels[i * sizeCols + j] = (int)(G[i * sizeCols + j] * (255.0 / largestG));
//         }
//     }
// };

// std::vector<int> cannyFilter(std::vector<int> &pixels, int sizeRows, int sizeCols, double lowerThreshold, double higherThreshold)
// {
//     std::vector<int> pixelsCanny(sizeRows * sizeCols);
//     int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
//     int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
//     double *G = new double[sizeRows * sizeCols];
//     std::vector<int> theta(sizeRows * sizeCols);
//     double largestG = 0;

//     // perform canny edge detection on everything but the edges
//     for (int i = 1; i < sizeRows - 1; i++)
//     {
//         for (int j = 1; j < sizeCols - 1; j++)
//         {
//             // find gx and gy for each pixel
//             double gxValue = 0;
//             double gyValue = 0;
//             for (int x = -1; x <= 1; x++)
//             {
//                 for (int y = -1; y <= 1; y++)
//                 {
//                     gxValue = gxValue + (gx[1 - x][1 - y] * (double)(pixels[(i + x) * sizeCols + j + y]));
//                     gyValue = gyValue + (gy[1 - x][1 - y] * (double)(pixels[(i + x) * sizeCols + j + y]));
//                 }
//             }

//             // Find gradient value of current pixel
//             G[i * sizeCols + j] = std::sqrt(std::pow(gxValue, 2) + std::pow(gyValue, 2));
//             double atanResult = atan2(gyValue, gxValue) * 180.0 / 3.14159265;
//             theta[i * sizeCols + j] = (int)(180.0 + atanResult);

//             if (G[i * sizeCols + j] > largestG)
//             {
//                 largestG = G[i * sizeCols + j];
//             }

//             // setting the edges
//             if (i == 1)
//             {
//                 G[i * sizeCols + j - 1] = G[i * sizeCols + j];
//                 theta[i * sizeCols + j - 1] = theta[i * sizeCols + j];
//             }
//             else if (j == 1)
//             {
//                 G[(i - 1) * sizeCols + j] = G[i * sizeCols + j];
//                 theta[(i - 1) * sizeCols + j] = theta[i * sizeCols + j];
//             }
//             else if (i == sizeRows - 1)
//             {
//                 G[i * sizeCols + j + 1] = G[i * sizeCols + j];
//                 theta[i * sizeCols + j + 1] = theta[i * sizeCols + j];
//             }
//             else if (j == sizeCols - 1)
//             {
//                 G[(i + 1) * sizeCols + j] = G[i * sizeCols + j];
//                 theta[(i + 1) * sizeCols + j] = theta[i * sizeCols + j];
//             }

//             // setting the corners
//             if (i == 1 && j == 1)
//             {
//                 G[(i - 1) * sizeCols + j - 1] = G[i * sizeCols + j];
//                 theta[(i - 1) * sizeCols + j - 1] = theta[i * sizeCols + j];
//             }
//             else if (i == 1 && j == sizeCols - 1)
//             {
//                 G[(i - 1) * sizeCols + j + 1] = G[i * sizeCols + j];
//                 theta[(i - 1) * sizeCols + j + 1] = theta[i * sizeCols + j];
//             }
//             else if (i == sizeRows - 1 && j == 1)
//             {
//                 G[(i + 1) * sizeCols + j - 1] = G[i * sizeCols + j];
//                 theta[(i + 1) * sizeCols + j - 1] = theta[i * sizeCols + j];
//             }
//             else if (i == sizeRows - 1 && j == sizeCols - 1)
//             {
//                 G[(i + 1) * sizeCols + j + 1] = G[i * sizeCols + j];
//                 theta[(i + 1) * sizeCols + j + 1] = theta[i * sizeCols + j];
//             }

//             // round to the nearest 45 degrees
//             theta[i * sizeCols + j] = round(theta[i * sizeCols + j] / 45) * 45;
//         }
//     }

//     performNonMaximumSuppresion(G, theta, sizeCols, sizeRows, pixelsCanny, largestG);

//     // double threshold
//     bool changes;
//     do
//     {
//         changes = false;
//         for (int i = 1; i < sizeRows - 1; i++)
//         {
//             for (int j = 1; j < sizeCols - 1; j++)
//             {
//                 if (G[i * sizeCols + j] < (lowerThreshold * largestG))
//                 {
//                     G[i * sizeCols + j] = 0;
//                 }
//                 else if (G[i * sizeCols + j] >= (higherThreshold * largestG))
//                 {
//                     continue;
//                 }
//                 else if (G[i * sizeCols + j] < (higherThreshold * largestG))
//                 {
//                     int tempG = G[i * sizeCols + j];
//                     G[i * sizeCols + j] = 0;
//                     for (int x = -1; x <= 1; x++)
//                     {
//                         bool breakNestedLoop = false;
//                         for (int y = -1; y <= 1; y++)
//                         {
//                             if (x == 0 && y == 0)
//                             {
//                                 continue;
//                             }
//                             if (G[(i + x) * sizeCols + (j + y)] >= (higherThreshold * largestG))
//                             {
//                                 G[i * sizeCols + j] = (higherThreshold * largestG);
//                                 changes = true;
//                                 breakNestedLoop = true;
//                                 break;
//                             }
//                         }
//                         if (breakNestedLoop)
//                         {
//                             break;
//                         }
//                     }
//                 }
//                 pixelsCanny[i * sizeCols + j] = (int)(G[i * sizeCols + j] * (255.0 / largestG));
//             }
//         }
//     } while (changes);
//     return pixelsCanny;
// };

// void cannyEdgeDetection(uint8_t *inputImage, double lowerThreshold, double higherThreshold, int width, int height, int channels)
// {
//     std::vector<int> pixels = imgToArray(inputImage, height, width, channels);

//     // GAUSSIAN_BLUR:

//     std::vector<std::vector<double>> kernel = {{2.0, 4.0, 5.0, 4.0, 2.0},
//                                                {4.0, 9.0, 12.0, 9.0, 4.0},
//                                                {5.0, 12.0, 15.0, 12.0, 5.0},
//                                                {4.0, 9.0, 12.0, 9.0, 4.0},
//                                                {2.0, 4.0, 5.0, 4.0, 2.0}};
//     double kernelConst = (1.0 / 159.0);
//     std::vector<int> pixelsBlur = gaussianBlur(pixels, kernel, kernelConst, height, width, channels);

//     // GRAYSCALE:

//     std::vector<int> pixelsGray = rgbToGrayscale(pixelsBlur, height, width, channels);

//     // CANNY_FILTER:

//     std::vector<int> pixelsCanny = cannyFilter(pixelsGray, height, width, lowerThreshold, higherThreshold);

//     uint8_t *outputImage = (unsigned char *)malloc(width * height);
//     arrayToImg(pixelsCanny, outputImage, height, width, 1);

//     stbi_write_jpg(outputImagePath, width, height, 1, outputImage, 100);
// }

// int main()
// {
//     int width, height, channels;

//     // Load the input image
//     uint8_t *inputImage = stbi_load(inputImagePath, &width, &height, &channels, STBI_rgb);
//     if (inputImage == NULL)
//     {
//         printf("Failed to load image: %s\n", inputImagePath);
//         return -1;
//     }

//     // if (width < 1400 || height < 1400)
//     // {
//     //     printf("Choose image with greater resolution\n");
//     //     return -1;
//     // }

//     if (channels != 3)
//     {
//         printf("Images is not in RGB format\n");
//         return -1;
//     }

//     double lowerThreshold = 0.03;
//     double higherThreshold = 0.1;

//     cannyEdgeDetection(inputImage, lowerThreshold, higherThreshold, width, height, channels);

//     return 0;
// }
