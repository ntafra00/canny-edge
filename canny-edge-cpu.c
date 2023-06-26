#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int convertRGBToGrayscale(int width, int height, unsigned char *outputImage, unsigned char *inputImage)
{
    int imageSize = width * height;

    for (int i = 0; i < imageSize; i++)
    {
        unsigned char r = inputImage[3 * i];
        unsigned char g = inputImage[3 * i + 1];
        unsigned char b = inputImage[3 * i + 2];
        // applying luminosity formula for grayscale
        outputImage[i] = (unsigned char)(0.2989 * r + 0.5870 * g + 0.1140 * b);
    }

    return 0;
}

int applyGaussianFilter(int width, int height, unsigned char *outputImage, unsigned char *inputImage)
{
    return 0;
}

int performCannyEdgeDetection(unsigned char *inputImage, unsigned char *outputImage, int width, int height)
{
    // convert given RGB image to grayscale
    convertRGBToGrayscale(width, height, outputImage, inputImage);

    // apply gaussian filter in order to reduce noise
    applyGaussianFilter(width, height, outputImage, inputImage);

    return 0;
}

int main()
{
    const char *inputImagePath = "input-cpu.jpg";
    const char *outputImagePath = "output-cpu.jpg";
    int width, height, channels;

    // Load the input image
    unsigned char *inputImage = stbi_load(inputImagePath, &width, &height, &channels, STBI_rgb);
    if (inputImage == NULL)
    {
        printf("Failed to load image: %s\n", inputImagePath);
        return -1;
    }

    if (width < 1400 || height < 1400)
    {
        printf("Choose image with greater resolution\n");
        return -1;
    }

    if (channels != 3)
    {
        printf("Images is not in RGB format\n");
        return -1;
    }

    // Allocate memory for the output image
    unsigned char *outputImage = (unsigned char *)malloc(width * height);

    // Apply canny edge filter operation

    performCannyEdgeDetection(inputImage, outputImage, width, height);

    // Write the output image to file
    stbi_write_jpg(outputImagePath, width, height, 1, outputImage, 100);

    // Free the allocated memory
    stbi_image_free(inputImage);
    free(outputImage);

    return 0;
}
