import cv2
import numpy as np
import os
import glob

# Path to the folder containing the images
image_path = '/Users/csc/TRStrain/TR image'
# Path to the folder containing the text files
text_path = '/Users/csc/TRStrain/strain'
# Path to the results folder
results_path = '/Users/csc/TRStrain/results'
# Get a list of all the image filenames
image_filenames = glob.glob(os.path.join(image_path, '*.jpg'))

# Sort the filenames
image_filenames.sort()

# Iterate through the sorted filenames and process each image
for image_filename in image_filenames:
    # Read the image
    image = cv2.imread(image_filename)

    # Define a range for the red color
    lower_red = np.array([0, 0, 200])
    upper_red = np.array([100, 100, 255])

    # Create a mask for the red color
    mask = cv2.inRange(image, lower_red, upper_red)

    # Find the contours of the red lines
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding box that includes all the red lines
    x_min, y_min, x_max, y_max = np.inf, np.inf, 0, 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # Crop the image to the bounding box
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Read the corresponding text file
    base_name = os.path.splitext(os.path.basename(image_filename))[0]
    text_filename = os.path.join(text_path, base_name + '.txt')
    with open(text_filename, 'r') as text_file:
        lines = text_file.readlines()
        # Get the desired height from the third line
        numbers = [int(float(x.strip())) for x in lines[0].split(';') if x.strip()!='' ]

        desired_height = int(lines[2].strip())
        desired_width = int(lines[3].strip())
        print(f"Desired height: {desired_height}, desired width: {desired_width}")

    # Resize the cropped image using the desired height and cropped width
    resized_image = cv2.resize(cropped_image, (desired_width, desired_height))

    # Create a result list
    results = []
    for number in numbers:
        if number >= desired_width:
            results.append(0)
        else:
            # Extract the given column from the resized image
            columns = resized_image[:, number]
            # Find all the indices where the red color appears in the column
            red_indices = [i for i, pixel in enumerate(columns) if
                           (pixel[2] > 200 and pixel[0] < 100 and pixel[1] < 100)]
            # Add the smallest red color index to the results list
            results.append(min(red_indices, default=0))
    print('resized_image.shape', resized_image.shape)
    # Write results list to the results path
    os.makedirs(results_path, exist_ok=True)
    results_filename = os.path.join(results_path, base_name + '.txt')
    with open(results_filename, 'w') as results_file:
        results_file.write(','.join(map(str, results)))


    # Create an output directory if it doesn't exist
    output_directory = os.path.join(os.path.dirname(image_path), 'output')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save the resized image
    output_filename = os.path.join(output_directory, os.path.basename(image_filename))
    cv2.imwrite(output_filename, resized_image)

    print(f"Processed and saved {output_filename}")

    # Iterate through the numbers and corresponding result values
    for x, y in zip(numbers, results):
        # Draw a circle on the resized image at the (x, y) coordinate
        # 5 is the radius of the circle, and (0, 255, 0) is the color (green in this case)
        cv2.circle(resized_image, (x, y), 5, (0, 255, 0), -1)

    # Display the resized image with circles
    cv2.imshow('Resized Image with Circles', resized_image)
    cv2.waitKey(1)


# Close all OpenCV windows
cv2.destroyAllWindows()
