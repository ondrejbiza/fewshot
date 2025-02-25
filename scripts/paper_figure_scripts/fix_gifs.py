import imageio.v2 as iio



import os

directory_path = "/home/rthomp12/Documents"

for filename in os.listdir(directory_path)[21:]:

    file_path = os.path.join(directory_path, filename)
    if os.path.isfile(file_path):

        gif = iio.get_reader(file_path)

        # Initialize the duration
        duration = 0

        # Loop through the frames and sum the durations
        for frame in gif:
            duration += gif.get_meta_data(index=frame)["duration"]

        frames = iio.mimread(file_path)
        print(file_path)
        # The default duration of each frame
        iio.mimwrite(file_path, frames[1:], duration=duration/1000.0)


