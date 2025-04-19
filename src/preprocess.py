def parse_image_metadata(image_dir):
    import os
    import re

    pattern = re.compile(r'C(\d+)_F(\d+)_s(\d+)_w(\d+)\.TIF$', re.IGNORECASE)
    image_data = []

    for filename in os.listdir(image_dir):
        match = pattern.search(filename)
        if match:
            cell_count = int(match.group(1))
            image_path = os.path.join(image_dir, filename)
            image_data.append((image_path, cell_count))

    return image_data
