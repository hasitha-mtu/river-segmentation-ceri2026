import glob
import os

def update_file_names(file_names, suffix):
    for file_name in file_names:
        [name, extension] = file_name.split(sep=".")
        print(name)
        new_file_name = f'{name}_{suffix}.{extension}'
        os.rename(file_name, new_file_name)


if __name__ == "__main__":
    # DJI_20250324092908_0001_V
    # DJI_20250625120445_0063_V
    # DJI_20250728094218_0071_V
    # DJI_20250728094105_0007_V
    image_path = 'data/raw/images/*.png'
    mask_path = 'data/raw/masks/*.png'
    images = glob.glob(image_path)
    print(f'image_path count: {len(images)}')
    masks = glob.glob(mask_path)
    print(f'mask_path count: {len(masks)}')

    # # March images
    # march_image_path = 'data/raw/images/DJI_202503*.png'
    # march_mask_path = 'data/raw/masks/DJI_202503*.png'
    # march_images = glob.glob(march_image_path)
    # print(f'march_images count: {len(march_images)}')
    # march_masks = glob.glob(march_mask_path)
    # print(f'march_masks count: {len(march_masks)}')
    # update_file_names(march_images, 'March')
    # update_file_names(march_masks, 'March')

    # June images
    # june_image_path = 'data/raw/images/DJI_20250625*.png'
    # june_mask_path = 'data/raw/masks/DJI_20250625*.png'
    # june_images = glob.glob(june_image_path)
    # print(f'june_images count: {len(june_images)}')
    # june_masks = glob.glob(june_mask_path)
    # print(f'june_masks count: {len(june_masks)}')
    # update_file_names(june_images, 'June')
    # update_file_names(june_masks, 'June')

    # July images
    # july_image_path = 'data/raw/images/DJI_202507*.png'
    july_mask_path = 'data/raw/masks/DJI_202507*.png'
    # july_images = glob.glob(july_image_path)
    # print(f'july_images count: {len(july_images)}')
    july_masks = glob.glob(july_mask_path)
    print(f'july_masks count: {len(july_masks)}')
    # update_file_names(july_images, 'July')
    update_file_names(july_masks, 'July')


    images = glob.glob(image_path)
    print(f'image_path count: {len(images)}')
    masks = glob.glob(mask_path)
    print(f'mask_path count: {len(masks)}')

