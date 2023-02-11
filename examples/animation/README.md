# Animation

## Setup

1. Clone `git@github.com:google-research/frame-interpolation.git`, and then install requirements listed in `frame-interpolation/requirements.txt`.
```
git clone git@github.com:google-research/frame-interpolation.git
pip install -r frame-interpolation/requirements.txt
```
2. Download [`saved_model` folder](https://drive.google.com/drive/folders/1i9Go1YI2qiFWeT5QtywNFmYAA74bhXWj?usp=sharing) to this folder.

## Example

Before running, `mkdir -p tmp`.

```
python -u img2img_animation.py \
--prompt_start 'Gregory Manchess style, a kitchen, yellow wall, wooden floor, wooden table and cupboard, vases of flowers' \
--prompt_end 'Gregory Manchess style, rain forest, birds in trees | Gregory Manchess style, a city in the future, street view | Gregory Manchess style, a fantasy landscape, mountains, river' \
--init_image finalscene-impasto-extrastrength-blurred.jpg \
--width 1024 --height 576 --prompt_strength 0.9 --num_animation_frames 10 --guidance_scale 12 --film_interpolation --intermediate_output --seed 0
```
