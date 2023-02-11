# Animation

## Setup

1. Clone `git@github.com:google-research/frame-interpolation.git` to this folder, and then install requirements listed in `frame-interpolation/requirements.txt`.
```
git clone git@github.com:google-research/frame-interpolation.git
pip install -r frame-interpolation/requirements.txt
```
2. Download [`saved_model` folder](https://drive.google.com/drive/folders/1i9Go1YI2qiFWeT5QtywNFmYAA74bhXWj?usp=sharing) to this folder.
3. Additional dependencies: `av`.
```
pip install av
```

## Example

```
python -u img2img_animation.py \
--prompts 'Gregory Manchess style, a kitchen, yellow wall, wooden floor, wooden table and cupboard, vases of flowers' \
'Gregory Manchess style, rain forest, birds in trees' \
'Gregory Manchess style, a city in the future, street view' \
'Gregory Manchess style, a fantasy landscape, mountains, river' \
--init_image data/finalscene-impasto-extrastrength-blurred.jpg \
--width 1024 --height 576 --strength 0.8 --num_animation_frames 10 --guidance_scale 5 --film_interpolation --seed 0 \
--intermediate_output --output_path results
```
