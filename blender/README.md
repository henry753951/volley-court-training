# Blender synthetic court36 dataset

This directory contains the procedural Blender 5.2 scene and renderer for the
canonical 36-point volleyball-court pose contract.

The generator creates a regulation 9 m x 18 m court, center net, poles,
antennae, glossy/noisy floor materials, randomized arena lighting, simple
mannequin players, signboards and other occluders. Court lines, service/attack
line extensions and generic event lettering are packed material masks: they
are flat decals, not raised geometry. The net is a transparent 10 cm grid mask
rather than a solid plane.

Orange/coral playing zones with teal or blue surrounds dominate the weighted
palette distribution, matching common international broadcasts; wood and
red/navy variants are deliberately uncommon. People and props can occur both
inside and outside the court. Arena backgrounds independently sample four
profiles: the original empty exterior, a red championship arena, a steel-wall
small gym with usually bilateral bleachers, and a yellow double-grandstand
arena with roof trusses and media boxes. Grandstand crowd density varies per
image. Fictional UV-mapped sponsor boards use the packed
`assets/fictional-sponsor-led-atlas-v1.png` atlas generated for this project;
no real sponsor or tournament logo is copied.

An optional fifth profile uses a 2:1 equirectangular arena panorama as the
Blender World background. Its panorama loading, rotation and strength logic is
isolated in `volleyball_synthetic/sky_sphere.py`. When this profile is active,
all generated 3D arena shells, spectators and advertising LED boards are
hidden; only the panorama supplies the distant arena and audience. Its normal
weighted profile remains zero; the combined runner schedules it explicitly and exactly.

Camera position, focal length, sensor width and principal-point shift vary per
image. A championship-main mode is strongly weighted toward the centered-net,
full-court sideline composition used by indoor volleyball broadcasts and must
retain at least 28 court keypoints. In addition to high broadcast, endline,
corner and handheld views, the camera samples both ends of the court for
wide-server and server-close-up shots. A separate net-aligned broadcast mode
keeps the camera laterally outside the sideline and within 0.15 m of the net
line. Camera height stays only 0.15-0.85 m above the selected 2.24 m or 2.43 m
net, then pans and zooms 44-76 mm toward one of the four server corners so only
a small court corner remains in frame. These partial views intentionally write
off-screen court points as visibility `0`.

Floor roughness, exposure and lamp intensity still vary, but direct area-light
specular discs are disabled. This preserves subdued indoor reflections without
the unrealistic blown-out white circles produced by an exposed studio light.
The current YAML preset uses a brighter broadcast range, lower randomized
court/surround roughness, randomized clear-coat response and Eevee screen-trace
ray tracing so polished-floor highlights and reflections are more visible.

Dataset renders also randomize capture quality per image. With the default
1280x720 maximum they sample 854x480, 960x540, 1024x576 and 1280x720 while
preserving the camera aspect ratio. JPEG quality, small warm/cool or
green/magenta casts, white balance, gamma, sensor grain, vignette, chromatic
shift and usually broad-focus directional blur vary independently. Every applied
capture profile is stored in that image's metadata. Use
`--fixed-capture-quality` to disable these effects, or
`--minimum-resolution-y` to change the minimum output height.

Directional blur includes both ordinary camera motion and occasional stronger
horizontal `whip_pan` blur. Full-frame defocus is now only 2 percent; 64
percent remains unblurred because fixed volleyball broadcast cameras normally
keep the court and venue legible. About 18 percent of frames receive a generic
fictional broadcast score UI after camera blur/noise, so the overlay remains
sharp like a real production graphic. Any court point under its opaque screen
region is exported as visibility `1` with `occluded_by: broadcast_ui`.

Up to 32 of 36 reusable foreground mannequins may render at once. Besides
normal match spacing, scene randomization includes net clusters, sideline
clusters, two-team huddles, and server-focused shots with extra court and
sideline clutter. These people are separate from the randomized spectators in
the arena stands.

Visibility is exported in Ultralytics YOLO Pose format:

- `0`: outside the image or not projectable; coordinates are written as zero.
- `1`: projected location is known but a rendered object blocks the camera ray.
- `2`: projected and directly visible.

The 36 names and indices are identical to the manually annotated dataset. The
first ten points are semantic anchors; indices 10-35 are the one-third and
two-third subdivisions of the same 13 court segments.

## Files

- `volleyball-court-synthetic.blend`: generated project file, ready to inspect.
- `generate_volleyball_dataset.py`: thin backward-compatible Blender CLI entry.
- `volleyball_synthetic/config.py`: YAML parsing and weighted sampling helpers.
- `volleyball_synthetic/arena_assets.py`: calibrated panorama discovery and
  optional user-override merging.
- `volleyball_synthetic/constants.py`: immutable canonical 36-point geometry.
- `volleyball_synthetic/scene_builder.py`: materials, court/net/arena geometry and
  reusable Blender scene construction.
- `volleyball_synthetic/randomization.py`: per-frame camera, people, props,
  arena, material and lighting randomization.
- `volleyball_synthetic/capture.py`: resolution, compression, color, blur,
  noise and fictional broadcast UI post-processing.
- `volleyball_synthetic/labels.py`: projection, occlusion and YOLO Pose export.
- `volleyball_synthetic/dataset.py`: render loop, metadata and manifest output.
- `volleyball_synthetic/sky_sphere.py`: isolated equirectangular World
  background implementation.
- `render_sky_sphere_preview.py`: forced-profile one-frame experiment runner.
- `generate_sky_sphere_dataset.py`: balanced sky-sphere-only dataset runner;
  keeps the normal camera, people, lighting and capture randomization while
  disabling generated 3D arena shells, audience and advertising LED boards.
- `generate_combined_arena_dataset.py`: exact, shuffled mixture of all ten
  calibrated panorama arenas and all four generated virtual arena profiles.
- `sky_sphere_calibrator.py`: Blender-native live calibration panel for
  panorama rotation, horizon, apparent distance and brightness.
- `randomization-config.yaml`: all selection weights, numeric ranges, render
  defaults and human-editable star ratings used by the generator.
- `visualize_yolo_pose.py`: label overlay/contact-sheet validator.

## Render the sky-sphere experiment

```powershell
Set-Location path\to\volley-court-training

blender `
  --background `
  --python blender\render_sky_sphere_preview.py `
  -- `
  --config blender\randomization-config.yaml `
  --output artifacts\blender-skysphere-experiment-v1\preview.jpg `
  --blend blender\volleyball-court-skysphere-experiment.blend `
  --seed 20260811
```

This test runner forces `sky_sphere`, defaults to a lower
`handheld_sideline` camera so the panorama horizon remains visible, packs the
selected panorama into the experiment `.blend`, and fails if any generated
advertising LED board remains visible. Pass `--camera-mode championship_main`
or another configured mode to inspect its interaction with the same panorama.

## Generate 50 images from the ten black-floor sky spheres

```powershell
Set-Location path\to\volley-court-training

blender `
  --background `
  --python blender\generate_sky_sphere_dataset.py `
  -- `
  --assets blender\assets\sky_spheres\blackfloor-v1 `
  --output datasets\court36-synthetic-skysphere-blackfloor-50-v1 `
  --count 50 `
  --expected-assets 10 `
  --seed 20260811
```

With 50 images and ten assets, each sky sphere is rendered exactly five times.
The normal YAML still controls court colors and reflections, camera modes,
people and occluders, light intensity, 480-720p output, JPEG compression,
filters, white balance, sensor noise, motion blur and fictional broadcast UI.
The specialized manifest verifies that generated arena backgrounds, audience
and advertising LED counts remain zero in every image.

## Calibrate the ten sky spheres interactively

Use Blender itself instead of a Three.js approximation so the preview uses the
same World mapping and camera projection as the final renderer:

Double-click `blender/open-sky-sphere-calibrator.cmd`, or run the equivalent
PowerShell command below. Do not open `volleyball-court-synthetic.blend`
afterward: that is the older saved virtual-arena inspection scene and replacing
the generated calibration scene removes the live panel.

```powershell
Set-Location path\to\volley-court-training

blender `
  --python blender\sky_sphere_calibrator.py `
  -- `
  --config blender\randomization-config.yaml `
  --assets blender\assets\sky_spheres\blackfloor-v1 `
  --expected-assets 10 `
  --seed 20260811
```

The script opens the generated court in camera/rendered view. Open the `N`
sidebar if it is hidden, then choose the `Court Calib` tab. `Previous` and
`Next` switch arenas. The main controls are yaw correction, temporary preview
rotation, horizon offset, apparent distance, projection X/Y, and brightness
minimum/maximum. `Camera View + Lock` does not require a numeric keypad and
also lets normal viewport navigation move the locked camera.

`Preview rotation` is deliberately not persisted. `Save Overrides YAML`
writes
`assets/sky_spheres/blackfloor-v1/arena-calibration-overrides.yaml`; both the
50-image and combined 2,000-image runners automatically merge this file over
the reviewed `arena-calibration.yaml`. A World environment is physically at
infinite distance, so a literal sphere radius would not alter the result. The
panel's apparent-distance control changes the effective vertical projection
scale, which is the parameter that actually changes perceived venue size.

The original `arenas.yaml` is retained as source provenance. Its repeated
`recommended_origin_uv: 0.72` is not used as calibration: visual review found
the ten floor boundaries vary approximately from 0.535 to 0.620.

## Randomization YAML

The generator loads `blender/randomization-config.yaml` by default. Blender's
bundled Python does not need PyYAML: `volleyball_synthetic/config.py` includes
a strict parser for the mapping/list/scalar subset used by this file, and
automatically uses `yaml.safe_load` when PyYAML is available.

`selection_weight_mode: exact` uses each entry's numeric `weight`, preserving
the documented current percentages. Change it to `stars` to ignore those
numbers and derive weights from `☆☆☆☆☆` through `★★★★★` using the
`star_weights` table at the top of the YAML. Numeric ranges such as camera
positions, player counts, lighting, roughness, blur and JPEG quality are read
directly from the same file. Explicit command-line values override the
`generation` section without changing the YAML.

## Generate the combined 2000-image dataset

This is the primary mixed command. It creates exactly 1,000 sky-sphere samples
(100 from each of ten panoramas) plus 1,000 generated virtual-arena samples.
The virtual half keeps the YAML ratio: 300 championship, 240 empty exterior,
240 small gym and 220 yellow grandstand. The complete schedule is shuffled by
seed before rendering; all other YAML camera, player, distraction, lighting,
reflection and capture-quality parameters still randomize per image.

```powershell
Set-Location path\to\volley-court-training

blender `
  --background `
  --python blender\generate_combined_arena_dataset.py `
  -- `
  --config blender\randomization-config.yaml `
  --assets blender\assets\sky_spheres\blackfloor-v1 `
  --output datasets\court36-synthetic-combined-2000-v1 `
  --count 2000 `
  --sky-sphere-count 1000 `
  --expected-assets 10 `
  --seed 20260811 `
  --resolution-x 1280 `
  --resolution-y 720 `
  --minimum-resolution-y 480 `
  --samples 48
```

The generator refuses to write into a dataset that already contains images.
Use `--overwrite` only when replacement is intentional. Built-in YOLO
`fliplr` should remain zero when this dataset is mixed with the adaptive
sideline/endline annotated dataset.

The panorama PNGs already contain baked audience, walls and edge boards. In
sky-sphere samples the generator therefore disables its own duplicate 3D
audience, walls and LED objects; it does not erase baked pixels from the source
panoramas. Virtual-arena samples continue to use generated 3D arena elements.

The older `generate_volleyball_dataset.py` command remains available for a
virtual-only weighted dataset, and `generate_sky_sphere_dataset.py` remains
available for a panorama-only balanced dataset.

## Visualize labels

```powershell
path\to\volley-court-training\.venv\Scripts\python.exe `
  blender\visualize_yolo_pose.py `
  --dataset datasets\court36-synthetic-blender-v1 `
  --output artifacts\court36-synthetic-blender-v1-labels `
  --limit 12
```
