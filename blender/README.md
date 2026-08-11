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
shift and slight defocus/directional blur vary independently. Every applied
capture profile is stored in that image's metadata. Use
`--fixed-capture-quality` to disable these effects, or
`--minimum-resolution-y` to change the minimum output height.

Directional blur includes both ordinary camera motion and occasional stronger
horizontal `whip_pan` blur. About 18 percent of frames receive a generic
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
- `generate_volleyball_dataset.py`: Blender-side scene and dataset generator.
- `randomization-config.yaml`: all selection weights, numeric ranges, render
  defaults and human-editable star ratings used by the generator.
- `visualize_yolo_pose.py`: label overlay/contact-sheet validator.

## Randomization YAML

The generator loads `blender/randomization-config.yaml` by default. Blender's
bundled Python does not need PyYAML: the script includes a strict parser for
the mapping/list/scalar subset used by this file, and automatically uses
`yaml.safe_load` when PyYAML is available.

`selection_weight_mode: exact` uses each entry's numeric `weight`, preserving
the documented current percentages. Change it to `stars` to ignore those
numbers and derive weights from `☆☆☆☆☆` through `★★★★★` using the
`star_weights` table at the top of the YAML. Numeric ranges such as camera
positions, player counts, lighting, roughness, blur and JPEG quality are read
directly from the same file. Explicit command-line values override the
`generation` section without changing the YAML.

## Generate 2000 images

Run this from PowerShell after inspecting the preview scene:

```powershell
Set-Location H:\Repos\volley-court-training

& 'E:\GameLibrary\SteamLibrary\steamapps\common\Blender\blender.exe' `
  --background `
  blender\volleyball-court-synthetic.blend `
  --python blender\generate_volleyball_dataset.py `
  -- `
  --config blender\randomization-config.yaml `
  --output datasets\court36-synthetic-blender-yaml-v1 `
  --count 2000 `
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

## Visualize labels

```powershell
H:\Repos\volley-court-training\.venv\Scripts\python.exe `
  blender\visualize_yolo_pose.py `
  --dataset datasets\court36-synthetic-blender-v1 `
  --output artifacts\court36-synthetic-blender-v1-labels `
  --limit 12
```
