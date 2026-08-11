@echo off
setlocal
cd /d "%~dp0.."
if not defined BLENDER_EXE set "BLENDER_EXE=blender.exe"
"%BLENDER_EXE%" ^
  --python blender\sky_sphere_calibrator.py ^
  -- ^
  --config blender\randomization-config.yaml ^
  --assets blender\assets\sky_spheres\blackfloor-v1 ^
  --expected-assets 10 ^
  --seed 20260811
endlocal
