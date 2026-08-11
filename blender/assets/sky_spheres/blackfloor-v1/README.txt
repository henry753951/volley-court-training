Volleyball Arena Skyball Pack

Files:
- images/arena_01.png ... arena_10.png
- arenas.yaml

Blender quick setup:
1. World Properties > Color > Environment Texture.
2. Load one PNG.
3. Projection: Equirectangular.
4. Use Texture Coordinate + Mapping nodes if yaw rotation is needed.
5. The bottom black region is intentional and should be covered/replaced visually by your Blender-built floor/court geometry.

Important:
- These are 8-bit PNG panoramas, suitable as background/environment textures.
- They are not calibrated HDRIs and should not be treated as physically accurate illumination maps.
- arenas.yaml contains semantic placement hints only, not metric camera calibration.
