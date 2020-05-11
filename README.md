# Minecraft Block Rasterizer
<center>
<img height="64" src="demo/banner.png" />
</center>

This lightweight library produces images of blocks using Minecrafts block model files.
You could use this to generate isometric views of Minecraft worlds – without the time-consuming hassle of having to write manual rendering code for each block, which needs to be extended whenever a new snapshot comes along (I'm looking at you, [Minecraft-Overviewer](https://github.com/overviewer/Minecraft-Overviewer)!)

## Dependecies
* numpy
* pillow
* _…yup, that's it!_

## Features
* Supports **transparency** (for example in billboards)
* Supports **multipart objects** (brewing stand, fences, etc.)
* Supports **rotated elements** (lantern, levers, etc.)

**Warning:** Some blocks do not have model files (most notably chests). Some blocks have model files, but miss certain elements (for instance the bell model only contains the posts).

## Usage
```python
from rasterizer import Rasterizer

renderer = Rasterizer(
  "{path to Minecraft}/20w19a/20w19a.jar",
  dimensions=(240, 240)
)

image = renderer.render_block(
  block_name="campfire",
  variant="facing=north,lit=true"
)

image.save("demo/campfire.png")
```
