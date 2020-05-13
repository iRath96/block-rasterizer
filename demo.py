from rasterizer import Rasterizer, JarFileProvider
from PIL import Image

WIDTH = 256
HEIGHT = WIDTH
SPACER = 16

file_provider = JarFileProvider(
    "/Users/alex/Library/Application Support/minecraft/versions/20w19a/20w19a.jar"
)

rasterizer = Rasterizer(
    file_provider,
    (WIDTH, HEIGHT)
)

blocks = [
    rasterizer.render_block("beacon"),
    rasterizer.render_block("tnt"),
    rasterizer.render_block("lectern", "facing=south"),
    rasterizer.render_block("hopper", "facing=south"),
    rasterizer.render_block("campfire", "facing=north,lit=true"),
    rasterizer.render_block(
        "brewing_stand",
        "has_bottle_0=true,has_bottle_1=false,has_bottle_2=false"
    ),
    rasterizer.render_block("dead_brain_coral_fan")
]

banner = Image.new("RGBA", (
    len(blocks) * (WIDTH + SPACER) - SPACER,
    HEIGHT
))

for i,block in enumerate(blocks):
    banner.paste(block, (i * (WIDTH + SPACER), 0))

banner.save("demo/banner.png")
