from rasterizer import Rasterizer
from PIL import Image

WIDTH = 256
HEIGHT = WIDTH
SPACER = 16

renderer = Rasterizer(
    "/Users/alex/Library/Application Support/minecraft/versions/20w19a/20w19a.jar",
    (WIDTH, HEIGHT)
)

blocks = [
    renderer.render_block("beacon"),
    renderer.render_block("tnt"),
    renderer.render_block("lectern", "facing=south"),
    renderer.render_block("hopper", "facing=south"),
    renderer.render_block("campfire", "facing=north,lit=true"),
    renderer.render_block(
        "brewing_stand",
        "has_bottle_0=true,has_bottle_1=false,has_bottle_2=false"
    ),
    renderer.render_block("dead_brain_coral_fan")
]

banner = Image.new("RGBA", (
    len(blocks) * (WIDTH + SPACER) - SPACER,
    HEIGHT
))
for i,block in enumerate(blocks):
    banner.paste(block, (i * (WIDTH + SPACER), 0))
banner.save("demo/banner.png")
